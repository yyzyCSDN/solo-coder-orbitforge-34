from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping

from orbitforge.core.state import TimeWindow
from orbitforge.power.battery import BatteryState
from orbitforge.thermal.nodal import ThermalNode, step_nodes


@dataclass(frozen=True)
class ResourceInterval:
    """Solar-array, eclipse and thermal environment over one constant interval."""

    start_tai_s: float
    end_tai_s: float
    solar_array_w: float = 0.0
    base_load_w: float = 0.0
    sunlight_factor: float = 1.0
    environment_k: float = 293.0
    in_eclipse: bool | None = None

    def __post_init__(self):
        if self.end_tai_s < self.start_tai_s:
            raise ValueError('negative resource interval')
        if self.in_eclipse:
            if self.sunlight_factor not in (0.0, 1.0):
                raise ValueError('in_eclipse and sunlight_factor disagree')
            object.__setattr__(self, 'sunlight_factor', 0.0)
        if not 0.0 <= self.sunlight_factor <= 1.0:
            raise ValueError('sunlight_factor must be in [0, 1]')

    @property
    def window(self):
        return TimeWindow(self.start_tai_s, self.end_tai_s)

    @property
    def eclipse(self):
        return self.sunlight_factor == 0.0

    @property
    def solar_generation_w(self):
        return self.solar_array_w * self.sunlight_factor


@dataclass(frozen=True)
class ResourceViolation:
    domain: str
    kind: str
    time_tai_s: float
    value: float
    limit: float
    activity: str | None = None
    node: str | None = None

    @property
    def message(self):
        if self.domain == 'battery':
            return (
                f'{self.kind} violated at t={self.time_tai_s}: '
                f'{self.value:.6g} vs limit {self.limit:.6g}'
            )
        return (
            f'{self.node or self.domain} {self.kind} violated at '
            f't={self.time_tai_s}: {self.value:.6g} K vs limit {self.limit:.6g} K'
        )


class ResourceViolationError(ValueError):
    def __init__(self, activity, violations, simulation=None):
        self.activity = activity
        self.violations = violations
        self.simulation = simulation
        activity_name = getattr(activity, 'name', activity)
        messages = '; '.join(v.message for v in violations)
        super().__init__(f'resource violation for {activity_name}: {messages}')


@dataclass
class ResourceModel:
    """Battery, solar-array, eclipse and thermal model for one planning period."""

    intervals: list[ResourceInterval]
    battery: BatteryState | None = None
    thermal_nodes: list[ThermalNode] | None = None
    conductances: list[tuple[str, str, float]] | None = None
    radiation_coefficients: Mapping[str, float] | None = None
    thermal_limits_k: Mapping[str, tuple[float, float]] | None = None
    min_soc: float = 0.0
    max_soc: float = 1.0
    thermal_step_s: float | None = None

    def __post_init__(self):
        if not 0.0 <= self.min_soc <= self.max_soc <= 1.0:
            raise ValueError('invalid SOC limits')
        if self.thermal_step_s is not None and self.thermal_step_s <= 0.0:
            raise ValueError('thermal_step_s must be positive')
        previous = None
        for interval in sorted(self.intervals, key=lambda i: i.start_tai_s):
            if previous is not None and interval.start_tai_s < previous.end_tai_s:
                raise ValueError('resource intervals may not overlap')
            previous = interval


@dataclass
class ResourceSimulation:
    ok: bool
    violations: list[ResourceViolation]
    battery: BatteryState | None
    thermal_nodes: dict[str, ThermalNode]
    samples: list[dict[str, Any]]

    @property
    def minimum_energy_wh(self):
        return min((s['energy_wh'] for s in self.samples if s['energy_wh'] is not None), default=None)

    @property
    def minimum_soc(self):
        return min((s['soc'] for s in self.samples if s['soc'] is not None), default=None)


def _activity_window(activity):
    return activity.window


def _activity_load_w(activity):
    return float(getattr(activity, 'load_w', 0.0) or 0.0)


def _activity_heat_w(activity, node_name, single_thermal_node):
    heat = getattr(activity, 'heat_w', 0.0) or 0.0
    if isinstance(heat, Mapping):
        return float(heat.get(node_name, 0.0))
    if single_thermal_node:
        return float(heat)
    if heat:
        raise ValueError('numeric heat_w requires a model with exactly one thermal node')
    return 0.0


def _ordered_activities(activities):
    return sorted(activities, key=lambda a: (_activity_window(a).start_tai_s, a.name))


def _active_activities(activities, t):
    return [
        a
        for a in activities
        if _activity_window(a).start_tai_s <= t < _activity_window(a).end_tai_s
    ]


def _interval_at(intervals, t):
    containing = [
        i
        for i in intervals
        if i.start_tai_s <= t < i.end_tai_s
        or (t == i.end_tai_s == max(x.end_tai_s for x in intervals))
    ]
    if containing:
        return sorted(containing, key=lambda i: i.start_tai_s)[-1]
    return ResourceInterval(t, t)


def _interpolate_crossing(t0, t1, value0, value1, limit):
    if value1 == value0:
        return t0
    fraction = (limit - value0) / (value1 - value0)
    fraction = min(1.0, max(0.0, fraction))
    return t0 + fraction * (t1 - t0)


def _cause_for_time(activities, t):
    active = _active_activities(activities, t)
    if active:
        return max(active, key=lambda a: a.name).name
    ended = [a for a in activities if _activity_window(a).end_tai_s <= t]
    if ended:
        return max(ended, key=lambda a: (_activity_window(a).end_tai_s, a.name)).name
    return None


def simulate_resources(
    activities,
    model: ResourceModel,
    start_tai_s: float | None = None,
    end_tai_s: float | None = None,
    cause_activity: str | None = None,
):
    """Propagate all resources through the complete activity/environment timeline."""

    activities = list(activities)
    intervals = sorted(model.intervals, key=lambda i: i.start_tai_s)
    bounds = [i.start_tai_s for i in intervals] + [i.end_tai_s for i in intervals]
    bounds.extend(
        t
        for a in activities
        for t in (_activity_window(a).start_tai_s, _activity_window(a).end_tai_s)
    )
    if not bounds:
        raise ValueError('cannot determine resource simulation horizon')

    start = min(bounds) if start_tai_s is None else start_tai_s
    end = max(bounds) if end_tai_s is None else end_tai_s
    if end < start:
        raise ValueError('negative simulation horizon')

    violations: list[ResourceViolation] = []
    reported: set[tuple[str, str | None, str]] = set()

    def record(domain, kind, t, value, limit, node=None):
        key = (domain, node, kind)
        if key in reported:
            return
        reported.add(key)
        cause = cause_activity or _cause_for_time(activities, t)
        violations.append(ResourceViolation(domain, kind, t, value, limit, cause, node))

    battery = None if model.battery is None else replace(model.battery)
    original_nodes = list(model.thermal_nodes or [])
    nodes = [replace(node) for node in original_nodes]
    node_by_name = {node.name: node for node in nodes}
    original_by_name = {node.name: node for node in original_nodes}
    limits = model.thermal_limits_k or {}
    radiation = model.radiation_coefficients or {}
    single_node = len(nodes) == 1

    temperatures = {name: node.temperature_k for name, node in node_by_name.items()}

    if battery is not None:
        if battery.soc < model.min_soc:
            record('battery', 'min_soc', start, battery.soc, model.min_soc)
        if battery.soc > model.max_soc:
            record('battery', 'max_soc', start, battery.soc, model.max_soc)
    for name, temp in temperatures.items():
        if name in limits:
            low, high = limits[name]
            if temp < low:
                record('thermal', 'min_temperature', start, temp, low, name)
            if temp > high:
                record('thermal', 'max_temperature', start, temp, high, name)

    samples = [
        {
            't': start,
            'energy_wh': None if battery is None else battery.energy_wh,
            'soc': None if battery is None else battery.soc,
            'temperatures': dict(temperatures),
            'solar_generation_w': 0.0,
            'load_w': 0.0,
            'sunlight_factor': 1.0,
            'eclipse': False,
            'activities': (),
        }
    ]

    points = {start, end}
    for interval in intervals:
        if interval.end_tai_s > start and interval.start_tai_s < end:
            points.add(max(start, min(end, interval.start_tai_s)))
            points.add(max(start, min(end, interval.end_tai_s)))
    for activity in activities:
        window = _activity_window(activity)
        if window.end_tai_s > start and window.start_tai_s < end:
            points.add(max(start, min(end, window.start_tai_s)))
            points.add(max(start, min(end, window.end_tai_s)))

    for t0, t1 in zip(sorted(points), sorted(points)[1:]):
        duration = t1 - t0
        if duration <= 0.0:
            continue

        interval = _interval_at(intervals, t0)
        active = _active_activities(activities, t0)
        load_w = interval.base_load_w + sum(_activity_load_w(a) for a in active)

        energy_before = None if battery is None else battery.energy_wh
        if battery is not None:
            soc_before = battery.soc
            battery.step(interval.solar_generation_w, load_w, duration)
            soc_after = battery.soc
            low_limit = model.min_soc
            high_limit = model.max_soc
            if soc_after < low_limit and soc_before >= low_limit:
                crossing = _interpolate_crossing(
                    t0,
                    t1,
                    energy_before,
                    battery.energy_wh,
                    battery.capacity_wh * low_limit,
                )
                record('battery', 'min_soc', crossing, battery.energy_wh, battery.capacity_wh * low_limit)
            if soc_after > high_limit and soc_before <= high_limit:
                crossing = _interpolate_crossing(
                    t0,
                    t1,
                    energy_before,
                    battery.energy_wh,
                    battery.capacity_wh * high_limit,
                )
                record('battery', 'max_soc', crossing, battery.energy_wh, battery.capacity_wh * high_limit)

        for node in nodes:
            heat = sum(_activity_heat_w(a, node.name, single_node) for a in active)
            node.internal_w = original_by_name[node.name].internal_w + heat

        if nodes:
            requested_step = duration if model.thermal_step_s is None else model.thermal_step_s
            step_count = max(1, math.ceil(duration / requested_step))
            dt = duration / step_count
            for substep in range(step_count):
                sub_t0 = t0 + substep * dt
                before = dict(temperatures)
                step_nodes(nodes, model.conductances or [], interval.environment_k, radiation, dt)
                temperatures = {name: node.temperature_k for name, node in node_by_name.items()}
                for name, temp in temperatures.items():
                    if name not in limits:
                        continue
                    low, high = limits[name]
                    previous_temp = before[name]
                    if temp < low and previous_temp >= low:
                        fraction = 0.0 if temp == previous_temp else (low - previous_temp) / (temp - previous_temp)
                        crossing = sub_t0 + min(1.0, max(0.0, fraction)) * dt
                        record('thermal', 'min_temperature', crossing, temp, low, name)
                    if temp > high and previous_temp <= high:
                        fraction = 0.0 if temp == previous_temp else (high - previous_temp) / (temp - previous_temp)
                        crossing = sub_t0 + min(1.0, max(0.0, fraction)) * dt
                        record('thermal', 'max_temperature', crossing, temp, high, name)

        temperatures = {name: node.temperature_k for name, node in node_by_name.items()}
        samples.append(
            {
                't': t1,
                'energy_wh': None if battery is None else battery.energy_wh,
                'soc': None if battery is None else battery.soc,
                'temperatures': dict(temperatures),
                'solar_generation_w': interval.solar_generation_w,
                'load_w': load_w,
                'sunlight_factor': interval.sunlight_factor,
                'eclipse': interval.eclipse,
                'activities': tuple(a.name for a in active),
            }
        )

    violations.sort(key=lambda v: (v.time_tai_s, v.domain, v.node or ''))
    return ResourceSimulation(
        ok=not violations,
        violations=violations,
        battery=battery,
        thermal_nodes=node_by_name,
        samples=samples,
    )


def evaluate_resource_timeline(activities, model: ResourceModel, start_tai_s=None, end_tai_s=None):
    """Return the first activity whose prefix makes the complete timeline infeasible."""

    activities = list(activities)
    baseline = simulate_resources([], model, start_tai_s, end_tai_s)
    if baseline.violations:
        return baseline

    prefix = []
    for activity in _ordered_activities(activities):
        prefix.append(activity)
        simulation = simulate_resources(
            prefix,
            model,
            start_tai_s,
            end_tai_s,
            cause_activity=activity.name,
        )
        if simulation.violations:
            return simulation
    return simulate_resources(activities, model, start_tai_s, end_tai_s)

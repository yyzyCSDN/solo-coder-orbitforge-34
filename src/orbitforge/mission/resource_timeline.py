from __future__ import annotations
from dataclasses import dataclass, replace
from orbitforge.core.state import TimeWindow
from orbitforge.power.battery import BatteryState
from orbitforge.power.solar import panel_power_w
from orbitforge.thermal.nodal import ThermalNode, step_nodes

@dataclass(frozen=True)
class EclipseWindow:
    window: TimeWindow
    sunlight_fraction: float = 0.0

@dataclass(frozen=True)
class BusModel:
    battery_capacity_wh: float
    initial_energy_wh: float
    min_energy_wh: float = 0.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    panel_area_m2: float = 0.0
    panel_efficiency: float = 0.0
    panel_degradation: float = 1.0
    incidence_rad: float = 0.0
    baseline_load_w: float = 0.0

@dataclass(frozen=True)
class NodeModel:
    name: str
    heat_capacity_j_k: float
    initial_k: float
    internal_w: float = 0.0
    radiation_coeff: float = 0.0
    min_k: float | None = None
    max_k: float | None = None

@dataclass(frozen=True)
class ResourceModel:
    bus: BusModel
    nodes: tuple = ()
    conductances: tuple = ()
    eclipses: tuple = ()
    env_sunlit_k: float = 278.0
    env_eclipse_k: float = 100.0
    thermal_dt_s: float = 30.0

@dataclass(frozen=True)
class Sample:
    t_tai_s: float
    energy_wh: float
    temperatures_k: dict

@dataclass(frozen=True)
class Violation:
    resource: str
    sense: str
    window: TimeWindow
    worst: float
    limit: float
    causes: tuple = ()

    @property
    def margin(self):
        if self.sense == 'below_min':
            return self.worst - self.limit
        return self.limit - self.worst

@dataclass(frozen=True)
class SimulationResult:
    samples: tuple
    violations: tuple
    start_tai_s: float
    end_tai_s: float

    @property
    def ok(self):
        return not self.violations

    @property
    def final_energy_wh(self):
        return self.samples[-1].energy_wh

def simulate(activities, model, start_tai_s=None, end_tai_s=None, tail_s=0.0):
    activities = sorted(activities, key=lambda a: (a.window.start_tai_s, a.name))
    if start_tai_s is None:
        start_tai_s = min([a.window.start_tai_s for a in activities] + [e.window.start_tai_s for e in model.eclipses], default=0.0)
    if end_tai_s is None:
        end_tai_s = max([a.window.end_tai_s for a in activities] + [e.window.end_tai_s for e in model.eclipses], default=start_tai_s)
    end_tai_s += tail_s
    points = {start_tai_s, end_tai_s}
    for w in [a.window for a in activities] + [e.window for e in model.eclipses]:
        if start_tai_s < w.start_tai_s < end_tai_s:
            points.add(w.start_tai_s)
        if start_tai_s < w.end_tai_s < end_tai_s:
            points.add(w.end_tai_s)
    node_names = {n.name for n in model.nodes}
    for a in activities:
        if a.demand is not None and a.demand.heat_w and (a.demand.node not in node_names):
            raise ValueError(f'unknown thermal node: {a.demand.node}')
    if model.nodes and model.thermal_dt_s <= 0.0:
        raise ValueError('thermal_dt_s')
    battery = BatteryState(model.bus.battery_capacity_wh, model.bus.initial_energy_wh, model.bus.charge_efficiency, model.bus.discharge_efficiency)
    nodes = [ThermalNode(n.name, n.heat_capacity_j_k, n.initial_k, n.internal_w) for n in model.nodes]
    radiation = {n.name: n.radiation_coeff for n in model.nodes if n.radiation_coeff}
    generation_sunlit_w = panel_power_w(model.bus.panel_area_m2, model.bus.panel_efficiency, model.bus.incidence_rad, degradation=model.bus.panel_degradation)
    samples = [_sample(start_tai_s, battery, nodes)]
    bounds = sorted(points)
    for lo, hi in zip(bounds, bounds[1:]):
        mid = 0.5 * (lo + hi)
        fraction = _sunlight_fraction(model.eclipses, mid)
        generation_w = generation_sunlit_w * fraction
        load_w = model.bus.baseline_load_w
        heat_w = {n.name: n.internal_w for n in model.nodes}
        for a in activities:
            if a.demand is None or not (a.window.start_tai_s <= mid < a.window.end_tai_s):
                continue
            load_w += a.demand.load_w
            if a.demand.heat_w:
                heat_w[a.demand.node] += a.demand.heat_w
        environment_k = fraction * model.env_sunlit_k + (1.0 - fraction) * model.env_eclipse_k
        step_s = model.thermal_dt_s if nodes else hi - lo
        t = lo
        while t < hi - 1e-09:
            dt = min(step_s, hi - t)
            for n in nodes:
                n.internal_w = heat_w[n.name]
            battery.step(generation_w, load_w, dt)
            if nodes:
                step_nodes(nodes, model.conductances, environment_k, radiation, dt)
            t += dt
            samples.append(_sample(t, battery, nodes))
    return SimulationResult(tuple(samples), tuple(_scan(samples, model)), start_tai_s, end_tai_s)

def explain_violations(result, activities, model):
    if not result.violations:
        return result
    activities = list(activities)
    explained = []
    for v in result.violations:
        ranked = []
        for i, a in enumerate(activities):
            if a.window.start_tai_s >= v.window.end_tai_s:
                continue
            rest = activities[:i] + activities[i + 1:]
            without = simulate(rest, model, start_tai_s=result.start_tai_s, end_tai_s=result.end_tai_s)
            margin = _window_margin(without, v)
            if margin is None:
                continue
            restored = margin - v.margin
            if restored > 1e-09:
                ranked.append((a.name, restored))
        ranked.sort(key=lambda row: (-row[1], row[0]))
        explained.append(replace(v, causes=tuple((name for name, _ in ranked))))
    return replace(result, violations=tuple(explained))

def _sample(t, battery, nodes):
    return Sample(t, battery.energy_wh, {n.name: n.temperature_k for n in nodes})

def _sunlight_fraction(eclipses, t):
    fraction = 1.0
    for e in eclipses:
        if e.window.start_tai_s <= t < e.window.end_tai_s:
            fraction = min(fraction, e.sunlight_fraction)
    return fraction

def _scan(samples, model):
    tracks = [('battery', 'below_min', model.bus.min_energy_wh)]
    for n in model.nodes:
        if n.min_k is not None:
            tracks.append((n.name, 'below_min', n.min_k))
        if n.max_k is not None:
            tracks.append((n.name, 'above_max', n.max_k))
    out = []
    for resource, sense, limit in tracks:
        run = []
        run_start = None
        prev = None
        for s in samples:
            value = s.energy_wh if resource == 'battery' else s.temperatures_k[resource]
            bad = value < limit if sense == 'below_min' else value > limit
            if bad:
                if not run:
                    run_start = _crossing(prev, (s.t_tai_s, value), limit) if prev else s.t_tai_s
                run.append((s.t_tai_s, value))
            elif run:
                out.append(_close_run(resource, sense, limit, run, run_start))
                run = []
            prev = (s.t_tai_s, value)
        if run:
            out.append(_close_run(resource, sense, limit, run, run_start))
    return out

def _crossing(prev, cur, limit):
    (t0, v0), (t1, v1) = (prev, cur)
    if v1 == v0:
        return t1
    return t0 + (limit - v0) * (t1 - t0) / (v1 - v0)

def _close_run(resource, sense, limit, run, start):
    values = [v for _, v in run]
    worst = min(values) if sense == 'below_min' else max(values)
    return Violation(resource, sense, TimeWindow(start, run[-1][0]), worst, limit)

def _window_margin(result, violation):
    values = []
    for s in result.samples:
        if violation.window.start_tai_s - 1e-09 <= s.t_tai_s <= violation.window.end_tai_s + 1e-09:
            values.append(s.energy_wh if violation.resource == 'battery' else s.temperatures_k[violation.resource])
    if not values:
        return None
    if violation.sense == 'below_min':
        return min(values) - violation.limit
    return violation.limit - max(values)

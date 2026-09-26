from orbitforge.core.state import TimeWindow
from orbitforge.mission.resource_timeline import (
    ResourceInterval,
    ResourceModel,
    ResourceViolationError,
)
from orbitforge.mission.scheduler import greedy_schedule
from orbitforge.mission.timeline import Activity, Timeline
from orbitforge.power.battery import BatteryState
from orbitforge.thermal.nodal import ThermalNode


def _model():
    intervals = [
        ResourceInterval(
            start_tai_s=0,
            end_tai_s=1800,
            solar_array_w=1000,
            base_load_w=90,
            sunlight_factor=1.0,
            environment_k=290,
        ),
        ResourceInterval(
            start_tai_s=1800,
            end_tai_s=3600,
            solar_array_w=1000,
            base_load_w=90,
            sunlight_factor=0.0,
            environment_k=290,
        ),
    ]
    return ResourceModel(
        intervals=intervals,
        battery=BatteryState(capacity_wh=1000, energy_wh=300),
        thermal_nodes=[
            ThermalNode('payload', heat_capacity_j_k=3600, temperature_k=290),
            ThermalNode('electronics', heat_capacity_j_k=36_000, temperature_k=290),
        ],
        conductances=[('payload', 'electronics', 10.0)],
        radiation_coefficients={},
        thermal_limits_k={'payload': (270, 305)},
        min_soc=0.30,
        thermal_step_s=10,
    )


def test_baseline_timeline_is_feasible():
    simulation = Timeline(_model()).evaluate_resources()
    assert simulation.ok


def test_high_power_activity_is_rejected_for_later_eclipse_violations():
    timeline = Timeline(_model())
    activity = Activity(
        'high-power-payload',
        TimeWindow(600, 1800),
        'payload',
        priority=10,
        load_w=1240,
        heat_w={'electronics': 500.0},
    )

    try:
        timeline.add(activity)
    except ResourceViolationError as exc:
        violations_by_domain = {v.domain: v for v in exc.violations}
    else:
        raise AssertionError('high-power activity was accepted from start margin only')

    assert 'battery' in violations_by_domain
    assert 'thermal' in violations_by_domain

    battery_violation = violations_by_domain['battery']
    assert battery_violation.activity == activity.name
    assert battery_violation.time_tai_s > 1800
    assert battery_violation.kind == 'min_soc'

    thermal_violation = violations_by_domain['thermal']
    assert thermal_violation.activity == activity.name
    assert thermal_violation.node == 'payload'
    assert activity.window.end_tai_s < thermal_violation.time_tai_s < 3000


def test_greedy_scheduler_records_resource_rejection():
    safe = Activity(
        'safe-payload',
        TimeWindow(0, 600),
        'safe-instrument',
        priority=5,
        load_w=100,
        heat_w=0,
    )
    expensive = Activity(
        'high-power-payload',
        TimeWindow(600, 1800),
        'high-power-instrument',
        priority=10,
        load_w=1240,
        heat_w={'electronics': 500.0},
    )

    timeline, rejected = greedy_schedule([safe, expensive], _model())

    assert expensive in rejected
    assert [a.name for a in timeline.activities] == ['safe-payload']
    rejected_activity, simulation = timeline.resource_rejections[0]
    assert rejected_activity is expensive
    assert not simulation.ok
    assert {v.activity for v in simulation.violations} == {expensive.name}

from orbitforge.core.state import TimeWindow
from orbitforge.mission.timeline import Activity, ResourceDemand
from orbitforge.mission.resource_timeline import (
    BusModel, EclipseWindow, NodeModel, ResourceModel, simulate, explain_violations)
from orbitforge.mission.scheduler import resource_checked_schedule
from orbitforge.power.battery import BatteryState, eclipse_margin_wh

def _power_model():
    return ResourceModel(
        bus=BusModel(battery_capacity_wh=200.0, initial_energy_wh=200.0, min_energy_wh=40.0,
                     panel_area_m2=2.0, panel_efficiency=0.3, baseline_load_w=50.0),
        eclipses=(EclipseWindow(TimeWindow(2500.0, 9000.0)),))

def _payload():
    return Activity('payload_burst', TimeWindow(1000.0, 2200.0), 'cam', 10, ResourceDemand(load_w=1200.0))

def test_start_margin_alone_would_clear_the_payload():
    at_start = BatteryState(200.0, 200.0)
    assert at_start.energy_wh >= 40.0
    assert eclipse_margin_wh(at_start, 6500.0, 50.0) > 0.0

def test_payload_breaks_battery_inside_later_eclipse():
    model = _power_model()
    assert simulate([], model).ok
    result = simulate([_payload()], model)
    assert not result.ok
    v = result.violations[0]
    assert v.resource == 'battery' and v.sense == 'below_min'
    assert v.window.start_tai_s > 2500.0
    assert v.window.end_tai_s <= 9000.0
    explained = explain_violations(result, [_payload()], model)
    assert explained.violations[0].causes == ('payload_burst',)

def test_scheduler_rejects_payload_and_keeps_light_activity():
    model = _power_model()
    downlink = Activity('downlink', TimeWindow(1000.0, 2200.0), 'tx', 5, ResourceDemand(load_w=50.0))
    result = resource_checked_schedule([_payload(), downlink], model)
    names = [a.name for a in result.timeline.activities]
    assert names == ['downlink']
    assert result.ok
    rejection = next(r for r in result.rejected if r.activity.name == 'payload_burst')
    assert rejection.reason == 'resource_violation'
    v = rejection.violations[0]
    assert v.resource == 'battery'
    assert v.window.start_tai_s > _payload().window.end_tai_s
    assert 'payload_burst' in v.causes

def test_attribution_ranks_main_contributor_first():
    model = _power_model()
    big = _payload()
    small = Activity('beacon', TimeWindow(1000.0, 1500.0), 'tx', 1, ResourceDemand(load_w=50.0))
    result = explain_violations(simulate([big, small], model), [big, small], model)
    causes = result.violations[0].causes
    assert causes[0] == 'payload_burst'
    assert 'beacon' in causes

def _thermal_model():
    return ResourceModel(
        bus=BusModel(battery_capacity_wh=1e6, initial_energy_wh=1e6),
        nodes=(NodeModel('cam', 2000.0, 290.0), NodeModel('bus', 8000.0, 290.0, max_k=303.0)),
        conductances=(('cam', 'bus', 0.5),),
        env_sunlit_k=290.0,
        env_eclipse_k=290.0)

def test_thermal_peak_lands_after_activity_end():
    model = _thermal_model()
    heater = Activity('cam_heat', TimeWindow(1000.0, 2000.0), 'cam', 10, ResourceDemand(load_w=300.0, heat_w=300.0, node='cam'))
    assert simulate([], model, start_tai_s=1000.0, end_tai_s=10000.0).ok
    result = simulate([heater], model, tail_s=8000.0)
    assert not result.ok
    v = result.violations[0]
    assert v.resource == 'bus' and v.sense == 'above_max'
    assert v.window.start_tai_s > heater.window.end_tai_s
    explained = explain_violations(result, [heater], model)
    assert explained.violations[0].causes == ('cam_heat',)

def test_unknown_thermal_node_rejected():
    model = _thermal_model()
    bad = Activity('bad', TimeWindow(0.0, 10.0), 'cam', 1, ResourceDemand(load_w=1.0, heat_w=1.0, node='nope'))
    try:
        simulate([bad], model)
        assert False
    except ValueError:
        pass

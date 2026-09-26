from __future__ import annotations
from dataclasses import dataclass, replace
from .timeline import Timeline, Activity
from .resource_timeline import simulate, explain_violations, SimulationResult

def greedy_schedule(candidates):
    timeline = Timeline()
    rejected = []
    for a in sorted(candidates, key=lambda x: (-x.priority, x.window.end_tai_s, x.name)):
        try:
            timeline.add(a)
        except ValueError:
            rejected.append(a)
    return (timeline, rejected)

def score_schedule(timeline):
    return sum((a.priority * max(0.0, a.window.duration_s) for a in timeline.activities))

@dataclass(frozen=True)
class Rejection:
    activity: Activity
    reason: str
    violations: tuple = ()

@dataclass(frozen=True)
class ScheduleResult:
    timeline: Timeline
    rejected: tuple
    simulation: SimulationResult

    @property
    def ok(self):
        return self.simulation.ok

def resource_checked_schedule(candidates, model, tail_s=0.0):
    timeline = Timeline()
    rejected = []
    baseline = {}
    for a in sorted(candidates, key=lambda x: (-x.priority, x.window.end_tai_s, x.name)):
        try:
            timeline.add(a)
        except ValueError:
            rejected.append(Rejection(a, 'resource_conflict'))
            continue
        result = simulate(timeline.activities, model, tail_s=tail_s)
        worse = tuple((v for v in result.violations if v.margin < baseline.get((v.resource, v.sense), float('inf')) - 1e-09))
        if worse:
            timeline.remove(a.name)
            detail = explain_violations(replace(result, violations=worse), timeline.activities + [a], model)
            rejected.append(Rejection(a, 'resource_violation', detail.violations))
            continue
        baseline = {}
        for v in result.violations:
            key = (v.resource, v.sense)
            baseline[key] = min(baseline.get(key, float('inf')), v.margin)
    return ScheduleResult(timeline, tuple(rejected), simulate(timeline.activities, model, tail_s=tail_s))

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping
from orbitforge.core.state import TimeWindow
from .resource_timeline import (
    ResourceViolationError,
    evaluate_resource_timeline,
    simulate_resources,
)

if TYPE_CHECKING:
    from .resource_timeline import ResourceModel, ResourceSimulation

@dataclass(frozen=True)
class Activity:
    name: str
    window: TimeWindow
    resource: str
    priority: int = 0
    load_w: float = 0.0
    heat_w: float | Mapping[str, float] = 0.0

class Timeline:

    def __init__(self, resource_model: 'ResourceModel | None' = None):
        self.activities = []
        self.resource_model = resource_model
        self.resource_rejections = []
        self._baseline_resource_simulation = None
        if resource_model is not None:
            simulation = simulate_resources([], resource_model)
            if simulation.violations:
                raise ResourceViolationError(None, simulation.violations, simulation)
            self._baseline_resource_simulation = simulation

    def add(self, a: Activity):
        for e in self.activities:
            if e.resource == a.resource and e.window.intersects(a.window):
                raise ValueError(f'resource conflict: {a.resource}')
        if self.resource_model is not None:
            simulation = simulate_resources(
                self.activities + [a],
                self.resource_model,
                cause_activity=a.name,
            )
            if simulation.violations:
                self.resource_rejections.append((a, simulation))
                raise ResourceViolationError(a, simulation.violations, simulation)
        self.activities.append(a)
        self.activities.sort(key=lambda x: (x.window.start_tai_s, -x.priority, x.name))
        return self

    def evaluate_resources(self, start_tai_s=None, end_tai_s=None) -> 'ResourceSimulation':
        if self.resource_model is None:
            raise ValueError('timeline has no resource model')
        return evaluate_resource_timeline(
            self.activities, self.resource_model, start_tai_s, end_tai_s
        )

    def gaps(self, start, end, resource):
        cur = start
        out = []
        for a in [x for x in self.activities if x.resource == resource and x.window.end_tai_s > start and (x.window.start_tai_s < end)]:
            if a.window.start_tai_s > cur:
                out.append(TimeWindow(cur, min(a.window.start_tai_s, end)))
            cur = max(cur, a.window.end_tai_s)
        if cur < end:
            out.append(TimeWindow(cur, end))
        return out

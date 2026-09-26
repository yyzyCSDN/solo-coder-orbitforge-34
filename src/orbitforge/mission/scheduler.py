from __future__ import annotations

from .timeline import Timeline
from .resource_timeline import ResourceModel


def greedy_schedule(candidates, resource_model: ResourceModel | None = None):
    timeline = Timeline(resource_model)
    rejected = []
    for a in sorted(candidates, key=lambda x: (-x.priority, x.window.end_tai_s, x.name)):
        try:
            timeline.add(a)
        except ValueError:
            rejected.append(a)
    return (timeline, rejected)

def score_schedule(timeline):
    return sum((a.priority * max(0.0, a.window.duration_s) for a in timeline.activities))

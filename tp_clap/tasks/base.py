# Copyright 2026 Mohan Li. Licensed under the Apache License, Version 2.0.
"""Task registry shared by every evaluation benchmark."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple


@dataclass
class EvalContext:
    """Everything a task needs: the loaded model and the dataset paths for it."""

    model: Any
    audio_processor: Any
    tokenizer: Any
    device: str
    paths: Dict[str, Any] = field(default_factory=dict)

    def path(self, key):
        """Return a configured path for this task, or raise a helpful error."""
        if key not in self.paths:
            raise KeyError(f"Missing dataset path {key!r} in the config for this task.")

        return self.paths[key]


@dataclass
class TaskResult:
    """Metrics for one benchmark, and optionally the key of its headline number.

    Leave ``primary_metric`` unset when no single number stands for the task, as
    with the per-subset accuracies of the AQA benchmarks; the report then lists
    every metric instead of leading with one.
    """

    metrics: Dict[str, float]
    primary_metric: Optional[str] = None

    @property
    def primary(self):
        return self.metrics[self.primary_metric] if self.primary_metric else None


@dataclass
class Task:
    """A registered benchmark."""

    name: str
    group: str
    metric: str                      # headline metric shown in a run's report
    required_paths: Tuple[str, ...]  # keys the task needs inside its config section
    fn: Callable[[EvalContext], TaskResult]
    config_key: str = ""             # config section to read; defaults to the name
    metrics_summary: str = ""        # all metrics the task reports, for --list-tasks

    def __post_init__(self):
        # Tasks over the same dataset (NSynth, MagnaTagATune) share one section.
        self.config_key = self.config_key or self.name
        # Most tasks report one metric, so the summary defaults to the headline.
        self.metrics_summary = self.metrics_summary or self.metric

    def __call__(self, context):
        return self.fn(context)


TASK_REGISTRY: Dict[str, Task] = {}

# Benchmark families, in the order they are reported.
GROUPS = ("retrieval", "classification", "a2a_retrieval", "aqa")


def register(name, group, metric, required_paths=(), config_key=None, metrics_summary=None):
    """Decorator registering an evaluation function as a named task."""

    def decorator(fn):
        if name in TASK_REGISTRY:
            raise ValueError(f"Task {name!r} is already registered.")
        if group not in GROUPS:
            raise ValueError(f"Unknown group {group!r}; expected one of {GROUPS}.")

        TASK_REGISTRY[name] = Task(
            name=name,
            group=group,
            metric=metric,
            required_paths=tuple(required_paths),
            fn=fn,
            config_key=config_key or name,
            metrics_summary=metrics_summary,
        )

        return fn

    return decorator


def get_task(name):
    if name not in TASK_REGISTRY:
        raise KeyError(f"Unknown task {name!r}. Available: {', '.join(list_tasks())}")

    return TASK_REGISTRY[name]


def list_tasks(group=None):
    """Task names, ordered by group then registration order."""
    names = [name for name, task in TASK_REGISTRY.items() if group is None or task.group == group]

    return sorted(names, key=lambda name: GROUPS.index(TASK_REGISTRY[name].group))


__all__ = [
    "EvalContext",
    "TaskResult",
    "Task",
    "TASK_REGISTRY",
    "GROUPS",
    "register",
    "get_task",
    "list_tasks",
]

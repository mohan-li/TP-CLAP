# Copyright 2026 Mohan Li. Licensed under the Apache License, Version 2.0.
"""Evaluation tasks.

Importing this package registers every benchmark in
:data:`tp_clap.tasks.base.TASK_REGISTRY`.
"""

from .base import (
    GROUPS,
    TASK_REGISTRY,
    EvalContext,
    Task,
    TaskResult,
    get_task,
    list_tasks,
    register,
)

# Imported for their registration side effects.
from . import a2a_retrieval, aqa, classification, retrieval  # noqa: F401,E402


__all__ = [
    "GROUPS",
    "TASK_REGISTRY",
    "EvalContext",
    "Task",
    "TaskResult",
    "get_task",
    "list_tasks",
    "register",
]

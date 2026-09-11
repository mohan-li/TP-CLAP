# Copyright 2026 Mohan Li. Licensed under the Apache License, Version 2.0.
"""Driver that loads checkpoints and runs them over a set of benchmarks."""

import logging
import os
import textwrap
import time
from collections import OrderedDict

import torch
import yaml
from transformers import AutoFeatureExtractor, AutoTokenizer

from .configuration_tp_clap import TPCLAPConfig
from .modeling_tp_clap import TPCLAPModel
from .tasks import GROUPS, EvalContext, get_task, list_tasks


logger = logging.getLogger(__name__)


def load_config(config_path):
    """Read the YAML config describing the model, the checkpoints and the datasets."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"No config at {config_path}. Copy configs/datasets.example.yaml to "
            "configs/datasets.yaml and fill in the paths to your datasets."
        )

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    datasets = config.get("datasets") or {}
    for name, entry in datasets.items():
        if not isinstance(entry, dict):
            raise ValueError(f"datasets.{name} must be a mapping of keys to paths.")
        datasets[name] = {
            key: os.path.expanduser(value) if isinstance(value, str) else value
            for key, value in entry.items()
        }

    config["datasets"] = datasets
    config["checkpoints"] = {
        name: os.path.expanduser(path)
        for name, path in (config.get("checkpoints") or {}).items()
    }
    config.setdefault("model", {})

    return config


def resolve_checkpoint(task_name, config, default_checkpoint=None):
    """Which checkpoint a task runs with.

    A per-task entry under ``checkpoints:`` wins, because the attribute-focused
    retrieval benchmarks have their own fine-tuned weights. Otherwise the command
    line checkpoint is used, falling back to ``checkpoints.default``.
    """
    checkpoints = config.get("checkpoints") or {}
    checkpoint = checkpoints.get(task_name) or default_checkpoint or checkpoints.get("default")

    if checkpoint is None:
        raise ValueError(
            f"No checkpoint for task {task_name!r}: pass --checkpoint, or set "
            "checkpoints.default in the config."
        )

    return checkpoint


class ModelProvider:
    """Loads checkpoints on demand, keeping a single model on the device.

    Benchmarks are grouped by checkpoint in the reporting order, so switching
    models costs a handful of reloads rather than holding several ~900MB models
    on the GPU at once.
    """

    def __init__(self, config, device="cuda:0", strict=True):
        self.model_config = TPCLAPConfig(**config.get("model", {}))
        self.device = device
        self.strict = strict

        self.audio_processor = AutoFeatureExtractor.from_pretrained(
            self.model_config.audio_encoder, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.text_encoder)

        self._checkpoint = None
        self._model = None

    def model_for(self, checkpoint):
        if checkpoint != self._checkpoint:
            self.release()
            logger.info("Loading %s on %s", checkpoint, self.device)
            self._model = TPCLAPModel.from_checkpoint(
                checkpoint, self.model_config, device=self.device, strict=self.strict
            )
            self._checkpoint = checkpoint

        return self._model

    def release(self):
        if self._model is None:
            return

        self._model = None
        self._checkpoint = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def resolve_tasks(names, config):
    """Split requested tasks into runnable ones and ones missing dataset paths."""
    names = list_tasks() if not names or "all" in names else list(names)

    runnable = []
    skipped = OrderedDict()

    for name in names:
        task = get_task(name)
        paths = config["datasets"].get(task.config_key, {})
        missing = [key for key in task.required_paths if key not in paths]

        if missing:
            skipped[name] = (
                f"not configured (datasets.{task.config_key} missing: {', '.join(missing)})"
            )
        else:
            runnable.append(name)

    return runnable, skipped


def run_tasks(provider, task_names, config, default_checkpoint=None, fail_fast=False):
    """Run each task in turn, collecting metrics (and errors) per task."""
    results = OrderedDict()

    for name in task_names:
        task = get_task(name)
        checkpoint = resolve_checkpoint(name, config, default_checkpoint)

        logger.info("Evaluating %s with %s", name, checkpoint)
        started = time.time()

        try:
            context = EvalContext(
                model=provider.model_for(checkpoint),
                audio_processor=provider.audio_processor,
                tokenizer=provider.tokenizer,
                device=provider.device,
                paths=config["datasets"].get(task.config_key, {}),
            )
            result = task(context)
        except Exception as error:  # keep going: one broken dataset should not sink the run
            if fail_fast:
                raise
            logger.exception("Task %s failed", name)
            results[name] = {"group": task.group, "error": f"{type(error).__name__}: {error}"}
            continue

        results[name] = {
            "group": task.group,
            "metric": task.metric,
            "checkpoint": checkpoint,
            "primary_metric": result.primary_metric,
            "primary": result.primary,
            "metrics": result.metrics,
            "runtime_s": round(time.time() - started, 1),
        }

    return results


def format_report(results, skipped=None):
    """Render results as a plain-text table."""
    lines = []
    width = max([len(name) for name in results] + [len(name) for name in (skipped or {})] + [12])

    for group in GROUPS:
        group_results = [(name, entry) for name, entry in results.items() if entry.get("group") == group]
        if not group_results:
            continue

        lines.append("")
        lines.append(f"{group.upper()}")
        lines.append("-" * (width + 46))

        for name, entry in group_results:
            if "error" in entry:
                lines.append(f"{name:<{width}}  FAILED  {entry['error']}")
                continue

            details = "  ".join(
                f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
                for key, value in entry["metrics"].items()
                if key != entry["primary_metric"]
            )

            if entry["primary_metric"] is None:
                # No single headline number: list every metric instead.
                lines.append(f"{name:<{width}}  {details}".rstrip())
            else:
                lines.append(
                    f"{name:<{width}}  {entry['metric']}={entry['primary']:.4f}  {details}".rstrip()
                )

    checkpoints = OrderedDict()
    for name, entry in results.items():
        if "checkpoint" in entry:
            checkpoints.setdefault(entry["checkpoint"], []).append(name)

    if checkpoints:
        lines.append("")
        lines.append("CHECKPOINTS")
        lines.append("-" * (width + 46))
        for checkpoint, task_names in checkpoints.items():
            lines.append(
                textwrap.fill(
                    f"{checkpoint}  ->  " + ", ".join(task_names),
                    width=width + 46,
                    subsequent_indent="    ",
                )
            )

    if skipped:
        lines.append("")
        lines.append("SKIPPED")
        lines.append("-" * (width + 46))
        for name, reason in skipped.items():
            lines.append(f"{name:<{width}}  {reason}")

    return "\n".join(lines)


__all__ = [
    "load_config",
    "resolve_checkpoint",
    "ModelProvider",
    "resolve_tasks",
    "run_tasks",
    "format_report",
]

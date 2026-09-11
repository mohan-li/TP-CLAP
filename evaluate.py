#!/usr/bin/env python
# Copyright 2026 Mohan Li. Licensed under the Apache License, Version 2.0.
"""Evaluate TP-CLAP on one or more benchmarks.

Which weights each benchmark runs with comes from the `checkpoints:` section of the
config, so no flag is needed for a full run: the a2a_retrieval_* benchmarks pick up
their own fine-tuned checkpoints while everything else uses `checkpoints.default`.
`--checkpoint` replaces that default, which is how you evaluate a new main model.

Examples:
    python evaluate.py --tasks all
    python evaluate.py --tasks esc50 gtzan --device cuda:1
    python evaluate.py --checkpoint checkpoints/my-new-model.pt --tasks esc50
    python evaluate.py --list-tasks
"""

import argparse
import json
import logging
import os
import sys
import warnings

from tp_clap.evaluation import (
    ModelProvider,
    format_report,
    load_config,
    resolve_tasks,
    run_tasks,
)
from tp_clap.tasks import TASK_REGISTRY, list_tasks


DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "datasets.yaml")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        help="Model .pt to use in place of checkpoints.default. Benchmarks that pin "
             "their own checkpoint in the config (the a2a_retrieval_* ones) keep it.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="YAML config with the model settings and dataset paths (default: configs/datasets.yaml).",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        metavar="TASK",
        help="Benchmarks to run, or 'all' (default). Choices: " + ", ".join(list_tasks()),
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device (default: cuda:0).")
    parser.add_argument("--output", help="Write the results to this JSON file.")
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Allow the checkpoint to only partially match the model.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort on the first failing task instead of reporting it and continuing.",
    )
    parser.add_argument("--list-tasks", action="store_true", help="List the available benchmarks and exit.")
    parser.add_argument("--verbose", action="store_true", help="Show debug logging.")

    return parser.parse_args(argv)


def print_task_list():
    width = max(len(name) for name in TASK_REGISTRY)
    group_width = max(len(task.group) for task in TASK_REGISTRY.values())

    for name in list_tasks():
        task = TASK_REGISTRY[name]
        print(f"{name:<{width}}  {task.group:<{group_width}}  {task.metrics_summary}")


def main(argv=None):
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    warnings.filterwarnings("ignore")

    if args.list_tasks:
        print_task_list()
        return 0

    try:
        config = load_config(args.config)
        task_names, skipped = resolve_tasks(args.tasks, config)
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if not task_names:
        print("No runnable tasks: fill in the dataset paths in " + args.config, file=sys.stderr)
        print(format_report({}, skipped=skipped), file=sys.stderr)
        return 1

    pinned = [name for name in task_names if (config.get("checkpoints") or {}).get(name)]
    if args.checkpoint and pinned:
        logging.info(
            "%d benchmark(s) keep their own checkpoint from %s: %s",
            len(pinned), args.config, ", ".join(pinned),
        )

    provider = ModelProvider(config, device=args.device, strict=not args.non_strict)

    results = run_tasks(
        provider, task_names, config,
        default_checkpoint=args.checkpoint, fail_fast=args.fail_fast,
    )

    print(format_report(results, skipped=skipped))

    if args.output:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(
                {
                    "device": args.device,
                    "results": results,
                    "skipped": skipped,
                },
                f,
                indent=2,
            )
        print(f"\nWrote {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

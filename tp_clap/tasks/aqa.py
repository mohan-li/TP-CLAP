# Copyright 2026 Mohan Li. Licensed under the Apache License, Version 2.0.
"""Multiple-choice audio question answering (AQA) benchmarks (MMAU, MMAR).

A choice is scored by two complementary similarities that are summed:

1. the prompt-conditioned audio embedding against each answer on its own, and
2. the plain audio embedding against each "<question> <answer>." sentence.
"""

import logging
import os
from collections import defaultdict

import torch
from tqdm import tqdm

from ..encoding import SAMPLE_RATE, encode_prompted_audio, encode_texts, encode_waveforms, load_waveform
from .base import EvalContext, TaskResult, register


logger = logging.getLogger(__name__)


@torch.no_grad()
def score_choices(context: EvalContext, waveform, question, choices):
    """Return one score per choice for a single clip."""
    model, tokenizer = context.model, context.tokenizer

    # 1. Prompt-conditioned audio vs. each answer on its own.
    conditioned_embed = encode_prompted_audio(
        model, context.audio_processor, tokenizer, waveform, question, context.device
    )
    choice_embeds = encode_texts(
        model, tokenizer, list(choices), context.device,
        batch_size=len(choices), show_progress=False,
    )
    scores = conditioned_embed @ choice_embeds.T

    # 2. Plain audio vs. the question and answer concatenated.
    audio_embed = encode_waveforms(model, context.audio_processor, waveform, context.device)
    concat_embeds = encode_texts(
        model, tokenizer, [f"{question} {choice}." for choice in choices], context.device,
        batch_size=len(choices), show_progress=False,
    )
    scores = scores + audio_embed @ concat_embeds.T

    return scores


def run_multiple_choice(context: EvalContext, items, categories, desc):
    """Score an iterable of ``(category, waveform_fn, question, choices, answer)`` items."""
    correct = defaultdict(int)
    total = defaultdict(int)
    skipped = 0

    for category, get_waveform, question, choices, answer in tqdm(items, desc=desc):
        if category not in categories:
            continue

        try:
            waveform = get_waveform()
            target = list(choices).index(answer)
            prediction = score_choices(context, waveform, question, choices).argmax().item()
        except (ValueError, RuntimeError, OSError) as error:
            # Unreadable audio, or an answer that is not among the choices.
            logger.debug("Skipping item: %s", error)
            skipped += 1
            continue

        total[category] += 1
        correct[category] += int(prediction == target)

    num_total = sum(total.values())
    logger.info(
        "%s: scored %d examples, skipped %d", desc, num_total, skipped
    )

    metrics = {
        f"{category}_acc": correct[category] / total[category]
        for category in categories
        if total[category] > 0
    }

    return TaskResult(metrics=metrics)


@register(
    "mmau",
    group="aqa",
    metric="accuracy per subset",
    required_paths=("hf_dataset",),
)
def evaluate_mmau(context: EvalContext):
    """MMAU, read straight from the Hub (audio is bundled with the dataset)."""
    from datasets import load_dataset

    dataset = load_dataset(
        context.path("hf_dataset"),
        split=context.paths.get("split", "v05.15.25"),
    )

    items = (
        (
            item["task"],
            lambda item=item: item["audio"]["array"],
            item["question"],
            item["choices"],
            item["answer"],
        )
        for item in dataset
    )

    return run_multiple_choice(context, items, categories=("sound", "music"), desc="MMAU")


@register(
    "mmar",
    group="aqa",
    metric="accuracy per subset",
    required_paths=("hf_dataset", "audio_root"),
)
def evaluate_mmar(context: EvalContext):
    """MMAR: metadata from the Hub, audio from a local download directory."""
    from datasets import load_dataset

    dataset = load_dataset(
        context.path("hf_dataset"),
        split=context.paths.get("split", "test"),
    )
    audio_root = context.path("audio_root")

    items = (
        (
            item["modality"],
            lambda item=item: load_waveform(os.path.join(audio_root, item["audio_path"]), SAMPLE_RATE),
            item["question"],
            item["choices"],
            item["answer"],
        )
        for item in dataset
    )

    return run_multiple_choice(
        context, items, categories=("sound", "music", "mix-sound-music"), desc="MMAR"
    )


__all__ = ["score_choices", "run_multiple_choice", "evaluate_mmau", "evaluate_mmar"]

# Copyright 2026 Mohan Li. Licensed under the Apache License, Version 2.0.
"""Audio-text retrieval benchmarks (AudioCaps, Clotho)."""

import os

import pandas as pd

from ..encoding import encode_audio_paths, encode_texts
from ..metrics import retrieval_recalls
from .base import EvalContext, TaskResult, register


K_LIST = (1, 5, 10)


def load_audiocaps(csv_path, audio_dir):
    """Official AudioCaps test csv: ``youtube_id, start_time, caption`` (5 rows per clip)."""
    df = pd.read_csv(csv_path)

    audio_map = {}  # (youtube_id, start_time) -> audio index
    audio_paths = []
    captions = []
    caption_to_audio = []

    for _, row in df.iterrows():
        youtube_id = str(row["youtube_id"]).strip()
        start_time = str(row["start_time"]).strip()
        caption = str(row["caption"]).strip()

        key = (youtube_id, start_time)
        if key not in audio_map:
            audio_map[key] = len(audio_paths)
            audio_paths.append(os.path.join(audio_dir, f"{youtube_id}_{start_time}.wav"))

        captions.append(caption)
        caption_to_audio.append(audio_map[key])

    return audio_paths, captions, caption_to_audio


def load_clotho(csv_path, audio_dir):
    """Official Clotho caption csv: ``file_name, caption_1 ... caption_5``."""
    audio_paths = []
    captions = []
    caption_to_audio = []

    df = pd.read_csv(csv_path)
    caption_columns = [f"caption_{i}" for i in range(1, 6)]

    for audio_index, row in df.iterrows():
        audio_paths.append(os.path.join(audio_dir, str(row["file_name"]).strip()))

        for column in caption_columns:
            caption = str(row[column]).strip()
            if caption and caption.lower() != "nan":
                captions.append(caption)
                caption_to_audio.append(audio_index)

    return audio_paths, captions, caption_to_audio


def run_retrieval(context: EvalContext, audio_paths, captions, caption_to_audio, audio_batch_size):
    audio_embeds = encode_audio_paths(
        context.model,
        context.audio_processor,
        audio_paths,
        context.device,
        batch_size=audio_batch_size,
    )
    text_embeds = encode_texts(context.model, context.tokenizer, captions, context.device)

    a2t, t2a = retrieval_recalls(audio_embeds, text_embeds, caption_to_audio, K_LIST)

    metrics = {f"a2t_{key}": value for key, value in a2t.items()}
    metrics.update({f"t2a_{key}": value for key, value in t2a.items()})

    return TaskResult(metrics=metrics, primary_metric="a2t_R@1")


@register(
    "audiocaps",
    group="retrieval",
    metric="a2t R@1",
    required_paths=("csv", "audio_dir"),
    metrics_summary="R@1,5,10",
)
def evaluate_audiocaps(context: EvalContext):
    audio_paths, captions, caption_to_audio = load_audiocaps(
        context.path("csv"), context.path("audio_dir")
    )

    return run_retrieval(context, audio_paths, captions, caption_to_audio, audio_batch_size=64)


@register(
    "clotho",
    group="retrieval",
    metric="a2t R@1",
    required_paths=("csv", "audio_dir"),
    metrics_summary="R@1,5,10",
)
def evaluate_clotho(context: EvalContext):
    audio_paths, captions, caption_to_audio = load_clotho(
        context.path("csv"), context.path("audio_dir")
    )

    # Clotho clips vary from 15s to 30s; encoding one at a time avoids padding
    # every clip in a batch out to the longest one.
    return run_retrieval(context, audio_paths, captions, caption_to_audio, audio_batch_size=1)


__all__ = ["load_audiocaps", "load_clotho", "evaluate_audiocaps", "evaluate_clotho"]

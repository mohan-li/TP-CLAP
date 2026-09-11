# Copyright 2026 Mohan Li. Licensed under the Apache License, Version 2.0.
"""Attribute-focused audio-to-audio retrieval (NSynth, MagnaTagATune).

Each clip is embedded twice from one audio-encoder pass: once through the
prompt-conditioned head, using an instruction that names the attribute of
interest ("What is the tempo of the music?"), and once through the plain audio
head. The conditioned embeddings are the queries and the plain ones the
database, so the same clips are retrieved differently depending on which
attribute the instruction asks about. Every task is reported twice:
`conditioned_` for the instruction-conditioned queries and `unconditioned_` for
the baseline.

NSynth attributes are single-label, so relevance is an exact label match and the
report is precision@K plus mAP. MagnaTagATune clips carry several tags, so
relevance is the Jaccard overlap of their tag sets, giving a soft precision (SP@K)
and a soft mAP (SmAP).
"""

import json
import os

import numpy as np
import torch
from tqdm import tqdm

from ..encoding import SAMPLE_RATE, encode_conditioned_and_plain
from ..metrics import jaccard_relevance, label_relevance, ranked_retrieval_scores
from .base import EvalContext, TaskResult, register
from .classification import NSYNTH_LABELS


KS = (5, 10, 20)

# Scores are ranked in chunks of queries so the full N x N matrices are never
# materialised at once.
CHUNK_SIZE = 64

# A self-match is pushed to the end of its own ranking rather than dropped, so
# every ranking has the same length.
SELF_MATCH_SCORE = -1e9


def rank_against_database(query_embeds, database_embeds, relevance_fn, precision_key, map_key):
    """Average ranking scores over all queries, in chunks."""
    num_queries = query_embeds.shape[0]
    totals = {}

    for start in tqdm(range(0, num_queries, CHUNK_SIZE), desc="Ranking"):
        end = min(start + CHUNK_SIZE, num_queries)

        similarity = query_embeds[start:end] @ database_embeds.T
        similarity[torch.arange(end - start), torch.arange(start, end)] = SELF_MATCH_SCORE

        scores = ranked_retrieval_scores(
            similarity,
            relevance_fn(start, end),
            ks=KS,
            precision_key=precision_key,
        )

        for name, values in scores.items():
            totals[name] = totals.get(name, 0.0) + values.sum().item()

    metrics = {name: total / num_queries for name, total in totals.items()}
    metrics[map_key] = metrics.pop("AP")

    return metrics


def run_a2a_retrieval(
    context: EvalContext,
    audio_paths,
    relevance_fn,
    prompt,
    batch_size=64,
    loader=None,
    precision_key="P",
    map_key="mAP",
):
    """Encode the clips, then rank them by the conditioned and the plain embedding."""
    conditioned_embeds, audio_embeds = encode_conditioned_and_plain(
        context.model,
        context.audio_processor,
        context.tokenizer,
        audio_paths,
        prompt,
        context.device,
        batch_size=batch_size,
        loader=loader,
    )

    metrics = {}
    for prefix, query_embeds in (("conditioned", conditioned_embeds), ("unconditioned", audio_embeds)):
        scores = rank_against_database(
            query_embeds, audio_embeds, relevance_fn, precision_key, map_key
        )
        metrics.update({f"{prefix}_{name}": value for name, value in scores.items()})

    return TaskResult(metrics=metrics, primary_metric=f"conditioned_{precision_key}@{KS[-1]}")


# ---------------------------------------------------------------------------
# NSynth: one attribute label per note
# ---------------------------------------------------------------------------

NSYNTH_SOURCE_LABELS = ["acoustic", "electronic", "synthetic"]

# Order matches the paper's Table 4 (Instrument, Pitch, Source): each attribute is
# registered as a task in this dict's iteration order, and that is what the report
# prints under, so keep it in this order rather than alphabetising it.
NSYNTH_ATTRIBUTES = {
    "instrument": "What is the instrument playing the musical note?",
    "pitch": "What is the MIDI pitch of the musical note?",
    "source": "What is the instrument source of the musical note?",
}


def load_nsynth(json_path, audio_dir, attribute):
    """Official NSynth ``examples.json`` plus its audio directory."""
    with open(json_path) as f:
        data = json.load(f)

    instrument_index = {label: i for i, label in enumerate(NSYNTH_LABELS)}
    source_index = {label: i for i, label in enumerate(NSYNTH_SOURCE_LABELS)}

    audio_paths = []
    labels = []

    for name, info in data.items():
        audio_paths.append(os.path.join(audio_dir, f"{name}.wav"))

        if attribute == "instrument":
            labels.append(instrument_index[info["instrument_family_str"]])
        elif attribute == "source":
            labels.append(source_index[info["instrument_source_str"]])
        else:  # MIDI pitch, used directly as the class
            labels.append(int(info["pitch"]))

    return audio_paths, labels


def register_nsynth(attribute):
    prompt = NSYNTH_ATTRIBUTES[attribute]

    def evaluate(context: EvalContext):
        audio_paths, labels = load_nsynth(
            context.path("json"), context.path("audio_dir"), attribute
        )
        labels = torch.tensor(labels, device=context.device)

        return run_a2a_retrieval(
            context,
            audio_paths,
            lambda start, end: label_relevance(labels[start:end], labels),
            prompt,
        )

    evaluate.__name__ = f"evaluate_nsynth_{attribute}"

    ks = ",".join(str(k) for k in KS)

    return register(
        f"a2a_retrieval_nsynth_{attribute}",
        group="a2a_retrieval",
        metric=f"conditioned_P@{KS[-1]}",
        required_paths=("json", "audio_dir"),
        config_key="nsynth",
        metrics_summary=f"P@{ks}, mAP",
    )(evaluate)


for _attribute in NSYNTH_ATTRIBUTES:
    register_nsynth(_attribute)


# ---------------------------------------------------------------------------
# MagnaTagATune: several tags per clip
# ---------------------------------------------------------------------------

MTT_GENRES = [
    "rock", "pop", "country", "electronic", "techno", "metal",
    "dance", "classical", "new age", "ambient", "opera", "indian",
]

MTT_INSTRUMENTS = [
    "piano", "guitar", "violin", "cello", "harp", "flute",
    "harpsichord", "sitar", "synth", "drums", "strings",
]

# Order matches the paper's Table 5 (Genre, Instrument, Tempo); see the note on
# NSYNTH_ATTRIBUTES above.
MTT_ATTRIBUTES = {
    "genre": "What is the genre of the music?",
    "instrument": "What is the instrument playing the music?",
    "tempo": "What is the tempo of the music?",
}

# The clips are ~29s; the middle 10s window is the one that is scored.
MTT_WINDOW = slice(10 * SAMPLE_RATE, 20 * SAMPLE_RATE)


def select_mtt_tags(tags, attribute):
    """The tags of interest for one attribute, or an empty list to skip the clip."""
    if attribute == "genre":
        # MagnaTagATune spells it "classic"; the vocabulary uses "classical".
        return [tag for tag in (tag.replace("classic", "classical") for tag in tags) if tag in MTT_GENRES]

    if attribute == "instrument":
        return [tag for tag in tags if tag in MTT_INSTRUMENTS]

    # Tempo: keep only clips tagged unambiguously fast or slow.
    is_fast, is_slow = "fast" in tags, "slow" in tags
    if is_fast == is_slow:
        return []

    return ["fast"] if is_fast else ["slow"]


def load_magnatagatune(dataset_id, split, audio_dir, attribute):
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, split=split)

    audio_paths = []
    tag_sets = []

    for item in tqdm(dataset, total=dataset.num_rows, desc="Reading MagnaTagATune tags"):
        tags = select_mtt_tags(item["tag_top50"], attribute)
        if not tags:
            continue

        audio_paths.append(os.path.join(audio_dir, str(item["path"]).strip()))
        tag_sets.append(tags)

    return audio_paths, tag_sets


def build_multi_hot(tag_sets, device):
    """(N, num_tags) multi-hot matrix over the tags present in this subset."""
    vocabulary = sorted({tag for tags in tag_sets for tag in tags})
    tag_index = {tag: i for i, tag in enumerate(vocabulary)}

    multi_hot = torch.zeros(len(tag_sets), len(vocabulary), device=device)
    for row, tags in enumerate(tag_sets):
        for tag in tags:
            multi_hot[row, tag_index[tag]] = 1.0

    return multi_hot


def load_mtt_waveform(path):
    """MagnaTagATune ships stereo mp3s at 32kHz; take the mono 10-20s window."""
    import resampy
    import soundfile as sf

    waveform, sample_rate = sf.read(path)

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    if sample_rate != SAMPLE_RATE:
        waveform = resampy.resample(waveform, sample_rate, SAMPLE_RATE)

    return waveform.astype(np.float32)[MTT_WINDOW]


def register_magnatagatune(attribute):
    prompt = MTT_ATTRIBUTES[attribute]

    def evaluate(context: EvalContext):
        audio_paths, tag_sets = load_magnatagatune(
            context.path("hf_dataset"),
            context.paths.get("split", "test"),
            context.path("audio_dir"),
            attribute,
        )
        multi_hot = build_multi_hot(tag_sets, context.device)

        return run_a2a_retrieval(
            context,
            audio_paths,
            lambda start, end: jaccard_relevance(multi_hot[start:end], multi_hot),
            prompt,
            loader=load_mtt_waveform,
            precision_key="SP",
            map_key="SmAP",
        )

    evaluate.__name__ = f"evaluate_magnatagatune_{attribute}"

    ks = ",".join(str(k) for k in KS)

    return register(
        f"a2a_retrieval_mtt_{attribute}",
        group="a2a_retrieval",
        metric=f"conditioned_SP@{KS[-1]}",
        required_paths=("hf_dataset", "audio_dir"),
        config_key="magnatagatune",
        metrics_summary=f"SP@{ks}, SmAP",
    )(evaluate)


for _attribute in MTT_ATTRIBUTES:
    register_magnatagatune(_attribute)

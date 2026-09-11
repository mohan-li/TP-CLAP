# Copyright 2026 Mohan Li. Licensed under the Apache License, Version 2.0.
"""Zero-shot classification benchmarks, grouped into the sound and music domains.

Every task here follows the same recipe: embed each clip, embed one natural
language prompt per class, and rank classes by cosine similarity. Single-label sets
report top-1 accuracy; multi-label FSD50K reports mAP over the same similarity matrix.
Datasets differ only in how the file list is built, which prompt template is used, and
how audio is batched, so they are declared as specs rather than written out one by one.
"""

import json
import os

import numpy as np
import pandas as pd
import torch

from ..encoding import encode_audio_paths, encode_texts
from ..metrics import accuracy, mean_average_precision
from .base import EvalContext, TaskResult, register


SOUND_TEMPLATE = "The sound of {label}."
MUSIC_TEMPLATE = "The music of {label}."
GENRE_TEMPLATE = "The music of {label} genre."
EMOTION_TEMPLATE = "The speech with {article} {label} tone."

# Clips shorter than this break the audio front-end, so they are zero-padded.
MIN_SAMPLES = 2400


def indefinite_article(label):
    return "an" if label[:1].lower() in "aeiou" else "a"


def build_prompts(label_names, template):
    return [
        template.format(label=label, article=indefinite_article(label))
        for label in label_names
    ]


def encode_class_prompts(context: EvalContext, label_names, template=SOUND_TEMPLATE):
    """Embed one prompt per class, all in a single batch."""
    prompts = build_prompts(label_names, template)

    return encode_texts(
        context.model,
        context.tokenizer,
        prompts,
        context.device,
        batch_size=max(len(prompts), 1),
        show_progress=False,
    )


def run_classification(
    context: EvalContext,
    audio_paths,
    labels,
    label_names,
    template=SOUND_TEMPLATE,
    batch_size=64,
    min_samples=None,
):
    audio_embeds = encode_audio_paths(
        context.model,
        context.audio_processor,
        audio_paths,
        context.device,
        batch_size=batch_size,
        min_samples=min_samples,
    )
    text_embeds = encode_class_prompts(context, label_names, template)

    metrics = {"accuracy": accuracy(audio_embeds @ text_embeds.T, labels)}

    return TaskResult(metrics=metrics, primary_metric="accuracy")


def register_classification(
    name,
    loader,
    required_paths,
    template=SOUND_TEMPLATE,
    batch_size=64,
    min_samples=None,
    config_key=None,
):
    """Register a single-label zero-shot benchmark from its loader and protocol."""

    def evaluate(context: EvalContext):
        audio_paths, labels, label_names = loader(context)

        return run_classification(
            context, audio_paths, labels, label_names,
            template=template, batch_size=batch_size, min_samples=min_samples,
        )

    evaluate.__name__ = f"evaluate_{name}"

    return register(
        name,
        group="classification",
        metric="accuracy",
        required_paths=required_paths,
        config_key=config_key,
    )(evaluate)


# ---------------------------------------------------------------------------
# Dataset loaders: (context) -> (audio_paths, labels, label_names)
# ---------------------------------------------------------------------------

def load_esc50(context):
    df = pd.read_csv(context.path("csv"))
    audio_dir = context.path("audio_dir")

    label_names = [label.replace("_", " ") for label in sorted(df["category"].unique())]
    label_to_index = {label: i for i, label in enumerate(label_names)}

    audio_paths = [os.path.join(audio_dir, name) for name in df["filename"]]
    labels = [label_to_index[category.replace("_", " ")] for category in df["category"]]

    return audio_paths, labels, label_names


def load_fsd50k_eval(csv_path, audio_dir):
    """Official FSD50K ``eval.csv``: ``fname, labels`` with comma separated labels."""
    df = pd.read_csv(csv_path)

    label_names = sorted({
        label.lower().replace("_", " ")
        for labels in df["labels"]
        for label in labels.split(",")
    })
    label_to_index = {label: i for i, label in enumerate(label_names)}

    audio_paths = []
    targets = np.zeros((len(df), len(label_names)), dtype=np.float32)

    for row_index, (_, row) in enumerate(df.iterrows()):
        audio_paths.append(os.path.join(audio_dir, f"{row['fname']}.wav"))

        for label in row["labels"].split(","):
            targets[row_index, label_to_index[label.lower().replace("_", " ")]] = 1

    return audio_paths, torch.from_numpy(targets), label_names


def load_us8k(context):
    df = pd.read_csv(context.path("csv"))
    audio_dir = context.path("audio_dir")

    label_names = [label.replace("_", " ") for label in sorted(df["class"].unique())]
    label_to_index = {label: i for i, label in enumerate(label_names)}

    audio_paths = [
        os.path.join(audio_dir, f"fold{row['fold']}", row["slice_file_name"])
        for _, row in df.iterrows()
    ]
    labels = [label_to_index[name.replace("_", " ")] for name in df["class"]]

    return audio_paths, labels, label_names


VOCALSOUND_LABELS = ["cough", "laughter", "sigh", "sneeze", "sniff", "throat clearing"]


def load_vocalsound(context):
    audio_dir = context.path("audio_dir")

    with open(context.path("json")) as f:
        data = json.load(f)["data"]

    label_to_index = {label: i for i, label in enumerate(VOCALSOUND_LABELS)}

    audio_paths = []
    labels = []

    for item in data:
        file_name = os.path.basename(item["wav"])
        audio_paths.append(os.path.join(audio_dir, file_name))

        # e.g. "f0003_0_throatclearing.wav" -> "throat clearing"
        label = file_name.rsplit("_", 1)[-1].removesuffix(".wav").replace("throatclearing", "throat clearing")
        labels.append(label_to_index[label])

    return audio_paths, labels, VOCALSOUND_LABELS


# Emotion names, keyed by the code used in the file names.
CREMAD_LABELS = {
    "ANG": "angry",
    "DIS": "disgusted",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}


def load_cremad(context):
    audio_dir = context.path("audio_dir")
    label_names = list(CREMAD_LABELS.values())
    label_to_index = {label: i for i, label in enumerate(label_names)}

    audio_paths = []
    labels = []

    for file_name in sorted(os.listdir(audio_dir)):
        audio_paths.append(os.path.join(audio_dir, file_name))
        labels.append(label_to_index[CREMAD_LABELS[file_name.split("_")[2]]])

    return audio_paths, labels, label_names


GTZAN_LABELS = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]


def load_gtzan(context):
    """GTZAN as shipped: one directory of clips per genre."""
    audio_dir = context.path("audio_dir")

    audio_paths = []
    labels = []

    for index, genre in enumerate(GTZAN_LABELS):
        genre_dir = os.path.join(audio_dir, genre)
        for file_name in sorted(os.listdir(genre_dir)):
            audio_paths.append(os.path.join(genre_dir, file_name))
            labels.append(index)

    return audio_paths, labels, GTZAN_LABELS


# Percussion instrument names, keyed by the pinyin used in the file names.
BEIJING_OPERA_LABELS = {
    "bangu": "clapper-drum",
    "daluo": "large gong",
    "naobo": "cymbals",
    "xiaoluo": "small gong",
}


def load_beijingopera(context):
    audio_dir = context.path("audio_dir")
    label_to_index = {key: i for i, key in enumerate(BEIJING_OPERA_LABELS)}

    audio_paths = []
    labels = []

    for file_name in sorted(os.listdir(audio_dir)):
        if not file_name.endswith(".wav"):
            continue

        audio_paths.append(os.path.join(audio_dir, file_name))
        labels.append(label_to_index[file_name.split("_")[-1].split("-")[0]])

    return audio_paths, labels, list(BEIJING_OPERA_LABELS.values())


NSYNTH_LABELS = [
    "bass", "brass", "flute", "guitar", "keyboard",
    "mallet", "organ", "reed", "string", "vocal",
]


def load_nsynth_instrument(context):
    audio_dir = context.path("audio_dir")

    with open(context.path("json")) as f:
        data = json.load(f)

    label_to_index = {label: i for i, label in enumerate(NSYNTH_LABELS)}

    audio_paths = []
    labels = []

    for name, info in data.items():
        audio_paths.append(os.path.join(audio_dir, f"{name}.wav"))
        labels.append(label_to_index[info["instrument_family_str"]])

    return audio_paths, labels, NSYNTH_LABELS


# ---------------------------------------------------------------------------
# Registration, in reporting order.
#
# `batch_size=1` on some datasets is deliberate: clips in a batch are padded to
# the longest one and embeddings are mean pooled over time, so batching changes
# the numbers on datasets with variable clip lengths.
# ---------------------------------------------------------------------------

register_classification(
    "esc50", load_esc50,
    required_paths=("csv", "audio_dir"), batch_size=64,
)


@register(
    "fsd50k",
    group="classification",
    metric="mAP",
    required_paths=("csv", "audio_dir"),
)
def evaluate_fsd50k(context: EvalContext):
    """FSD50K is multi-label, so it is scored with mAP instead of accuracy."""
    audio_paths, targets, label_names = load_fsd50k_eval(
        context.path("csv"), context.path("audio_dir")
    )

    # FSD50K clip lengths range from 0.3s to 30s, so clips are encoded one by one.
    audio_embeds = encode_audio_paths(
        context.model, context.audio_processor, audio_paths, context.device, batch_size=1
    )
    text_embeds = encode_class_prompts(context, label_names)

    mean_ap, _ = mean_average_precision(audio_embeds @ text_embeds.T, targets)

    return TaskResult(metrics={"mAP": mean_ap}, primary_metric="mAP")


register_classification(
    "us8k", load_us8k,
    required_paths=("csv", "audio_dir"),
    batch_size=1, min_samples=MIN_SAMPLES,
)
register_classification(
    "vocalsound", load_vocalsound,
    required_paths=("json", "audio_dir"), batch_size=1,
)
register_classification(
    "cremad", load_cremad,
    required_paths=("audio_dir",),
    template=EMOTION_TEMPLATE, batch_size=1,
)
register_classification(
    "gtzan", load_gtzan,
    required_paths=("audio_dir",),
    template=GENRE_TEMPLATE, batch_size=64,
)
register_classification(
    "beijingopera", load_beijingopera,
    required_paths=("audio_dir",),
    template=MUSIC_TEMPLATE, batch_size=1, min_samples=MIN_SAMPLES,
)
register_classification(
    "nsynth_instrument", load_nsynth_instrument,
    required_paths=("json", "audio_dir"), config_key="nsynth",
    template=MUSIC_TEMPLATE, batch_size=64,
)

# Copyright 2026 Mohan Li. Licensed under the Apache License, Version 2.0.
"""Batched audio/text embedding helpers shared by all evaluation tasks."""

import librosa
import numpy as np
import torch
from tqdm import tqdm


SAMPLE_RATE = 16000


def load_waveform(path, sample_rate=SAMPLE_RATE, min_samples=None):
    """Load a mono waveform, zero-padded up to ``min_samples`` if it is shorter.

    The padding guards against clips that are too short for the audio front-end.
    """
    waveform, _ = librosa.load(path, sr=sample_rate)

    if min_samples is not None and len(waveform) < min_samples:
        waveform = np.pad(waveform, (0, min_samples - len(waveform)))

    return waveform


@torch.no_grad()
def encode_waveforms(model, audio_processor, waveforms, device, sample_rate=SAMPLE_RATE):
    """Embed one batch of waveforms (a list of 1-D arrays) into audio embeddings."""
    audio_inputs = audio_processor(waveforms, sampling_rate=sample_rate, return_tensors="pt")
    audios = audio_inputs.input_values.to(device)

    return model.get_audio_embeds(audios=audios).float()


@torch.no_grad()
def encode_audio_paths(
    model,
    audio_processor,
    audio_paths,
    device,
    batch_size=64,
    sample_rate=SAMPLE_RATE,
    min_samples=None,
):
    """Embed a list of audio files.

    ``batch_size`` is part of the evaluation protocol, not just a speed knob:
    clips in a batch are padded to the longest one, and embeddings are mean
    pooled over time, so results depend on it. Keep the per-task defaults to
    reproduce the reported numbers.
    """
    audio_embeds = []

    for start in tqdm(range(0, len(audio_paths), batch_size), desc="Encoding audio"):
        waveforms = [
            load_waveform(path, sample_rate, min_samples)
            for path in audio_paths[start:start + batch_size]
        ]
        audio_embeds.append(encode_waveforms(model, audio_processor, waveforms, device, sample_rate))

    return torch.cat(audio_embeds, dim=0)


@torch.no_grad()
def encode_texts(
    model,
    tokenizer,
    texts,
    device,
    batch_size=64,
    show_progress=True,
):
    """Embed a list of strings into text embeddings."""
    text_embeds = []

    steps = range(0, len(texts), batch_size)
    for start in tqdm(steps, desc="Encoding text", disable=not show_progress):
        text_inputs = tokenizer(texts[start:start + batch_size], padding=True, return_tensors="pt")
        text_embeds.append(
            model.get_text_embeds(
                text_ids=text_inputs.input_ids.to(device),
                text_attention_mask=text_inputs.attention_mask.to(device),
            ).float()
        )

    return torch.cat(text_embeds, dim=0)


@torch.no_grad()
def encode_conditioned_and_plain(
    model,
    audio_processor,
    tokenizer,
    audio_paths,
    prompt,
    device,
    batch_size=64,
    sample_rate=SAMPLE_RATE,
    loader=None,
):
    """Embed each clip twice from a single audio-encoder pass.

    Returns ``(conditioned_embeds, audio_embeds)``: the same clips seen through
    the prompt-conditioned head and through the plain audio head. Audio-to-audio
    retrieval uses the first as queries and the second as the database.

    ``loader`` overrides how a path becomes a waveform (MagnaTagATune needs its
    own decoding), defaulting to :func:`load_waveform`.
    """
    loader = loader if loader is not None else (lambda path: load_waveform(path, sample_rate))

    conditioned_embeds = []
    audio_embeds = []

    for start in tqdm(range(0, len(audio_paths), batch_size), desc="Encoding audio"):
        waveforms = [loader(path) for path in audio_paths[start:start + batch_size]]

        audio_inputs = audio_processor(waveforms, sampling_rate=sample_rate, return_tensors="pt")
        audio_features = model.audio_encoder(audio_inputs.input_values.to(device)).logits

        prompt_inputs = tokenizer([prompt] * len(waveforms), padding=True, return_tensors="pt")

        conditioned_embeds.append(
            model.get_conditioned_audio_embeds(
                None,
                audio_features,
                prompt_inputs.input_ids.to(device),
                prompt_inputs.attention_mask.to(device),
            ).float()
        )
        audio_embeds.append(model.get_audio_embeds(None, audio_features).float())

    return torch.cat(conditioned_embeds, dim=0), torch.cat(audio_embeds, dim=0)


@torch.no_grad()
def encode_prompted_audio(model, audio_processor, tokenizer, waveform, prompt, device, sample_rate=SAMPLE_RATE):
    """Embed a single clip conditioned on a text prompt (the fusion branch)."""
    audio_inputs = audio_processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
    audios = audio_inputs.input_values.to(device)

    prompt_inputs = tokenizer([prompt], padding=True, return_tensors="pt")

    return model.get_conditioned_audio_embeds(
        audios,
        None,
        prompt_inputs.input_ids.to(device),
        prompt_inputs.attention_mask.to(device),
    ).float()


__all__ = [
    "SAMPLE_RATE",
    "load_waveform",
    "encode_waveforms",
    "encode_audio_paths",
    "encode_texts",
    "encode_conditioned_and_plain",
    "encode_prompted_audio",
]

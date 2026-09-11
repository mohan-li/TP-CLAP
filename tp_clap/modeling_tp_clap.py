# Copyright 2026 Mohan Li. Licensed under the Apache License, Version 2.0.
"""TP-CLAP: a prompt-conditioned audio-text model (inference only).

This module carries the parts needed to load a checkpoint and produce
embeddings. The training objective and its losses are not part of this release.
"""

import logging
import os

import torch
from torch import nn

from transformers import AutoModel, AutoModelForAudioClassification

from .configuration_tp_clap import TPCLAPConfig
from .fusion import TPCLAPFusionEncoder


logger = logging.getLogger(__name__)


class TPCLAPModel(nn.Module):
    """Audio-text model with three embedding heads sharing two backbones.

    * :meth:`get_audio_embeds` - unconditioned audio embedding.
    * :meth:`get_text_embeds` - text embedding (mean-pooled over valid tokens).
    * :meth:`get_conditioned_audio_embeds` - audio embedding conditioned on a
      text prompt through :class:`~tp_clap.fusion.TPCLAPFusionEncoder`.

    All three are L2-normalised, so a dot product is a cosine similarity.
    """

    def __init__(self, config: TPCLAPConfig):
        super().__init__()

        self.config = config

        # Audio encoder (classification head discarded)
        self.audio_encoder = AutoModelForAudioClassification.from_pretrained(
            config.audio_encoder,
            device_map="cpu",
            trust_remote_code=True
        ).encoder

        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(
            config.text_encoder,
            device_map="cpu"
        )

        # Prompt-conditioned fusion encoder
        self.fusion_encoder = TPCLAPFusionEncoder(
            hidden_size=self.text_encoder.config.hidden_size,
            num_layers=config.num_fusion_layers,
            num_heads=self.text_encoder.config.num_attention_heads
        )

        # Audio projector
        self.audio_projector = nn.Sequential(
            nn.Linear(self.audio_encoder.config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, config.embed_dim)
        )

        # Prompt-conditioned audio projector
        self.conditioned_audio_projector = nn.Sequential(
            nn.Linear(self.audio_encoder.config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, config.embed_dim)
        )

        # Text projector
        self.text_projector = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, config.embed_dim)
        )

    def load_checkpoint(self, checkpoint_path, strict=True):
        """Load a ``state_dict`` into this model."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        # Tolerate checkpoints saved from a DistributedDataParallel wrapper.
        if any(key.startswith("module.") for key in state_dict):
            state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}

        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        if missing:
            logger.warning("Missing keys when loading %s: %d", checkpoint_path, len(missing))
        if unexpected:
            logger.warning("Unexpected keys when loading %s: %d", checkpoint_path, len(unexpected))

        return missing, unexpected

    @classmethod
    def from_checkpoint(cls, checkpoint_path, config=None, device="cpu", strict=True):
        """Build the model from ``config`` and load ``checkpoint_path`` into it."""
        config = config if config is not None else TPCLAPConfig()
        model = cls(config)
        model.load_checkpoint(checkpoint_path, strict=strict)
        model.to(device)
        model.eval()

        return model

    def get_audio_embeds(
        self,
        audios: torch.FloatTensor = None,
        audio_features: torch.FloatTensor = None
    ):
        """Embed audio. Pass ``audio_features`` to reuse an encoder pass."""
        if audio_features is None:
            audio_features = self.audio_encoder(audios).logits

        embeds = audio_features.mean(1)
        embeds = self.audio_projector(embeds)
        embeds = nn.functional.normalize(embeds, p=2, dim=1)

        return embeds

    def get_text_embeds(
        self,
        text_ids: torch.LongTensor,
        text_attention_mask: torch.LongTensor
    ):
        """Embed text, mean pooling over the non-padded tokens."""
        text_features = self.text_encoder(text_ids, text_attention_mask).last_hidden_state

        embeds = (text_features * text_attention_mask.unsqueeze(-1)).sum(1) / text_attention_mask.sum(1, keepdim=True)
        embeds = self.text_projector(embeds)
        embeds = nn.functional.normalize(embeds, p=2, dim=1)

        return embeds

    def get_conditioned_audio_embeds(
        self,
        audios: torch.FloatTensor = None,
        audio_features: torch.FloatTensor = None,
        prompt_ids: torch.LongTensor = None,
        prompt_attention_mask: torch.Tensor = None
    ):
        """Embed audio conditioned on a text prompt."""
        if audio_features is None:
            audio_features = self.audio_encoder(audios).logits
        prompt_features = self.text_encoder(prompt_ids, prompt_attention_mask).last_hidden_state

        embeds = self.fusion_encoder(audio_features, prompt_features, prompt_attention_mask)
        embeds = self.conditioned_audio_projector(embeds)
        embeds = nn.functional.normalize(embeds, p=2, dim=1)

        return embeds


__all__ = ["TPCLAPModel"]

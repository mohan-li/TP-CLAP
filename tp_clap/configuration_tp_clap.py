# Copyright 2026 Mohan Li. Licensed under the Apache License, Version 2.0.
"""TP-CLAP model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class TPCLAPConfig(PretrainedConfig):
    """Configuration for :class:`~tp_clap.modeling_tp_clap.TPCLAPModel`.

    Args:
        audio_encoder: Hub id (or local path) of the audio backbone. The
            classification head is discarded and only its encoder is kept.
        text_encoder: Hub id (or local path) of the text backbone.
        embed_dim: Dimension of the shared audio/text embedding space.
        num_fusion_layers: Number of self/cross-attention layers used to
            condition audio features on a text prompt.

    Weights are loaded separately, see :meth:`TPCLAPModel.from_checkpoint`.
    """

    model_type = "tp_clap"

    def __init__(
        self,
        audio_encoder="mispeech/ced-base",
        text_encoder="google-bert/bert-base-uncased",
        embed_dim=1024,
        num_fusion_layers=2,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.embed_dim = embed_dim
        self.num_fusion_layers = num_fusion_layers


__all__ = ["TPCLAPConfig"]

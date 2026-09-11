# Copyright 2026 Mohan Li. Licensed under the Apache License, Version 2.0.
"""TP-CLAP: prompt-conditioned audio-text retrieval."""

from .configuration_tp_clap import TPCLAPConfig
from .fusion import TPCLAPFusionEncoder, TPCLAPFusionLayer
from .modeling_tp_clap import TPCLAPModel

__version__ = "0.1.0"

__all__ = [
    "TPCLAPConfig",
    "TPCLAPModel",
    "TPCLAPFusionEncoder",
    "TPCLAPFusionLayer",
    "__version__",
]

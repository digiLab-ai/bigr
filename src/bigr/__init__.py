"""BIGR: guided particle inference over a frozen recursive model.

    load_trm_backbone_checkpoint   load a TRM checkpoint (EMA weights by default)
    select_device                  best available device: CUDA > MPS > CPU
    TRMCoreEmbedder                token ids -> embedded input x
    TRMCoreDecoder                 hypothesis state y -> token logits
    TRMCoreQHead                   hypothesis state y -> raw Q-head logit
    TRMNativeCycleFTheta           the TRM recursion: M inner passes + one read-out
    TRMCoreFTheta                  single-pass variant (unit tests)
    StochasticFTheta               the stochastic proposal kernel
    BIGRFilter                     the guided particle filter (Algorithm 1)
    InferenceConfig                filter hyperparameters
    ParticleState, StepRecord, FilterTrace   run containers
"""

import torch

from bigr.backbone import (
    TRMCoreDecoder,
    TRMCoreEmbedder,
    TRMCoreFTheta,
    TRMCoreQHead,
    TRMNativeCycleFTheta,
    load_trm_backbone_checkpoint,
)
from bigr.config import InferenceConfig
from bigr.filter import BIGRFilter
from bigr.stochastic import StochasticFTheta
from bigr.types import FilterTrace, ParticleState, StepRecord


def select_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


__all__ = [
    "load_trm_backbone_checkpoint",
    "select_device",
    "TRMCoreEmbedder",
    "TRMCoreDecoder",
    "TRMCoreQHead",
    "TRMCoreFTheta",
    "TRMNativeCycleFTheta",
    "StochasticFTheta",
    "BIGRFilter",
    "ParticleState",
    "FilterTrace",
    "StepRecord",
    "InferenceConfig",
]

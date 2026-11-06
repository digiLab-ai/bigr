"""Shared pytest fixtures for the guided particle inference test suite.

Uses a minimal TRMModule (hidden_size=32, seq_len=4, vocab_size=5) to keep
all unit and integration tests fast with no checkpoint required.
"""

from __future__ import annotations

import math

import pytest
import torch

from bigr.backbone import (
    TRMCoreDecoder,
    TRMCoreEmbedder,
    TRMCoreFTheta,
    TRMCoreQHead,
    TRMNativeCycleFTheta,
)
from trm.nn.models.trm import TRMModule

# ---------------------------------------------------------------------------
# Tiny backbone dimensions
# ---------------------------------------------------------------------------

HIDDEN_SIZE = 32
SEQ_LEN = 4
VOCAB_SIZE = 5
BATCH_SIZE = 2
NUM_LAYERS = 1
NUM_HEADS = 4
H_CYCLES = 1
L_CYCLES = 1


@pytest.fixture(scope="session")
def tiny_trm() -> TRMModule:
    """Tiny TRMModule for fast unit/integration tests."""
    max_grid_size = max(1, int(math.sqrt(SEQ_LEN)))
    return TRMModule(
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        max_grid_size=max_grid_size,
        H_cycles=H_CYCLES,
        L_cycles=L_CYCLES,
        N_supervision=1,
        N_supervision_val=1,
        ffn_expansion=2,
        learning_rate=1e-4,
        learning_rate_emb=1e-2,
        weight_decay=0.0,
        warmup_steps=0,
        halt_exploration_prob=0.0,
        puzzle_emb_dim=0,
        puzzle_emb_len=0,
        rope_theta=10000,
        pos_emb_type="1d",
        use_mlp_t=False,
        use_conv_swiglu=False,
        use_board_swiglu=False,
        lr_min_ratio=1.0,
        use_muon=False,
        vocab_size=VOCAB_SIZE,
        num_puzzles=1,
        batch_size=BATCH_SIZE,
        pad_value=0,
        seq_len=SEQ_LEN,
        forward_dtype=torch.float32,
    )


@pytest.fixture(scope="session")
def tiny_f_theta(tiny_trm) -> TRMCoreFTheta:
    return TRMCoreFTheta(tiny_trm)


@pytest.fixture(scope="session")
def tiny_native_f(tiny_trm) -> TRMNativeCycleFTheta:
    return TRMNativeCycleFTheta(tiny_trm)


@pytest.fixture(scope="session")
def tiny_decoder(tiny_trm) -> TRMCoreDecoder:
    return TRMCoreDecoder(tiny_trm)


@pytest.fixture(scope="session")
def tiny_q_head(tiny_trm) -> TRMCoreQHead:
    return TRMCoreQHead(tiny_trm)


@pytest.fixture(scope="session")
def tiny_embedder(tiny_trm) -> TRMCoreEmbedder:
    return TRMCoreEmbedder(tiny_trm)

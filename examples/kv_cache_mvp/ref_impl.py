"""Pure numpy reference implementation for M12 end-to-end demo.

Provides a simple 3-layer MLP: embed → relu → linear (projection).
Used to validate VM output against numpy ground truth.
"""
from __future__ import annotations

import struct
import numpy as np


# Fixed-size demo model dimensions
VOCAB_SIZE  = 16
EMBED_DIM   = 8
HIDDEN_DIM  = 8
OUTPUT_DIM  = 8


def _fixed_embed_weight() -> np.ndarray:
    """Deterministic embedding table (VOCAB_SIZE × EMBED_DIM, float32)."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 0.1, (VOCAB_SIZE, EMBED_DIM)).astype(np.float32)


def _fixed_linear_weight() -> np.ndarray:
    """Deterministic projection weight (HIDDEN_DIM × OUTPUT_DIM, float32)."""
    rng = np.random.default_rng(7)
    return rng.normal(0, 0.1, (HIDDEN_DIM, OUTPUT_DIM)).astype(np.float32)


# Precomputed weights shared between reference and mock implementations
EMBED_WEIGHT  = _fixed_embed_weight()
LINEAR_WEIGHT = _fixed_linear_weight()


def reference_decode_step(token_id: int) -> np.ndarray:
    """numpy reference: embed(token_id) → relu → linear projection."""
    embedded = EMBED_WEIGHT[token_id % VOCAB_SIZE]   # (EMBED_DIM,) float32
    hidden   = np.maximum(embedded, 0.0)             # relu
    output   = hidden @ LINEAR_WEIGHT                # (OUTPUT_DIM,) float32
    return output.astype(np.float32)


def pack_f32(arr: np.ndarray) -> bytes:
    """Pack float32 numpy array as little-endian bytes."""
    return struct.pack(f"<{arr.size}f", *arr.ravel().tolist())


def unpack_f32(data: bytes | bytearray, count: int, offset: int = 0) -> np.ndarray:
    """Unpack float32 bytes to numpy array."""
    vals = struct.unpack_from(f"<{count}f", data, offset)
    return np.array(vals, dtype=np.float32)

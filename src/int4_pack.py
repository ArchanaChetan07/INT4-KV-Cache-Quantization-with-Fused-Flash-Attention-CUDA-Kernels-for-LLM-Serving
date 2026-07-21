"""INT4 nibble packing: two 4-bit values per byte.

The quantizer and kernels operate on one-value-per-byte uint8 for clarity;
this module provides the storage format — even indices in the low nibble,
odd indices in the high nibble along the last axis. Round-trips are exact.

    packed = pack_int4(q)              # last dim halves (padded if odd)
    q      = unpack_int4(packed, dim)  # exact recovery
"""

import numpy as np


def pack_int4(q: np.ndarray) -> np.ndarray:
    """Pack uint8 values in [0, 15] into nibbles along the last axis.

    Odd-length last axes are zero-padded before packing; pass the original
    length to unpack_int4 to strip the pad.
    """
    assert q.dtype == np.uint8, f"expected uint8, got {q.dtype}"
    assert q.max(initial=0) <= 15, "values must fit in 4 bits"

    if q.shape[-1] % 2 == 1:
        pad = np.zeros(q.shape[:-1] + (1,), dtype=np.uint8)
        q = np.concatenate([q, pad], axis=-1)

    lo = q[..., 0::2]
    hi = q[..., 1::2]
    return (lo | (hi << 4)).astype(np.uint8)


def unpack_int4(packed: np.ndarray, last_dim: int) -> np.ndarray:
    """Unpack nibbles back to one uint8 value per element.

    Args:
        packed: output of pack_int4
        last_dim: original (pre-padding) length of the last axis
    """
    assert packed.dtype == np.uint8

    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F

    out = np.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=np.uint8)
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out[..., :last_dim]


def packed_nbytes(shape: tuple) -> int:
    """Storage bytes for a packed tensor of the given unpacked shape."""
    last = (shape[-1] + 1) // 2
    n = last
    for d in shape[:-1]:
        n *= d
    return n

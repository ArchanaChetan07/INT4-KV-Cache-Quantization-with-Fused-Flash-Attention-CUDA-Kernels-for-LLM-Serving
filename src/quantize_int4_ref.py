"""INT4 Quantization Reference Implementation

Per-channel asymmetric INT4 quantization for KV cache.

Quantization formula:
  q = clip(round((kv - min_val) / scale), 0, 15)

Where:
  scale[b,h,c] = (max_val[b,h,c] - min_val[b,h,c]) / 15.0
  zp[b,h,c] = -min_val[b,h,c] / scale[b,h,c]

The scale and zero-point are stored per (block, kv_head, channel).
This provides fine-grained quantization while capturing channel-wise statistics.

Dequantization:
  kv_dequant = (q - zp) * scale
"""

import numpy as np
from typing import Tuple, Optional


def quantize_int4_ref(
    kv: np.ndarray,
    per_channel: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize KV cache to INT4 (asymmetric, per-channel).

    Args:
        kv: Input KV tensor of shape (num_blocks, kv_head_dim, num_heads, head_dim)
            or (..., seq_len, head_dim) for reference testing.
        per_channel: If True, use per-channel scales (default).
                    If False, use per-head scales.

    Returns:
        q: Quantized INT4 values, shape same as kv, dtype uint8
           (each byte stores two INT4 values: lower nibble, upper nibble)
        scale: Scale factors per channel/head, shape (..., head_dim)
        zp: Zero-points per channel/head, shape (..., head_dim)

    Note:
        - Asymmetric quantization captures the full range [min, max]
        - Per-channel granularity (one scale per dimension)
        - Quantization domain: [0, 15] for INT4
    """
    assert kv.ndim >= 2, f"KV must be at least 2D, got {kv.ndim}D"
    assert kv.dtype in [np.float32, np.float16], f"Expected FP32/FP16, got {kv.dtype}"

    kv = kv.astype(np.float32)  # Convert to FP32 for numerics

    if per_channel:
        # Per-channel: one scale per last dimension (head_dim)
        # Compute min/max over all dimensions except the last
        axes_to_reduce = tuple(range(kv.ndim - 1))
        min_val = np.min(kv, axis=axes_to_reduce, keepdims=True)
        max_val = np.max(kv, axis=axes_to_reduce, keepdims=True)
    else:
        # Per-head: one scale per all dimensions except last (for multi-dim case)
        # For simplicity, reduce over last dimension only
        min_val = np.min(kv, axis=-1, keepdims=True)
        max_val = np.max(kv, axis=-1, keepdims=True)

    # Compute scale: (max - min) / 15
    # Add small epsilon to avoid division by zero
    range_val = max_val - min_val
    range_val = np.maximum(range_val, 1e-8)
    scale = range_val / 15.0

    # Compute zero-point: -min / scale
    zp = -min_val / scale

    # Quantize: (kv - min_val) / scale, clipped to [0, 15]
    # This is equivalent to: (kv - min_val) / scale
    q_float = (kv - min_val) / scale
    q = np.clip(np.round(q_float), 0, 15).astype(np.uint8)

    # Remove keepdims for output: squeeze exactly the axes that were reduced.
    # per_channel reduced the leading axes (scale left with shape 1,...,1,C);
    # per-row reduced the last axis (scale left with shape ...,1).
    squeeze_axes = axes_to_reduce if per_channel else (-1,)
    scale = np.squeeze(scale, axis=squeeze_axes)
    zp = np.squeeze(zp, axis=squeeze_axes)

    return q, scale, zp


def dequantize_int4_ref(
    q: np.ndarray,
    scale: np.ndarray,
    zp: np.ndarray,
    target_shape: Optional[Tuple] = None,
    per_channel: bool = True,
) -> np.ndarray:
    """Dequantize INT4 values back to FP32.

    Args:
        q: Quantized INT4 values, shape (...,)
        scale: Scale factors — per_channel: shape (head_dim,) broadcasting
               over the last axis; per-row: shape (...,) broadcasting over
               all axes except the last.
        zp: Zero-points, same shape as scale.
        target_shape: If provided, reshape output to this shape.
        per_channel: Must match the flag used at quantization time — it
                     determines which axis the scales broadcast along.

    Returns:
        kv_dequant: Dequantized values, dtype float32
    """
    q = q.astype(np.float32)
    scale = scale.astype(np.float32)
    zp = zp.astype(np.float32)

    if per_channel:
        # Scales index the last axis: prepend broadcast dims
        while scale.ndim < q.ndim:
            scale = np.expand_dims(scale, axis=0)
            zp = np.expand_dims(zp, axis=0)
    else:
        # Scales index the leading axes: append a broadcast dim for rows
        while scale.ndim < q.ndim:
            scale = np.expand_dims(scale, axis=-1)
            zp = np.expand_dims(zp, axis=-1)

    # Dequantize: (q - zp) * scale
    # Actually, standard INT4 dequant is: q * scale + min
    # But since zp = -min / scale, we can use: (q - zp) * scale
    # However, that's not quite right. Let's use the correct formula:
    # kv = q * scale + min_val
    # Since min_val = -zp * scale:
    # kv = q * scale - zp * scale = (q - zp) * scale
    kv_dequant = q * scale - zp * scale

    if target_shape is not None:
        kv_dequant = kv_dequant.reshape(target_shape)

    return kv_dequant


def quantize_and_dequantize_ref(
    kv: np.ndarray,
    per_channel: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Quantize then dequantize to measure round-trip error.

    Args:
        kv: Input KV tensor
        per_channel: Per-channel quantization flag

    Returns:
        q: Quantized values
        scale: Scale factors
        zp: Zero-points
        kv_dequant: Dequantized values
    """
    q, scale, zp = quantize_int4_ref(kv, per_channel=per_channel)
    kv_dequant = dequantize_int4_ref(q, scale, zp, target_shape=kv.shape,
                                     per_channel=per_channel)
    return q, scale, zp, kv_dequant


def compute_quantization_error(
    kv_original: np.ndarray,
    kv_dequant: np.ndarray,
) -> Tuple[float, float, float]:
    """Compute quantization error metrics.

    Args:
        kv_original: Original FP32 values
        kv_dequant: Dequantized values

    Returns:
        mae: Mean absolute error
        rmse: Root mean squared error
        max_err: Maximum absolute error
    """
    err = np.abs(kv_original - kv_dequant)
    mae = np.mean(err)
    rmse = np.sqrt(np.mean(err ** 2))
    max_err = np.max(err)
    return mae, rmse, max_err


def quantization_error_bounds(
    kv_original: np.ndarray,
    per_channel: bool = True,
) -> dict:
    """Compute theoretical and actual quantization error bounds.

    Args:
        kv_original: Original KV values
        per_channel: Quantization granularity

    Returns:
        Dictionary with error metrics
    """
    q, scale, zp, kv_dequant = quantize_and_dequantize_ref(kv_original, per_channel=per_channel)
    mae, rmse, max_err = compute_quantization_error(kv_original, kv_dequant)

    # Theoretical bounds: max error is scale/2 per value
    # (INT4 divides range into 15 intervals, so quantization step = scale)
    max_theoretical_err = np.max(scale) / 2.0

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'max_error': float(max_err),
        'max_theoretical_bound': float(max_theoretical_err),
        'bound_tight': bool(max_err <= max_theoretical_err * 1.01),  # Allow 1% slack
    }


class INT4Quantizer:
    """Stateful INT4 quantizer for streaming quantization."""

    def __init__(self, per_channel: bool = True):
        self.per_channel = per_channel
        self.scales = None
        self.zps = None
        self.num_quantized = 0

    def quantize_block(self, kv_block: np.ndarray) -> np.ndarray:
        """Quantize a single KV block.

        Args:
            kv_block: KV tensor for one block

        Returns:
            Quantized INT4 values
        """
        q, scale, zp = quantize_int4_ref(kv_block, per_channel=self.per_channel)
        self.scales = scale
        self.zps = zp
        self.num_quantized += 1
        return q

    def dequantize_block(self, q_block: np.ndarray) -> np.ndarray:
        """Dequantize a block using stored scales/zps.

        Args:
            q_block: Quantized INT4 block

        Returns:
            Dequantized FP32 values
        """
        assert self.scales is not None, "No quantization stats available"
        return dequantize_int4_ref(q_block, self.scales, self.zps, target_shape=q_block.shape)

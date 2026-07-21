"""
Dispatch layer for Flash Decoding INT4.

Routes to CUDA kernels when available, falling back to NumPy references.
"""

import numpy as np
from typing import List, Optional, Tuple

from .quantize_int4_ref import quantize_int4_ref, dequantize_int4_ref
from .flash_decode_ref import online_softmax_ref, _online_softmax_fp32

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    from . import _C
    HAS_CUDA = _C is not None
except ImportError:
    HAS_CUDA = False
    _C = None

if not HAS_CUDA:
    # Opt-in JIT compile: FLASH_DECODE_JIT_CUDA=1 (needs GPU + nvcc)
    from ._jit import load_extension
    _C = load_extension("flash_decode_int4_C", "FLASH_DECODE_JIT_CUDA")
    HAS_CUDA = _C is not None


def _use_cuda(force: Optional[bool] = None) -> bool:
    if force is not None:
        return force and HAS_CUDA
    return HAS_CUDA and HAS_TORCH


def quantize_int4(
    kv: np.ndarray,
    use_cuda: Optional[bool] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize KV to INT4 (per-channel asymmetric). Dispatches to CUDA or reference.

    Args:
        kv: 2D array [num_rows, num_channels]
        use_cuda: Force backend (None = auto-detect)

    Returns:
        (q, scale, zp) as NumPy arrays
    """
    if _use_cuda(use_cuda) and kv.ndim == 2:
        kv_t = torch.from_numpy(kv.astype(np.float32)).cuda()
        q, scale, zp = _C.quantize_int4(kv_t)
        return q.cpu().numpy(), scale.cpu().numpy(), zp.cpu().numpy()

    return quantize_int4_ref(kv, per_channel=True)


def flash_decode(
    query: np.ndarray,
    k_q_blocks: List[np.ndarray],
    k_scales: List[np.ndarray],
    k_zps: List[np.ndarray],
    v_blocks: List[np.ndarray],
    block_lens: Optional[np.ndarray] = None,
    use_cuda: Optional[bool] = None,
) -> np.ndarray:
    """Fused online-softmax attention over INT4 quantized paged KV.

    Args:
        query: [batch, heads, head_dim]
        k_q_blocks: List of INT4-quantized key blocks [block_size, head_dim]
        k_scales: List of per-channel scales [head_dim]
        k_zps: List of per-channel zero-points [head_dim]
        v_blocks: List of FP32 value blocks [block_size, head_dim]
        block_lens: Effective length per block
        use_cuda: Force backend (None = auto-detect)

    Returns:
        Attention output [batch, heads, head_dim]
    """
    num_blocks = len(k_q_blocks)
    if block_lens is None:
        block_lens = np.array([b.shape[0] for b in k_q_blocks], dtype=np.int32)

    if _use_cuda(use_cuda):
        block_size = max(b.shape[0] for b in k_q_blocks)
        head_dim = query.shape[2]

        # Pad blocks to uniform size for CUDA layout
        k_q_arr = np.zeros((num_blocks, block_size, head_dim), dtype=np.uint8)
        v_arr = np.zeros((num_blocks, block_size, head_dim), dtype=np.float32)
        scale_arr = np.zeros((num_blocks, head_dim), dtype=np.float32)
        zp_arr = np.zeros((num_blocks, head_dim), dtype=np.float32)

        for i in range(num_blocks):
            n = k_q_blocks[i].shape[0]
            k_q_arr[i, :n] = k_q_blocks[i]
            v_arr[i, :n] = v_blocks[i]
            scale_arr[i] = k_scales[i]
            zp_arr[i] = k_zps[i]

        out = _C.flash_decode_int4(
            torch.from_numpy(query.astype(np.float32)).cuda(),
            torch.from_numpy(k_q_arr).cuda(),
            torch.from_numpy(scale_arr).cuda(),
            torch.from_numpy(zp_arr).cuda(),
            torch.from_numpy(v_arr).cuda(),
            torch.from_numpy(block_lens.astype(np.int32)).cuda(),
        )
        return out.cpu().numpy()

    key_scale_zp = [(k_q_blocks[i], k_scales[i], k_zps[i]) for i in range(num_blocks)]
    output, _ = online_softmax_ref(query, key_scale_zp, v_blocks, block_lens)
    return output


def backend_status() -> dict:
    """Return current backend availability."""
    return {
        'has_torch': HAS_TORCH,
        'has_cuda': HAS_CUDA,
        'active_backend': 'cuda' if _use_cuda() else 'reference',
    }

"""
vLLM integration shim for Flash Decoding with INT4 KV.

Provides a paged-KV-cache wrapper that quantizes blocks on write and
runs fused INT4 attention on decode. Designed to slot behind vLLM's
attention backend interface.

Usage (inside a vLLM fork or plugin):

    from flash_decode_int4.vllm_integration import INT4PagedKVCache

    cache = INT4PagedKVCache(num_blocks=4096, block_size=256, head_dim=128)
    cache.write_block(block_id, k_block, v_block)   # quantizes K on write
    out = cache.decode_attention(query, block_table)  # fused INT4 attention
"""

import numpy as np
from typing import Dict, List, Optional

from .quantize_int4_ref import quantize_int4_ref
from . import ops


class INT4PagedKVCache:
    """Paged KV cache with INT4-quantized keys and FP32/FP16 values.

    Keys are quantized per-(block, channel) on write; values stay full
    precision (they contribute linearly and are more error-sensitive).
    """

    def __init__(self, num_blocks: int, block_size: int, head_dim: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.head_dim = head_dim

        self.k_q: Dict[int, np.ndarray] = {}      # block_id -> uint8 [block_size, head_dim]
        self.k_scale: Dict[int, np.ndarray] = {}  # block_id -> float32 [head_dim]
        self.k_zp: Dict[int, np.ndarray] = {}     # block_id -> float32 [head_dim]
        self.v: Dict[int, np.ndarray] = {}        # block_id -> float32 [block_size, head_dim]
        self.lens: Dict[int, int] = {}            # effective tokens per block

    def write_block(self, block_id: int, k_block: np.ndarray, v_block: np.ndarray) -> None:
        """Quantize and store one KV block."""
        assert k_block.shape == v_block.shape
        q, scale, zp = ops.quantize_int4(k_block.astype(np.float32))
        self.k_q[block_id] = q
        self.k_scale[block_id] = scale
        self.k_zp[block_id] = zp
        self.v[block_id] = v_block.astype(np.float32)
        self.lens[block_id] = k_block.shape[0]

    def free_block(self, block_id: int) -> None:
        for d in (self.k_q, self.k_scale, self.k_zp, self.v, self.lens):
            d.pop(block_id, None)

    def decode_attention(
        self,
        query: np.ndarray,          # [batch, heads, head_dim]
        block_table: List[int],
    ) -> np.ndarray:
        """Fused online-softmax attention over the given blocks."""
        k_qs = [self.k_q[b] for b in block_table]
        scales = [self.k_scale[b] for b in block_table]
        zps = [self.k_zp[b] for b in block_table]
        values = [self.v[b] for b in block_table]
        lens = np.array([self.lens[b] for b in block_table], dtype=np.int32)

        return ops.flash_decode(query, k_qs, scales, zps, values, lens)

    def memory_stats(self) -> dict:
        """Actual vs FP16-equivalent memory footprint."""
        n = len(self.k_q)
        if n == 0:
            return {'blocks': 0, 'int4_mb': 0.0, 'fp16_equiv_mb': 0.0, 'ratio': 0.0}

        int4_bytes = sum(
            q.nbytes // 2 + s.nbytes + z.nbytes  # INT4 packs 2/byte in production
            for q, s, z in zip(self.k_q.values(), self.k_scale.values(), self.k_zp.values())
        ) + sum(v.nbytes // 2 for v in self.v.values())  # values as FP16
        fp16_bytes = sum(q.size * 2 for q in self.k_q.values()) + \
                     sum(v.size * 2 for v in self.v.values())

        return {
            'blocks': n,
            'int4_mb': round(int4_bytes / 1048576, 2),
            'fp16_equiv_mb': round(fp16_bytes / 1048576, 2),
            'ratio': round(fp16_bytes / max(int4_bytes, 1), 2),
        }

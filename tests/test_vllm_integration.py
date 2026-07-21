"""Integration tests for the vLLM shim (INT4PagedKVCache)."""

import numpy as np
import pytest

from src.vllm_integration import INT4PagedKVCache
from src.flash_decode_ref import _online_softmax_fp32, compute_attention_accuracy


class TestINT4PagedKVCache:

    def test_write_and_decode(self):
        """Write blocks, run decode attention, verify output shape and finiteness."""
        np.random.seed(0)
        cache = INT4PagedKVCache(num_blocks=16, block_size=128, head_dim=64)

        for b in range(4):
            k = np.random.randn(128, 64).astype(np.float32)
            v = np.random.randn(128, 64).astype(np.float32)
            cache.write_block(b, k, v)

        query = np.random.randn(2, 4, 64).astype(np.float32)
        out = cache.decode_attention(query, block_table=[0, 1, 2, 3])

        assert out.shape == (2, 4, 64)
        assert np.isfinite(out).all()

    def test_int4_decode_close_to_fp32(self):
        """INT4 cache attention must stay close to FP32 attention.

        Query is scaled by 1/sqrt(head_dim) as real attention does —
        unscaled random logits (std ~8) make softmax pathologically
        peaked and amplify quantization noise unrealistically.
        """
        np.random.seed(1)
        head_dim = 64
        cache = INT4PagedKVCache(num_blocks=8, block_size=256, head_dim=head_dim)

        keys, values = [], []
        for b in range(3):
            k = np.random.randn(256, head_dim).astype(np.float32)
            v = np.random.randn(256, head_dim).astype(np.float32)
            cache.write_block(b, k, v)
            keys.append(k)
            values.append(v)

        query = (np.random.randn(1, 2, head_dim) / np.sqrt(head_dim)).astype(np.float32)
        out_int4 = cache.decode_attention(query, block_table=[0, 1, 2])
        out_fp32 = _online_softmax_fp32(
            query, keys, values, np.full(3, 256, dtype=np.int32))

        metrics = compute_attention_accuracy(out_fp32, out_int4)
        assert metrics['mae'] < 0.05, f"INT4 drift too large: {metrics}"

    def test_free_block(self):
        cache = INT4PagedKVCache(num_blocks=4, block_size=64, head_dim=32)
        k = np.random.randn(64, 32).astype(np.float32)
        cache.write_block(0, k, k)
        cache.free_block(0)
        assert 0 not in cache.k_packed
        assert cache.memory_stats()['blocks'] == 0

    def test_memory_compression_ratio(self):
        """Stored footprint (nibble-packed K + FP16 V, measured from the
        actual arrays) must beat all-FP16 by ~1.5x+ overall."""
        cache = INT4PagedKVCache(num_blocks=16, block_size=256, head_dim=128)
        for b in range(8):
            k = np.random.randn(256, 128).astype(np.float32)
            cache.write_block(b, k, k)

        stats = cache.memory_stats()
        assert stats['ratio'] > 1.5, f"compression ratio too low: {stats}"

    def test_packed_storage_is_real(self):
        """Keys must actually be stored nibble-packed: half the bytes of the
        unpacked layout, and attention output must be unaffected."""
        np.random.seed(7)
        cache = INT4PagedKVCache(num_blocks=4, block_size=128, head_dim=64)
        k = np.random.randn(128, 64).astype(np.float32)
        v = np.random.randn(128, 64).astype(np.float32)
        cache.write_block(0, k, v)

        assert cache.k_packed[0].nbytes == 128 * 64 // 2, \
            "keys are not stored packed"

        query = (np.random.randn(1, 2, 64) / 8.0).astype(np.float32)
        out = cache.decode_attention(query, block_table=[0])
        assert np.isfinite(out).all()

    def test_variable_block_lengths(self):
        """Partial (short) final block must work."""
        np.random.seed(2)
        cache = INT4PagedKVCache(num_blocks=4, block_size=128, head_dim=32)
        cache.write_block(0, np.random.randn(128, 32).astype(np.float32),
                          np.random.randn(128, 32).astype(np.float32))
        cache.write_block(1, np.random.randn(40, 32).astype(np.float32),
                          np.random.randn(40, 32).astype(np.float32))

        query = np.random.randn(1, 1, 32).astype(np.float32)
        out = cache.decode_attention(query, block_table=[0, 1])
        assert np.isfinite(out).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

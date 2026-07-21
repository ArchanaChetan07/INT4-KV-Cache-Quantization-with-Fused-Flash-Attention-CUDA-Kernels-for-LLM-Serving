"""
Dispatch-layer tests for Flash Decoding INT4.

Verifies ops routes correctly; CUDA parity checks are skipped when
no compiled extension is present.
"""

import numpy as np
import pytest

from src.quantize_int4_ref import quantize_int4_ref
from src.flash_decode_ref import online_softmax_ref
from src import ops


class TestDispatchReference:
    """Ops layer must produce reference results when reference is forced."""

    def test_quantize_dispatch(self):
        np.random.seed(1)
        kv = np.random.randn(256, 64).astype(np.float32)

        q, scale, zp = ops.quantize_int4(kv, use_cuda=False)
        q_ref, scale_ref, zp_ref = quantize_int4_ref(kv, per_channel=True)

        np.testing.assert_array_equal(q, q_ref)
        np.testing.assert_allclose(scale, scale_ref, rtol=1e-6)
        np.testing.assert_allclose(zp, zp_ref, rtol=1e-6)

    def test_flash_decode_dispatch(self):
        np.random.seed(2)
        batch, heads, dim = 2, 4, 32
        block_size, num_blocks = 128, 3

        query = np.random.randn(batch, heads, dim).astype(np.float32)
        k_qs, scales, zps, values = [], [], [], []
        for _ in range(num_blocks):
            k = np.random.randn(block_size, dim).astype(np.float32)
            v = np.random.randn(block_size, dim).astype(np.float32)
            q, s, z = quantize_int4_ref(k, per_channel=True)
            k_qs.append(q); scales.append(s); zps.append(z); values.append(v)

        out = ops.flash_decode(query, k_qs, scales, zps, values, use_cuda=False)

        key_scale_zp = list(zip(k_qs, scales, zps))
        lens = np.full(num_blocks, block_size, dtype=np.int32)
        out_ref, _ = online_softmax_ref(query, key_scale_zp, values, lens)

        np.testing.assert_allclose(out, out_ref, rtol=1e-5)

    def test_backend_status(self):
        status = ops.backend_status()
        assert status['active_backend'] in ('cuda', 'reference')


@pytest.mark.skipif(not ops.HAS_CUDA, reason="compiled CUDA extension not available")
class TestCUDAParity:
    """CUDA kernels must match reference (when compiled)."""

    def test_quantize_parity(self):
        np.random.seed(3)
        kv = np.random.randn(512, 128).astype(np.float32)

        q_cuda, scale_cuda, zp_cuda = ops.quantize_int4(kv, use_cuda=True)
        q_ref, scale_ref, zp_ref = quantize_int4_ref(kv, per_channel=True)

        # Rounding at bin edges may differ by 1; scales must match tightly
        np.testing.assert_allclose(scale_cuda, scale_ref, rtol=1e-4)
        assert np.mean(np.abs(q_cuda.astype(int) - q_ref.astype(int))) < 0.01

    def test_flash_decode_parity(self):
        np.random.seed(4)
        batch, heads, dim = 2, 4, 64
        block_size, num_blocks = 256, 4

        query = np.random.randn(batch, heads, dim).astype(np.float32)
        k_qs, scales, zps, values = [], [], [], []
        for _ in range(num_blocks):
            k = np.random.randn(block_size, dim).astype(np.float32)
            v = np.random.randn(block_size, dim).astype(np.float32)
            q, s, z = quantize_int4_ref(k, per_channel=True)
            k_qs.append(q); scales.append(s); zps.append(z); values.append(v)

        out_cuda = ops.flash_decode(query, k_qs, scales, zps, values, use_cuda=True)
        out_ref = ops.flash_decode(query, k_qs, scales, zps, values, use_cuda=False)

        np.testing.assert_allclose(out_cuda, out_ref, rtol=1e-3, atol=1e-4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

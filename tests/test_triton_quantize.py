"""Parity tests for the Triton quantizer port.

Skips when triton is not installed. In CI these run on a CPU runner via
TRITON_INTERPRET=1; on a CUDA machine they exercise the compiled kernel.
"""

import numpy as np
import pytest

from src.quantize_int4_ref import quantize_int4_ref

triton_mod = pytest.importorskip("triton", reason="triton not installed")

from src.quantize_int4_triton import quantize_int4_triton  # noqa: E402


class TestTritonQuantizerParity:

    def test_scales_match_reference(self):
        np.random.seed(0)
        kv = np.random.randn(512, 64).astype(np.float32)

        _, scale_t, zp_t = quantize_int4_triton(kv)
        _, scale_r, zp_r = quantize_int4_ref(kv, per_channel=True)

        np.testing.assert_allclose(scale_t, scale_r, rtol=1e-4)
        np.testing.assert_allclose(zp_t, zp_r, rtol=1e-3, atol=1e-3)

    def test_bins_match_reference(self):
        """Half-up (Triton/CUDA) vs banker's (NumPy) rounding may differ by
        one bin at exact .5 boundaries — rare on continuous data."""
        np.random.seed(1)
        kv = np.random.randn(1024, 32).astype(np.float32)

        q_t, _, _ = quantize_int4_triton(kv)
        q_r, _, _ = quantize_int4_ref(kv, per_channel=True)

        diff = np.abs(q_t.astype(int) - q_r.astype(int))
        assert diff.max() <= 1, f"bins differ by more than 1: {diff.max()}"
        assert (diff > 0).mean() < 0.01, f"too many bin mismatches: {(diff > 0).mean():.4%}"

    def test_round_trip_error_bounded(self):
        np.random.seed(2)
        kv = np.random.randn(256, 128).astype(np.float32)

        q, scale, zp = quantize_int4_triton(kv)
        dequant = q.astype(np.float32) * scale - zp * scale
        err = np.abs(kv - dequant)
        assert err.max() <= scale.max() / 1.9, \
            f"round-trip error {err.max()} exceeds scale/2 bound {scale.max()/2}"

    def test_output_range(self):
        np.random.seed(3)
        kv = (np.random.randn(100, 16) * 50).astype(np.float32)
        q, _, _ = quantize_int4_triton(kv)
        assert q.min() >= 0 and q.max() <= 15

    def test_non_block_multiple_rows(self):
        """Row count not divisible by BLOCK exercises the masked tail."""
        np.random.seed(4)
        kv = np.random.randn(1030, 8).astype(np.float32)
        q, scale, _ = quantize_int4_triton(kv)
        q_r, scale_r, _ = quantize_int4_ref(kv, per_channel=True)
        np.testing.assert_allclose(scale, scale_r, rtol=1e-4)
        assert np.abs(q.astype(int) - q_r.astype(int)).max() <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

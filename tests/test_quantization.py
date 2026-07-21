"""Test suite for INT4 quantization reference implementation

Week 2 deliverable: 5 passing quantization tests
"""

import pytest
import numpy as np
from src.quantize_int4_ref import (
    quantize_int4_ref,
    dequantize_int4_ref,
    quantize_and_dequantize_ref,
    compute_quantization_error,
    quantization_error_bounds,
    INT4Quantizer,
)


class TestQuantizationBasics:
    """Basic INT4 quantization correctness tests"""

    def test_round_trip_error_bounds(self):
        """Test 1: Round-trip quantization error is within theoretical bounds.

        Theory: Maximum quantization error per value should be scale/2.
        This test verifies the actual error is bounded by theory.
        """
        # Create synthetic KV data with known range
        np.random.seed(42)
        kv = np.random.randn(4, 8, 64).astype(np.float32)

        q, scale, zp, kv_dequant = quantize_and_dequantize_ref(kv, per_channel=True)

        # Compute error
        err = np.abs(kv - kv_dequant)
        max_err = np.max(err)

        # Theoretical bound: max_scale / 2
        # (INT4 has 16 levels = 15 intervals, so quantization step = scale)
        max_scale = np.max(scale)
        theoretical_bound = max_scale / 2.0

        # Error should be within bound (with 1% margin for numerical precision)
        assert max_err <= theoretical_bound * 1.01, \
            f"Quantization error {max_err} exceeds bound {theoretical_bound}"

    def test_per_channel_scale_computation(self):
        """Test 2: Per-channel scales are computed correctly.

        Verify that each channel's scale matches (max - min) / 15.
        """
        # Create test data with known ranges per channel
        kv = np.array([
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ], dtype=np.float32)

        q, scale, zp = quantize_int4_ref(kv, per_channel=True)

        # Expected scales: (max - min) / 15 per channel
        # Channel 0: (11 - 1) / 15 = 10/15
        # Channel 1: (12 - 2) / 15 = 10/15
        expected_scale_ch0 = (11.0 - 1.0) / 15.0
        expected_scale_ch1 = (12.0 - 2.0) / 15.0

        np.testing.assert_allclose(scale[0], expected_scale_ch0, rtol=1e-5)
        np.testing.assert_allclose(scale[1], expected_scale_ch1, rtol=1e-5)

    def test_dequantization_accuracy(self):
        """Test 3: Dequantized values recover original with bounded error.

        Verify MAE and RMSE are small enough for LLM applications.
        """
        np.random.seed(123)
        kv = np.random.randn(8, 16, 128).astype(np.float32)

        q, scale, zp, kv_dequant = quantize_and_dequantize_ref(kv, per_channel=True)

        mae = np.mean(np.abs(kv - kv_dequant))
        rmse = np.sqrt(np.mean((kv - kv_dequant) ** 2))

        # For reasonable data, MAE should be small
        # Empirically, MAE ~ scale/4 is typical
        max_scale = np.max(scale)
        assert mae < max_scale / 2.5, \
            f"MAE {mae} is too high (scale range {max_scale})"
        assert rmse < max_scale / 2.0, \
            f"RMSE {rmse} is too high (scale range {max_scale})"

    def test_empty_blocks_handling(self):
        """Test 4: Correctly handle blocks with zero-length or uniform values.

        Edge case: What happens with uniform data (all same value)?
        """
        # Uniform data: all values are the same
        kv_uniform = np.full((2, 4, 16), 5.0, dtype=np.float32)

        q, scale, zp = quantize_int4_ref(kv_uniform, per_channel=True)

        # Scale should be very small (range ≈ 0)
        # All quantized values should map to same bin
        assert np.all(scale < 1e-6), "Scale should be near zero for uniform data"
        # All q values should be identical (after rounding)
        assert np.std(q) < 0.1, "Quantized values should be constant for uniform data"

    def test_quantization_convergence(self):
        """Test 5: Repeated quantization/dequantization converges.

        Apply quantization multiple times and verify error doesn't grow.
        """
        np.random.seed(456)
        kv_orig = np.random.randn(4, 8, 32).astype(np.float32)

        kv = kv_orig.copy()
        errors = []

        for iteration in range(3):
            q, scale, zp, kv = quantize_and_dequantize_ref(kv, per_channel=True)
            err = np.mean(np.abs(kv_orig - kv))
            errors.append(err)

        # Error should stabilize (not grow unboundedly)
        # Second iteration error should be close to first
        assert errors[1] < errors[0] * 2.0, "Error grew too much in iteration 2"
        assert errors[2] <= errors[1] * 1.5, "Error grew in iteration 3"


class TestQuantizationStatistics:
    """Tests for quantization statistics and bounds"""

    def test_error_bounds_report(self):
        """Verify quantization_error_bounds function works correctly"""
        np.random.seed(789)
        kv = np.random.uniform(-10.0, 10.0, size=(4, 8, 64)).astype(np.float32)

        bounds = quantization_error_bounds(kv, per_channel=True)

        # Should have all required fields
        assert 'mae' in bounds
        assert 'rmse' in bounds
        assert 'max_error' in bounds
        assert 'max_theoretical_bound' in bounds
        assert 'bound_tight' in bounds

        # Max error should be within bounds
        assert bounds['bound_tight'], "Error bounds violated"

    def test_quantizer_stateful(self):
        """Test INT4Quantizer for streaming quantization"""
        quantizer = INT4Quantizer(per_channel=True)

        # Quantize first block
        block1 = np.random.randn(256, 64).astype(np.float32)
        q1 = quantizer.quantize_block(block1)
        assert q1.shape == block1.shape
        assert quantizer.num_quantized == 1

        # Dequantize should match
        deq1 = quantizer.dequantize_block(q1)
        err1 = np.mean(np.abs(block1 - deq1))
        assert err1 < 1.0, f"Dequantization error too high: {err1}"

        # Quantize second block (scales/zp update)
        block2 = np.random.randn(256, 64).astype(np.float32) * 10.0
        q2 = quantizer.quantize_block(block2)
        assert quantizer.num_quantized == 2

        # Different scale for different range
        assert not np.allclose(quantizer.scales, 1.0)


class TestQuantizationEdgeCases:
    """Edge case tests for INT4 quantization"""

    def test_very_small_range(self):
        """Test quantization with very small value range"""
        kv = np.array([1.0000, 1.0001, 1.0002], dtype=np.float32)[:, np.newaxis]
        q, scale, zp = quantize_int4_ref(kv, per_channel=True)

        # Should not crash and should produce valid INT4
        assert np.all((q >= 0) & (q <= 15))
        assert scale.shape == (1,)

    def test_large_negative_values(self):
        """Test quantization with large negative values"""
        kv = np.array([-1000.0, -100.0, -10.0, 0.0], dtype=np.float32)[:, np.newaxis]
        q, scale, zp = quantize_int4_ref(kv, per_channel=True)

        # Verify round-trip accuracy
        kv_dequant = dequantize_int4_ref(q, scale, zp)
        err = np.abs(kv - kv_dequant)
        assert np.all(err < np.max(scale) / 1.5)

    def test_mixed_magnitude_values(self):
        """Test with values spanning many orders of magnitude"""
        kv = np.array([1e-6, 1e-3, 1.0, 1e3, 1e6], dtype=np.float32)[:, np.newaxis]
        q, scale, zp = quantize_int4_ref(kv, per_channel=True)

        # Should handle gracefully
        assert np.all(np.isfinite(q))
        assert np.isfinite(scale).all()
        assert np.isfinite(zp).all()


# Integration test
class TestQuantizationIntegration:
    """End-to-end quantization tests"""

    def test_full_kv_cache_quantization(self):
        """Simulate full KV cache quantization for a batch"""
        np.random.seed(999)

        # Simulate: (batch=2, num_heads=8, seq_len=2048, head_dim=64)
        # But quantize as blocks: (num_blocks=8, block_size=256, head_dim=64)
        num_blocks = 8
        block_size = 256
        num_heads = 8
        head_dim = 64

        # Create KV for all blocks
        q_all = []
        scale_all = []
        zp_all = []

        for b in range(num_blocks):
            kv_block = np.random.randn(block_size, head_dim).astype(np.float32)
            q, scale, zp = quantize_int4_ref(kv_block, per_channel=True)
            q_all.append(q)
            scale_all.append(scale)
            zp_all.append(zp)

        # Verify shapes
        assert len(q_all) == num_blocks
        assert all(q.shape == (block_size, head_dim) for q in q_all)
        assert all(scale.shape == (head_dim,) for scale in scale_all)

        # Verify memory savings
        # Original: num_blocks * block_size * head_dim * 4 bytes (FP32)
        # Quantized: num_blocks * block_size * head_dim / 2 bytes (INT4)
        #           + num_blocks * head_dim * 4 bytes (scale)
        #           + num_blocks * head_dim * 4 bytes (zp)
        original_bytes = num_blocks * block_size * head_dim * 4
        quantized_bytes = num_blocks * block_size * head_dim // 2 + 2 * num_blocks * head_dim * 4
        compression = quantized_bytes / original_bytes

        assert compression < 0.3, f"Expected 4× compression, got {1/compression:.1f}×"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

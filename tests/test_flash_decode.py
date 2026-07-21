"""Test suite for Flash Decoding kernel

Week 5 deliverable: Attention kernel tests
"""

import pytest
import numpy as np
from src.flash_decode_ref import (
    online_softmax_ref,
    _online_softmax_fp32,
    compute_attention_accuracy,
    attention_correctness_test,
)
from src.quantize_int4_ref import quantize_int4_ref, dequantize_int4_ref


class TestOnlineSoftmaxFP32:
    """Test online softmax for FP32 (unquantized) attention"""

    def test_single_batch_single_head(self):
        """Test basic online softmax with minimal dimensions"""
        batch_size, num_heads, head_dim = 1, 1, 32
        seq_len = 128

        query = np.random.randn(batch_size, num_heads, head_dim).astype(np.float32)
        key = np.random.randn(seq_len, head_dim).astype(np.float32)
        value = np.random.randn(seq_len, head_dim).astype(np.float32)

        output = _online_softmax_fp32(
            query,
            [key],
            [value],
            block_lens=np.array([seq_len], dtype=np.int32)
        )

        assert output.shape == (batch_size, num_heads, head_dim)
        assert np.isfinite(output).all()

    def test_multi_batch_multi_head(self):
        """Test online softmax with realistic batch and head dimensions"""
        batch_size, num_heads, head_dim = 4, 8, 64
        seq_len = 2048
        block_size = 256

        query = np.random.randn(batch_size, num_heads, head_dim).astype(np.float32)

        num_blocks = (seq_len + block_size - 1) // block_size
        key_list = [np.random.randn(block_size, head_dim).astype(np.float32)
                    for _ in range(num_blocks)]
        value_list = [np.random.randn(block_size, head_dim).astype(np.float32)
                      for _ in range(num_blocks)]

        block_lens = np.full(num_blocks, block_size, dtype=np.int32)
        block_lens[-1] = seq_len - (num_blocks - 1) * block_size

        output = _online_softmax_fp32(query, key_list, value_list, block_lens)

        assert output.shape == (batch_size, num_heads, head_dim)
        assert np.isfinite(output).all()

    def test_single_block(self):
        """Test with single KV block (no online aspect)"""
        batch_size, num_heads, head_dim = 2, 4, 32
        seq_len = 512

        query = np.random.randn(batch_size, num_heads, head_dim).astype(np.float32)
        key = np.random.randn(seq_len, head_dim).astype(np.float32)
        value = np.random.randn(seq_len, head_dim).astype(np.float32)

        output = _online_softmax_fp32(query, [key], [value], np.array([seq_len]))

        # Should produce valid output
        assert output.shape == (batch_size, num_heads, head_dim)
        assert not np.any(np.isnan(output))

    def test_variable_block_lengths(self):
        """Test with variable-length blocks (some shorter than block_size)"""
        batch_size, num_heads, head_dim = 2, 4, 32
        block_size = 256

        query = np.random.randn(batch_size, num_heads, head_dim).astype(np.float32)

        # Create blocks with varying lengths
        block_lens = np.array([256, 256, 128], dtype=np.int32)
        key_list = [np.random.randn(block_lens[i], head_dim).astype(np.float32)
                    for i in range(len(block_lens))]
        value_list = [np.random.randn(block_lens[i], head_dim).astype(np.float32)
                      for i in range(len(block_lens))]

        output = _online_softmax_fp32(query, key_list, value_list, block_lens)

        assert output.shape == (batch_size, num_heads, head_dim)
        assert np.isfinite(output).all()

    def test_output_range(self):
        """Test that output values are in reasonable range"""
        batch_size, num_heads, head_dim = 2, 4, 32
        seq_len = 512

        query = np.random.randn(batch_size, num_heads, head_dim).astype(np.float32)
        key = np.random.randn(seq_len, head_dim).astype(np.float32)
        value = np.random.randn(seq_len, head_dim).astype(np.float32)

        output = _online_softmax_fp32(query, [key], [value], np.array([seq_len]))

        # Output should be reasonable (not too large/small)
        assert np.abs(output).max() < 100.0
        assert np.abs(output).mean() < 10.0


class TestOnlineSoftmaxQuantized:
    """Test online softmax with INT4 quantized keys"""

    def test_quantized_int4_attention(self):
        """Test attention with INT4 quantized KV"""
        batch_size, num_heads, head_dim = 2, 4, 32
        seq_len = 512

        np.random.seed(42)
        query = np.random.randn(batch_size, num_heads, head_dim).astype(np.float32)
        key = np.random.randn(seq_len, head_dim).astype(np.float32)
        value = np.random.randn(seq_len, head_dim).astype(np.float32)

        # Quantize key
        k_q, scale, zp = quantize_int4_ref(key, per_channel=True)

        # Create key_scale_zp tuple
        key_scale_zp_list = [(k_q, scale, zp)]
        value_list = [value]

        output, log_partition = online_softmax_ref(
            query,
            key_scale_zp_list,
            value_list,
            np.array([seq_len])
        )

        assert output.shape == (batch_size, num_heads, head_dim)
        assert np.isfinite(output).all()
        assert np.isfinite(log_partition).all()

    def test_quantization_accuracy_degradation(self):
        """Verify quantization introduces small accuracy loss"""
        batch_size, num_heads, head_dim = 2, 4, 32
        seq_len = 512

        np.random.seed(123)
        query = np.random.randn(batch_size, num_heads, head_dim).astype(np.float32)
        key = np.random.randn(seq_len, head_dim).astype(np.float32)
        value = np.random.randn(seq_len, head_dim).astype(np.float32)

        # FP32 reference
        output_fp32 = _online_softmax_fp32(query, [key], [value], np.array([seq_len]))

        # INT4 quantized
        k_q, scale, zp = quantize_int4_ref(key, per_channel=True)
        key_scale_zp_list = [(k_q, scale, zp)]
        output_int4, _ = online_softmax_ref(query, key_scale_zp_list, [value], np.array([seq_len]))

        # Compute error
        metrics = compute_attention_accuracy(output_fp32, output_int4)

        # INT4 quantization should introduce small but measurable error.
        # Random Gaussian keys are worst-case for quantization (no channel
        # structure to exploit); real LLM KV quantizes far tighter.
        assert metrics['mae'] > 0.0
        assert metrics['mae'] < 0.15, "Quantization error too large"
        assert metrics['rmse'] < 0.25, "RMSE too large"

    def test_multiple_quantized_blocks(self):
        """Test attention over multiple INT4 quantized blocks"""
        batch_size, num_heads, head_dim = 2, 4, 32
        block_size = 256
        num_blocks = 4

        query = np.random.randn(batch_size, num_heads, head_dim).astype(np.float32)

        key_scale_zp_list = []
        value_list = []

        for _ in range(num_blocks):
            key = np.random.randn(block_size, head_dim).astype(np.float32)
            value = np.random.randn(block_size, head_dim).astype(np.float32)

            k_q, scale, zp = quantize_int4_ref(key, per_channel=True)
            key_scale_zp_list.append((k_q, scale, zp))
            value_list.append(value)

        block_lens = np.full(num_blocks, block_size, dtype=np.int32)

        output, log_partition = online_softmax_ref(
            query,
            key_scale_zp_list,
            value_list,
            block_lens
        )

        assert output.shape == (batch_size, num_heads, head_dim)
        assert np.isfinite(output).all()


class TestAttentionNumerics:
    """Numerical stability and precision tests"""

    def test_numerically_stable_large_logits(self):
        """Test stability with large logit values (potential overflow)"""
        batch_size, num_heads, head_dim = 1, 1, 32
        seq_len = 128

        # Create query and key that will produce large logits
        query = np.ones((batch_size, num_heads, head_dim), dtype=np.float32) * 100.0
        key = np.ones((seq_len, head_dim), dtype=np.float32) * 100.0
        value = np.random.randn(seq_len, head_dim).astype(np.float32)

        output = _online_softmax_fp32(query, [key], [value], np.array([seq_len]))

        # Should not overflow or produce NaN
        assert np.isfinite(output).all()
        assert not np.any(np.isnan(output))

    def test_numerically_stable_small_logits(self):
        """Test stability with very small logit values (potential underflow)"""
        batch_size, num_heads, head_dim = 1, 1, 32
        seq_len = 128

        # Very small query/key
        query = np.ones((batch_size, num_heads, head_dim), dtype=np.float32) * 1e-6
        key = np.ones((seq_len, head_dim), dtype=np.float32) * 1e-6
        value = np.random.randn(seq_len, head_dim).astype(np.float32)

        output = _online_softmax_fp32(query, [key], [value], np.array([seq_len]))

        assert np.isfinite(output).all()

    def test_attention_with_mask_compatibility(self):
        """Test attention kernel with masking scenarios (conceptual)"""
        # Note: Masking is typically applied elsewhere in the system
        # This test verifies that blocks can have different effective lengths
        batch_size, num_heads, head_dim = 2, 4, 32

        query = np.random.randn(batch_size, num_heads, head_dim).astype(np.float32)

        # Block 1: full length
        key1 = np.random.randn(256, head_dim).astype(np.float32)
        value1 = np.random.randn(256, head_dim).astype(np.float32)

        # Block 2: partial length (conceptual masking)
        key2 = np.random.randn(256, head_dim).astype(np.float32)
        value2 = np.random.randn(256, head_dim).astype(np.float32)

        block_lens = np.array([256, 128], dtype=np.int32)  # Second block only 128 effective

        output = _online_softmax_fp32(query, [key1, key2], [value1, value2], block_lens)

        assert output.shape == (batch_size, num_heads, head_dim)
        assert np.isfinite(output).all()


class TestAttentionCorrectness:
    """End-to-end correctness tests"""

    def test_attention_correctness_suite(self):
        """Run full correctness test suite"""
        result = attention_correctness_test(
            batch_size=2,
            num_heads=8,
            head_dim=64,
            seq_len=2048,
            block_size=256
        )

        # Test should pass
        assert result['metrics']['test_passed']
        assert result['metrics']['mae'] < 1e-6

    def test_attention_convergence_across_blocks(self):
        """Test that multi-block attention converges correctly"""
        batch_size, num_heads, head_dim = 1, 1, 32
        block_size = 64
        num_blocks = 8

        np.random.seed(456)
        query = np.random.randn(batch_size, num_heads, head_dim).astype(np.float32)

        key_list = []
        value_list = []
        for _ in range(num_blocks):
            key_list.append(np.random.randn(block_size, head_dim).astype(np.float32))
            value_list.append(np.random.randn(block_size, head_dim).astype(np.float32))

        block_lens = np.full(num_blocks, block_size, dtype=np.int32)

        # Run online softmax
        output = _online_softmax_fp32(query, key_list, value_list, block_lens)

        # Run as single concatenated block for reference
        key_concat = np.concatenate(key_list, axis=0)
        value_concat = np.concatenate(value_list, axis=0)
        output_ref = _online_softmax_fp32(query, [key_concat], [value_concat],
                                         np.array([block_size * num_blocks]))

        # Outputs should match (within numerical precision)
        metrics = compute_attention_accuracy(output_ref, output)
        assert metrics['mae'] < 1e-5, f"Multi-block vs single-block diff too large: {metrics}"


class TestAttentionAccuracy:
    """Tests for accuracy metrics"""

    def test_accuracy_metrics_computation(self):
        """Test accuracy metric calculations"""
        output_ref = np.random.randn(2, 4, 32).astype(np.float32)
        output_test = output_ref + np.random.randn(2, 4, 32).astype(np.float32) * 0.01

        metrics = compute_attention_accuracy(output_ref, output_test)

        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'max_error' in metrics
        assert 'relative_error' in metrics

        assert metrics['mae'] > 0.0
        assert metrics['rmse'] >= metrics['mae']
        assert metrics['max_error'] >= metrics['mae']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

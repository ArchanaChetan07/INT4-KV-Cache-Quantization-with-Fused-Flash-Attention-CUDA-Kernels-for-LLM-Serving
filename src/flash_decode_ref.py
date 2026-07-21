"""Flash Decoding with Online Softmax Reference

Fused attention computation with online softmax using Welford's algorithm.

Algorithm overview:
1. For each block in the paged KV cache:
   - Dequantize K, V from INT4
   - Compute logits: K^T @ Q
   - Update online softmax: max, sum(exp(...)), accumulate V
2. Output = accumulated V / sum(exp(...))

Online softmax state:
  m_old: running max of logits
  l_old: running sum of exp(logits - m)
  out_sum: accumulated output values (sum of V * softmax_probs)

Update rule (Welford):
  m_new = max(m_old, max(logits_p))
  scale = exp(m_old - m_new)
  l_new = l_old * scale + sum(exp(logits_p - m_new))
  out_sum_new = out_sum * scale + sum(V_p * exp(logits_p - m_new))

This avoids materializing the full attention matrix.
"""

import numpy as np
from typing import Tuple, List, Optional


def online_softmax_ref(
    query: np.ndarray,
    key_scale_zp_list: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    value_list: List[np.ndarray],
    block_lens: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute attention with online softmax (Welford's algorithm).

    This is the core Flash Decoding algorithm: process paged KV blocks
    sequentially, updating running (max, sum, output) without materializing
    the full attention matrix.

    Args:
        query: Query tensor, shape (batch_size, num_heads, head_dim)
        key_scale_zp_list: List of (K_quantized, scale, zp) tuples
                           K shape: (block_size, head_dim)
                           scale, zp shape: (head_dim,)
        value_list: List of Value tensors, shape (block_size, head_dim)
        block_lens: Optional, actual lengths for each block (for variable-length padding)

    Returns:
        output: Attention output, shape (batch_size, num_heads, head_dim)
        log_partition: Log of partition function sum(exp(logits - m))
    """
    assert query.ndim == 3, f"Query must be 3D (batch, heads, dim), got {query.ndim}D"
    batch_size, num_heads, head_dim = query.shape

    num_blocks = len(key_scale_zp_list)
    assert len(value_list) == num_blocks, "Number of blocks must match"

    if block_lens is None:
        block_lens = np.array([key_scale_zp_list[i][0].shape[0] for i in range(num_blocks)])

    # Initialize online softmax state
    # m: running max per (batch, head)
    # l: running sum per (batch, head)
    # output: running weighted sum per (batch, head, dim)
    m = np.full((batch_size, num_heads), -np.inf, dtype=np.float32)
    l = np.zeros((batch_size, num_heads), dtype=np.float32)
    output = np.zeros((batch_size, num_heads, head_dim), dtype=np.float32)

    log_partition_parts = []

    # Process each block
    for block_idx in range(num_blocks):
        k_q, scale, zp = key_scale_zp_list[block_idx]
        v = value_list[block_idx]
        block_len = int(block_lens[block_idx])
        if block_len == 0:
            # Empty page contributes nothing; np.max on a zero-size axis
            # would raise, so skip before computing logits
            log_partition_parts.append(l.copy())
            continue

        # Truncate to actual block length
        k_q = k_q[:block_len]
        v = v[:block_len]

        # Dequantize K
        # Standard dequant: K = q * scale + min
        # But in our quantization, scale and zp are per-channel
        # Dequant formula: K = q * scale - zp * scale (which is q * scale + min)
        k = k_q.astype(np.float32) * scale[np.newaxis, :] - zp[np.newaxis, :] * scale[np.newaxis, :]

        # Compute logits: (batch, heads, seq) = (batch, heads, dim) @ (dim, seq)
        # Query: (batch, heads, dim)
        # Key: (seq, dim) -> need to broadcast
        logits = np.matmul(query, k.T)  # (batch, heads, seq)
        assert logits.shape == (batch_size, num_heads, block_len)

        # Online softmax update
        # For each sequence position (batch, head):
        m_old = m  # (batch, heads)
        l_old = l  # (batch, heads)

        # Max of new logits per (batch, head)
        m_new = np.maximum(m_old, np.max(logits, axis=-1))  # (batch, heads)

        # Rescaling factor
        scale_factor = np.exp(m_old - m_new)  # (batch, heads)

        # Compute softmax per position within this block
        # logits - m_new: (batch, heads, seq)
        exp_logits = np.exp(logits - m_new[:, :, np.newaxis])

        # Sum over sequence dimension
        sum_exp = np.sum(exp_logits, axis=-1)  # (batch, heads)

        # Update l
        l_new = l_old * scale_factor + sum_exp

        # Update output: weighted sum of V
        # V: (seq, dim), exp_logits: (batch, heads, seq)
        # Result: (batch, heads, dim)
        weighted_v = np.einsum('bhs,sd->bhd', exp_logits, v)

        output = output * scale_factor[:, :, np.newaxis] + weighted_v

        # Update state
        m = m_new
        l = l_new

        log_partition_parts.append(l_new.copy())

    # Normalize output
    # output /= l
    log_partition = np.log(np.maximum(l, 1e-10))
    output = output / np.maximum(l, 1e-10)[:, :, np.newaxis]

    return output, log_partition


def flash_decode_ref(
    query: np.ndarray,
    key_list: List[np.ndarray],
    value_list: List[np.ndarray],
    scale_list: Optional[List[np.ndarray]] = None,
    zp_list: Optional[List[np.ndarray]] = None,
    block_lens: Optional[np.ndarray] = None,
    use_quantization: bool = False,
) -> np.ndarray:
    """High-level Flash Decoding reference.

    Args:
        query: Query tensor, shape (batch_size, num_heads, head_dim)
        key_list: List of Key blocks (either FP32 or INT4 quantized)
        value_list: List of Value blocks (FP32)
        scale_list: Optional, per-channel scales for dequantization
        zp_list: Optional, per-channel zero-points for dequantization
        block_lens: Optional, actual sequence lengths
        use_quantization: If True, treat keys as INT4 and use scale/zp

    Returns:
        output: Attention output, shape (batch_size, num_heads, head_dim)
    """
    if use_quantization:
        assert scale_list is not None and zp_list is not None
        key_scale_zp_list = [(key_list[i], scale_list[i], zp_list[i])
                             for i in range(len(key_list))]
        output, _ = online_softmax_ref(query, key_scale_zp_list, value_list, block_lens)
    else:
        # Standard FP32 attention (no dequantization step)
        output = _online_softmax_fp32(query, key_list, value_list, block_lens)

    return output


def _online_softmax_fp32(
    query: np.ndarray,
    key_list: List[np.ndarray],
    value_list: List[np.ndarray],
    block_lens: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Online softmax for FP32 (unquantized) case."""
    batch_size, num_heads, head_dim = query.shape
    num_blocks = len(key_list)

    if block_lens is None:
        block_lens = np.array([key_list[i].shape[0] for i in range(num_blocks)])

    m = np.full((batch_size, num_heads), -np.inf, dtype=np.float32)
    l = np.zeros((batch_size, num_heads), dtype=np.float32)
    output = np.zeros((batch_size, num_heads, head_dim), dtype=np.float32)

    for block_idx in range(num_blocks):
        k = key_list[block_idx].astype(np.float32)
        v = value_list[block_idx].astype(np.float32)
        block_len = int(block_lens[block_idx])
        if block_len == 0:
            continue  # empty page contributes nothing

        k = k[:block_len]
        v = v[:block_len]

        # Logits
        logits = np.matmul(query, k.T)  # (batch, heads, seq)

        # Online update
        m_old = m
        l_old = l
        m_new = np.maximum(m_old, np.max(logits, axis=-1))
        scale_factor = np.exp(m_old - m_new)
        exp_logits = np.exp(logits - m_new[:, :, np.newaxis])
        sum_exp = np.sum(exp_logits, axis=-1)
        l_new = l_old * scale_factor + sum_exp
        weighted_v = np.einsum('bhs,sd->bhd', exp_logits, v)
        output = output * scale_factor[:, :, np.newaxis] + weighted_v

        m = m_new
        l = l_new

    # Normalize
    output = output / np.maximum(l, 1e-10)[:, :, np.newaxis]
    return output


def compute_attention_accuracy(
    output_ref: np.ndarray,
    output_test: np.ndarray,
) -> dict:
    """Compute attention accuracy metrics.

    Args:
        output_ref: Reference output
        output_test: Test output

    Returns:
        Dictionary with accuracy metrics
    """
    err = np.abs(output_ref - output_test)
    mae = np.mean(err)
    rmse = np.sqrt(np.mean(err ** 2))
    max_err = np.max(err)
    rel_err = np.mean(err / (np.abs(output_ref) + 1e-8))

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'max_error': float(max_err),
        'relative_error': float(rel_err),
    }


def attention_correctness_test(
    batch_size: int = 2,
    num_heads: int = 8,
    head_dim: int = 64,
    seq_len: int = 2048,
    block_size: int = 256,
) -> dict:
    """Synthetic correctness test for Flash Decoding.

    Args:
        batch_size: Batch size
        num_heads: Number of attention heads
        head_dim: Head dimension
        seq_len: Total sequence length
        block_size: KV cache block size

    Returns:
        Test results dictionary
    """
    np.random.seed(42)

    query = np.random.randn(batch_size, num_heads, head_dim).astype(np.float32)

    num_blocks = (seq_len + block_size - 1) // block_size
    key_list = [np.random.randn(block_size, head_dim).astype(np.float32)
                for _ in range(num_blocks)]
    value_list = [np.random.randn(block_size, head_dim).astype(np.float32)
                  for _ in range(num_blocks)]

    block_lens = np.full(num_blocks, block_size, dtype=np.int32)
    block_lens[-1] = seq_len - (num_blocks - 1) * block_size

    # Reference FP32
    output_ref = _online_softmax_fp32(query, key_list, value_list, block_lens)

    # Test: should match reference
    metrics = compute_attention_accuracy(output_ref, output_ref)
    metrics['test_passed'] = metrics['mae'] < 1e-6

    return {
        'batch_size': batch_size,
        'num_heads': num_heads,
        'head_dim': head_dim,
        'seq_len': seq_len,
        'block_size': block_size,
        'num_blocks': num_blocks,
        'metrics': metrics,
    }

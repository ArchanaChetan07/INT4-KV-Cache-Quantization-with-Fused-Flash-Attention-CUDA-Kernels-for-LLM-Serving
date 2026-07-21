"""
Benchmark suite for Flash Decoding INT4.

Measures quantization and attention throughput on the active backend.
Writes results to results/*.json.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quantize_int4_ref import quantize_int4_ref
from src.flash_decode_ref import online_softmax_ref, _online_softmax_fp32
from src import ops


def _time_op(fn, iters: int = 20) -> float:
    for _ in range(3):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    return ((time.perf_counter() - start) / iters) * 1000.0


def bench_quantization(num_rows: int = 4096, head_dim: int = 128) -> dict:
    kv = np.random.randn(num_rows, head_dim).astype(np.float32)
    ms = _time_op(lambda: quantize_int4_ref(kv, per_channel=True))

    # Memory analysis
    fp16_bytes = num_rows * head_dim * 2
    int4_bytes = num_rows * head_dim // 2 + 2 * head_dim * 4
    return {
        'op': 'quantize_int4',
        'shape': [num_rows, head_dim],
        'ms_per_call': ms,
        'fp16_bytes': fp16_bytes,
        'int4_bytes': int4_bytes,
        'compression_vs_fp16': round(fp16_bytes / int4_bytes, 2),
    }


def bench_attention_fp32(
    batch_size: int = 8, num_heads: int = 8, head_dim: int = 64,
    seq_len: int = 2048, block_size: int = 256
) -> dict:
    query = np.random.randn(batch_size, num_heads, head_dim).astype(np.float32)
    num_blocks = seq_len // block_size
    keys = [np.random.randn(block_size, head_dim).astype(np.float32) for _ in range(num_blocks)]
    values = [np.random.randn(block_size, head_dim).astype(np.float32) for _ in range(num_blocks)]
    lens = np.full(num_blocks, block_size, dtype=np.int32)

    ms = _time_op(lambda: _online_softmax_fp32(query, keys, values, lens), iters=10)
    return {
        'op': 'attention_fp32_reference',
        'batch': batch_size, 'heads': num_heads, 'head_dim': head_dim,
        'seq_len': seq_len, 'ms_per_call': ms,
    }


def bench_attention_int4(
    batch_size: int = 8, num_heads: int = 8, head_dim: int = 64,
    seq_len: int = 2048, block_size: int = 256
) -> dict:
    query = np.random.randn(batch_size, num_heads, head_dim).astype(np.float32)
    num_blocks = seq_len // block_size

    key_scale_zp = []
    values = []
    for _ in range(num_blocks):
        k = np.random.randn(block_size, head_dim).astype(np.float32)
        v = np.random.randn(block_size, head_dim).astype(np.float32)
        q, s, z = quantize_int4_ref(k, per_channel=True)
        key_scale_zp.append((q, s, z))
        values.append(v)
    lens = np.full(num_blocks, block_size, dtype=np.int32)

    ms = _time_op(lambda: online_softmax_ref(query, key_scale_zp, values, lens), iters=10)
    return {
        'op': 'attention_int4_reference',
        'batch': batch_size, 'heads': num_heads, 'head_dim': head_dim,
        'seq_len': seq_len, 'ms_per_call': ms,
    }


def main():
    print("=" * 60)
    print("Flash Decoding INT4 - Benchmark Suite")
    print("=" * 60)

    status = ops.backend_status()
    print(f"Backend: {status['active_backend']}")
    print()

    results = {'backend': status['active_backend'], 'benchmarks': []}

    for bench_fn in [bench_quantization, bench_attention_fp32, bench_attention_int4]:
        result = bench_fn()
        results['benchmarks'].append(result)
        print(f"{result['op']:30s}: {result['ms_per_call']:.3f} ms/call")

    quant = results['benchmarks'][0]
    print(f"\nINT4 compression vs FP16: {quant['compression_vs_fp16']}x")

    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, 'latency_bench.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == '__main__':
    main()

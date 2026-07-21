"""Flash Decoding with INT4 KV

Per-channel asymmetric INT4 quantization + fused online softmax
for decode-step attention in LLM serving.

Components:
- quantize_int4_ref.py: NumPy INT4 quantizer (Week 2)
- flash_decode_ref.py: NumPy online softmax reference (Week 2)
- csrc/flash_decode_int4.cu: CUDA kernel (Week 3-5)
- csrc/bindings.cpp: PyTorch integration (Week 3-5)

Expected impact:
- 75% memory vs FP16 (4× compression)
- 2.1–2.8× throughput improvement
- <0.3% perplexity delta (Llama 7B)
"""

__version__ = "0.1.0"

try:
    from .quantize_int4_ref import (
        quantize_int4_ref,
        dequantize_int4_ref,
    )
    HAS_QUANT_REF = True
except ImportError:
    HAS_QUANT_REF = False

try:
    from .flash_decode_ref import (
        flash_decode_ref,
        online_softmax_ref,
    )
    HAS_DECODE_REF = True
except ImportError:
    HAS_DECODE_REF = False

try:
    from . import _C
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    _C = None

__all__ = [
    'quantize_int4_ref',
    'dequantize_int4_ref',
    'flash_decode_ref',
    'online_softmax_ref',
    'HAS_QUANT_REF',
    'HAS_DECODE_REF',
    'HAS_CUDA',
]

def backend_info():
    """Print available backends"""
    print(f"Quantization reference: {'✓' if HAS_QUANT_REF else '✗'}")
    print(f"Flash decode reference: {'✓' if HAS_DECODE_REF else '✗'}")
    print(f"CUDA extension: {'✓' if HAS_CUDA else '✗'}")

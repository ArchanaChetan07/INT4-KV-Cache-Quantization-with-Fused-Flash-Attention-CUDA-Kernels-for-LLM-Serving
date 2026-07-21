# Contributing

## Development setup

```bash
pip install -e ".[dev]"
pytest tests/ -q          # CPU-only; GPU-gated tests skip without CUDA
```

With a CUDA GPU + nvcc, enable the JIT-compiled extension (PowerShell/cmd on Windows):

```powershell
$env:FLASH_DECODE_JIT_CUDA = "1"
pytest tests/ -q
```

The Triton port validates without a GPU via interpreter mode:

```bash
TRITON_INTERPRET=1 pytest tests/test_triton_quantize.py -v
```

## The one rule: reference parity

`src/quantize_int4_ref.py` and `src/flash_decode_ref.py` are the executable
specification. Any kernel change (CUDA or Triton) must keep the parity tests
passing — attention output within `MAE < 1e-4` of the reference, quantizer
scales within `rtol=1e-4` with at most 1-bin rounding differences. If you
change intended behavior, change the reference first, then make kernels match.

## Guidelines

- New kernels ship in pairs: NumPy reference + GPU implementation + parity test
- Accuracy gates are non-negotiable: quantization round-trip ≤ scale/2,
  multi-block ≡ concatenated attention at MAE < 1e-5
- Benchmarks that back README claims are committed to `results/` as JSON
- Measured numbers and design targets stay clearly separated in docs

## Reporting issues

Include your GPU model, CUDA version, PyTorch version, and whether the failure
occurs on the reference, CUDA, or Triton path.

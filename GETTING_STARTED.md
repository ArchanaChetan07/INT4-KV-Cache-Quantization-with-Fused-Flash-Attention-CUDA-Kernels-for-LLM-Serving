# Project 2: Flash Decoding with INT4 KV — Getting Started

**Complete folder ready for 16-week implementation**

---

## 📋 What's Included

- ✅ README.md — Project overview
- ✅ Folder structure (src/, tests/, csrc/, etc.)
- ✅ Build configuration (setup.py, pyproject.toml, CMakeLists.txt)
- ✅ Package templates

**Not yet (Week 1):**
- INT4 quantizer reference (Week 2)
- Flash decode reference (Week 2)
- CUDA kernels (Week 3-5)

---

## 🚀 Quick Start

```bash
cd INT4-KV-Cache-Quantization-with-Fused-Flash-Attention-CUDA-Kernels-for-LLM-Serving
pip install -e .
pytest tests/ -q          # CPU-only: 40 passed, 2 skipped
```

With a CUDA GPU + nvcc (run from PowerShell/cmd on Windows, not Git Bash):

```powershell
$env:FLASH_DECODE_JIT_CUDA = "1"   # JIT-compiles csrc/ on import (cached)
pytest tests/ -q                    # 42 passed, 0 skipped — CUDA parity active
```

---

## 📖 Week-by-Week Plan

### Week 1–2: INT4 Quantization Reference
**Goal:** Implement per-channel asymmetric INT4 quantizer

**Deliverables:**
- [ ] `src/quantize_int4_ref.py` (NumPy)
- [ ] 5 quantization tests
- [ ] Correctness validated

**Key insight:** Per-(block, head, channel) scale + zero-point

### Week 3–5: Flash Decode CUDA Kernel
**Goal:** Fused online softmax + INT4 dequantization

**Deliverables:**
- [ ] `csrc/flash_decode_int4.cu` (CUDA kernel)
- [ ] `csrc/bindings.cpp` (PyTorch)
- [ ] Attention kernel tests passing

**Key insight:** Online softmax Welford algorithm

### Week 6–10: Integration & Benchmarking
**Goal:** Latency validation + vLLM integration

**Deliverables:**
- [ ] Latency benchmarks (committed JSON)
- [ ] vLLM integration ready
- [ ] Performance tests passing

### Week 11: Real Model Validation
**Goal:** Measure perplexity on Llama models

**Deliverables:**
- [ ] Llama 7B: <0.3% PPL delta ✓
- [ ] Llama 13B: <0.3% PPL delta
- [ ] Llama 70B: <0.5% PPL delta

**Gate:** Perplexity must pass before Week 12+

### Week 12–16: Production Deployment
**Goal:** Production-ready code

**Deliverables:**
- [ ] Operations runbook
- [ ] Monitoring + alerting
- [ ] Deployment playbook

---

## 📊 Success Criteria

| Phase | Metric | Target |
|-------|--------|--------|
| **Week 2** | Quant tests | 5/5 passing |
| **Week 5** | Kernel tests | All passing |
| **Week 7** | Latency | 82–110ms @ 64×2048 |
| **Week 11** | Perplexity (7B) | <0.3% delta |
| **Week 11** | Perplexity (70B) | <0.5% delta |

---

## 🎯 This Week

1. Understand INT4 quantization concept
2. Review online softmax algorithm
3. Plan Week 2 implementation

Then read `Project2-Complete-Technical-Plan.md` (in related docs folder) for:
- Detailed CUDA specifications
- Code templates for Week 3+
- Testing strategy

---

**Status:** Ready for Week 1-2 planning  
**Next:** Implement INT4 quantizer  
**Timeline:** 16 weeks to production  

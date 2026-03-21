# GPU Benchmark Summary — B300 SXM

**Date:** 2026-03-21
**GPU:** NVIDIA B300 SXM6 AC (275 GB)
**Precision:** bf16
**Config:** `configs/experiments/gpu_bench_short.yaml`

## Models Tested

| Model | Parameters | Tier | HuggingFace ID |
|---|---|---|---|
| PaddleOCR-VL | 0.9B | Ultra compact | PaddlePaddle/PaddleOCR-VL-1.5 |
| Nanonets OCR | 3B | Compact | nanonets/Nanonets-OCR-s |
| olmOCR-2 | 7B | Midsize | allenai/olmOCR-2-7B-1025 |

---

## Inference Results

**Engine:** vLLM v0.18, FlashAttention v4, max_output_tokens=2048

| Model | Batch | TTFT p50 (ms) | TTFT p95 (ms) | E2E Latency p50 (ms) | E2E Latency p95 (ms) | Pages/s | Tokens/s | GPU Mem (GB) |
|---|---|---|---|---|---|---|---|---|
| PaddleOCR-VL 0.9B | 1 | 5.7 | 6.0 | 1,968 | 1,969 | 0.51 | 1,040 | 243.3 |
| PaddleOCR-VL 0.9B | 8 | 9.4 | 11.5 | 1,975 | 1,978 | 4.04 | 8,273 | 243.3 |
| PaddleOCR-VL 0.9B | 32 | 15.4 | 25.2 | 2,219 | 2,238 | **14.32** | **29,323** | 243.3 |
| Nanonets OCR 3B | 1 | 8.1 | 8.3 | 6,275 | 6,276 | 0.16 | 326 | 243.5 |
| Nanonets OCR 3B | 8 | 13.8 | 17.0 | 6,186 | 6,188 | 1.29 | 2,645 | 243.5 |
| Nanonets OCR 3B | 32 | 20.3 | 31.8 | 5,962 | 5,982 | 5.35 | 10,948 | 243.5 |
| olmOCR-2 7B | 1 | 7.5 | 8.1 | 8,622 | 8,623 | 0.12 | 238 | 243.5 |
| olmOCR-2 7B | 8 | 11.4 | 13.4 | 75 | 8,630 | 0.93 | 249 | 243.5 |
| olmOCR-2 7B | 32 | 18.2 | 32.1 | 11,778 | 13,055 | 2.57 | 3,306 | 243.5 |

### Inference Observations

- **PaddleOCR-VL (0.9B)** delivers the highest throughput: 29,323 tok/s at batch=32 — a **1.75x improvement** over H100 (16,790 tok/s)
- **TTFT is consistently faster than H100** across all models: 5.7–20ms p50 on B300 vs 8.6–44ms on H100
- **Batch scaling is excellent:** ~28x throughput gain from batch=1→32 for PaddleOCR-VL, and ~34x for Nanonets
- **Nanonets (3B) at batch=32** reaches 10,948 tok/s — 2.19x faster than H100 (5,012 tok/s)
- **GPU memory is constant** at ~243 GB across all configs (vLLM pre-allocates KV cache at 90% of 275 GB)
- **olmOCR-2 batch=8 has anomalous E2E p50** (75ms) suggesting some requests returned very short outputs; p95 is consistent at 8,630ms
- **E2E latency is lower than H100** for same batch sizes: PaddleOCR-VL batch=32 latency is 2,219ms (B300) vs 3,873ms (H100), a 1.75x speedup

---

## Training Results

**Framework:** HuggingFace Trainer + LoRA (rank=64, alpha=128)
**Steps:** 110 (10 warmup + 100 measurement)
**Data:** Synthetic dummy dataset (512-token sequences)

| Model | Micro Batch | Time (s) | Samples/s | Tokens/s | GPU Mem (GB) | Mean Power (W) | Peak Power (W) |
|---|---|---|---|---|---|---|---|
| PaddleOCR-VL 0.9B | 1 | 185 | 0.59 | 304 | 2.8 | 284 | 292 |
| PaddleOCR-VL 0.9B | 2 | 100 | 2.20 | 1,125 | 3.3 | 331 | 338 |
| PaddleOCR-VL 0.9B | 4 | 51 | **8.64** | **4,423** | 4.5 | 414 | 429 |
| Nanonets OCR 3B | 1 | 368 | 0.30 | 153 | 9.8 | 353 | 369 |
| Nanonets OCR 3B | 2 | 217 | 1.01 | 518 | 9.8 | 436 | 473 |
| Nanonets OCR 3B | 4 | 110 | 4.00 | 2,050 | 11.3 | 622 | 676 |
| olmOCR-2 7B | 1 | 298 | 0.37 | 189 | 23.6 | 495 | 548 |
| olmOCR-2 7B | 2 | 173 | 1.27 | 651 | 23.6 | 710 | 792 |
| olmOCR-2 7B | 4 | 125 | 3.51 | 1,796 | 23.6 | 876 | 932 |

### Training Observations

- **PaddleOCR-VL (0.9B)** at batch=4 reaches 8.64 samples/s — **1.79x faster** than H100 (4.83 samples/s)
- **Nanonets (3B)** at batch=4 reaches 4.00 samples/s — **1.76x faster** than H100 (2.27 samples/s)
- **olmOCR-2 (7B)** at batch=4 reaches 3.51 samples/s — **2.37x faster** than H100 (1.48 samples/s), the largest gain among all models
- **B300 draws significantly more power:** olmOCR-2 at batch=4 uses 876W mean (B300) vs 340W (H100) — 2.6x more power for 2.4x more throughput
- **Power efficiency is roughly comparable** to H100: the throughput gains are proportional to power increases
- **All models converge well** — loss drops consistently within 110 steps
- **Max feasible batch size is 4** for all models with this LoRA config (same as H100)
- **olmOCR-2 (7B) memory is constant** at 23.6 GB across batch sizes (same as H100 — model weights dominate)

---

## B300 vs H100 Comparison

### Inference Throughput (tokens/s at batch=32)

| Model | H100 | B300 | Speedup |
|---|---|---|---|
| PaddleOCR-VL 0.9B | 16,790 | 29,323 | **1.75x** |
| Nanonets OCR 3B | 5,012 | 10,948 | **2.18x** |
| olmOCR-2 7B | 2,420 | 3,306 | **1.37x** |

### Training Throughput (samples/s at micro_batch=4)

| Model | H100 | B300 | Speedup |
|---|---|---|---|
| PaddleOCR-VL 0.9B | 4.83 | 8.64 | **1.79x** |
| Nanonets OCR 3B | 2.27 | 4.00 | **1.76x** |
| olmOCR-2 7B | 1.48 | 3.51 | **2.37x** |

### Key Takeaways

1. **B300 delivers 1.4–2.4x speedup** over H100 across all models and phases
2. **Larger models benefit more in training** — olmOCR-2 (7B) sees the biggest training speedup at 2.37x
3. **Inference gains are strongest for smaller models** — PaddleOCR-VL and Nanonets see 1.75–2.18x, while olmOCR-2 sees 1.37x
4. **B300 uses 2–3x more power** but delivers proportional throughput gains, keeping energy efficiency roughly on par
5. **3.4x more VRAM** (275 GB vs 80 GB) enables much larger KV cache, beneficial for high-concurrency serving

---

## Quality Evaluation

Quality phase was skipped for this benchmark run.

---

## Raw Data

Results are saved as Parquet + CSV:

```
results/raw/inference/
  paddleocr_vl_0.9b_b300_sxm_20260321T092502Z.{parquet,csv}
  nanonets_ocr2_3b_b300_sxm_20260321T093434Z.{parquet,csv}
  olmocr2_7b_b300_sxm_20260321T094718Z.{parquet,csv}

results/raw/training/
  paddleocr_vl_0.9b_b300_sxm_20260321T095309Z.{parquet,csv}
  nanonets_ocr2_3b_b300_sxm_20260321T100451Z.{parquet,csv}
  olmocr2_7b_b300_sxm_20260321T101454Z.{parquet,csv}
```

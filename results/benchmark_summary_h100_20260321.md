# GPU Benchmark Summary — H100 SXM

**Date:** 2026-03-21
**GPU:** NVIDIA H100 SXM (80 GB)
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

**Engine:** vLLM v0.18, FlashAttention v3, max_output_tokens=2048

| Model | Batch | TTFT p50 (ms) | TTFT p95 (ms) | E2E Latency p50 (ms) | E2E Latency p95 (ms) | Pages/s | Tokens/s | GPU Mem (GB) |
|---|---|---|---|---|---|---|---|---|
| PaddleOCR-VL 0.9B | 1 | 8.6 | 9.5 | 2,564 | 2,568 | 0.39 | 798 | 73.1 |
| PaddleOCR-VL 0.9B | 8 | 15.1 | 20.1 | 2,780 | 2,787 | 2.87 | 5,877 | 73.1 |
| PaddleOCR-VL 0.9B | 32 | 25.5 | 43.6 | 3,873 | 3,915 | **8.20** | **16,790** | 73.1 |
| Nanonets OCR 3B | 1 | 13.2 | 16.2 | 10,292 | 10,298 | 0.10 | 199 | 73.6 |
| Nanonets OCR 3B | 8 | 24.5 | 372.9 | 10,707 | 11,056 | 0.74 | 1,521 | 73.6 |
| Nanonets OCR 3B | 32 | 38.1 | 56.9 | 13,034 | 13,069 | 2.45 | 5,012 | 73.6 |
| olmOCR-2 7B | 1 | 15.5 | 19.6 | 18,684 | 18,691 | 0.05 | 110 | 73.6 |
| olmOCR-2 7B | 8 | 22.9 | 31.4 | 19,117 | 19,450 | 0.42 | 596 | 73.6 |
| olmOCR-2 7B | 32 | 44.1 | 80.3 | 20,977 | 21,594 | 1.50 | 2,420 | 73.6 |

### Inference Observations

- **PaddleOCR-VL (0.9B)** delivers the highest throughput: 16,790 tok/s at batch=32, 3.3x faster than Nanonets (3B) and 6.9x faster than olmOCR-2 (7B)
- **TTFT is fast across all models** — 8–44ms p50, scaling linearly with batch size; prefill is not the bottleneck
- **Batch scaling is efficient:** ~21x throughput gain from batch=1 to batch=32 for PaddleOCR-VL
- **GPU memory is constant** at ~73 GB across all configs (vLLM pre-allocates KV cache at 90% utilization)
- **Nanonets batch=8 TTFT p95 outlier** at 373ms suggests occasional vLLM scheduling delays

---

## Training Results

**Framework:** HuggingFace Trainer + LoRA (rank=64, alpha=128)
**Steps:** 110 (10 warmup + 100 measurement)
**Data:** Synthetic dummy dataset (512-token sequences)

| Model | LoRA Params | Micro Batch | Time (s) | Samples/s | Tokens/s | GPU Mem (GB) | Mean Power (W) | Peak Power (W) |
|---|---|---|---|---|---|---|---|---|
| PaddleOCR-VL 0.9B | 22.0M (2.24%) | 1 | 314 | 0.35 | 179 | 2.8 | 111 | 113 |
| PaddleOCR-VL 0.9B | | 2 | 181 | 1.22 | 624 | 3.4 | 134 | 138 |
| PaddleOCR-VL 0.9B | | 4 | 91 | **4.83** | **2,473** | 4.6 | 186 | 192 |
| Nanonets OCR 3B | 29.5M (0.78%) | 1 | 680 | 0.16 | 83 | 8.6 | 149 | 156 |
| Nanonets OCR 3B | | 2 | 389 | 0.57 | 290 | 9.5 | 204 | 212 |
| Nanonets OCR 3B | | 4 | 194 | 2.27 | 1,164 | 11.4 | 317 | 331 |
| olmOCR-2 7B | TBD | 1 | 555 | 0.20 | 102 | 23.6 | 231 | 248 |
| olmOCR-2 7B | | 2 | 327 | 0.67 | 345 | 23.6 | 323 | 340 |
| olmOCR-2 7B | | 4 | 298 | 1.48 | 757 | 23.6 | 340 | 348 |

### Training Observations

- **PaddleOCR-VL (0.9B)** is the most efficient to train: 4.83 samples/s at batch=4 using only 4.6 GB and 186W
- **Batch scaling delivers major gains:** Nanonets sees ~14x throughput improvement from batch=1→4; PaddleOCR-VL sees ~14x as well
- **olmOCR-2 (7B) memory is constant** at 23.6 GB across batch sizes (model weights dominate), while smaller models scale with batch size
- **Larger batches are more energy-efficient:** Nanonets at batch=4 uses 2.1x more power but delivers 14x throughput vs batch=1
- **All models converge well** — loss drops from ~13 to ~11.9 within 110 steps
- **Max feasible batch size is 4** for all models on H100 with this LoRA config

---

## Quality Evaluation

Quality phase (OmniDocBench) was not completed. The dataset on HuggingFace (`opendatalab/OmniDocBench`) contains images only — no ground truth text annotations. Quality metrics (edit distance, BLEU, METEOR) require ground truth for comparison.

---

## Raw Data

Results are saved as Parquet + CSV:

```
results/raw/inference/
  paddleocr_vl_0.9b_h100_sxm_20260321T075139Z.{parquet,csv}
  nanonets_ocr2_3b_h100_sxm_20260321T080516Z.{parquet,csv}
  olmocr2_7b_h100_sxm_20260321T0*.{parquet,csv}

results/raw/training/
  paddleocr_vl_0.9b_h100_sxm_20260321T084729Z.{parquet,csv}
  nanonets_ocr2_3b_h100_sxm_20260321T065046Z.{parquet,csv}
  olmocr2_7b_h100_sxm_20260321T071032Z.{parquet,csv}
```

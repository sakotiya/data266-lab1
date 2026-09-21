# Task 1 - Results: GPT-style LM from scratch (zoheb_waghu)

> **Metrics files.** `metrics_report.csv` uses the **team-agreed column schema** so my numbers
> line up with my teammate's. The team header omits some metrics the brief requires, so
> `metrics_report_extended.csv` carries the full set alongside it (see the table below for
> which columns are extra).
>
> Extra in the extended table: `model_name`, `config_path`, `device`.


Run `t1_baseline_20260919-232507` · config [configs/gpt_baseline.yaml](configs/gpt_baseline.yaml)
· checkpoint `t1_baseline_20260919-232507_best.pt` · raw log [t1_baseline_20260919-232507.jsonl](../../reproducibility/raw_logs/zoheb_waghu/task1_llm/t1_baseline_20260919-232507.jsonl)

## 1. Architecture

| Component | Choice | Why this choice |
|---|---|---|
| Transformer blocks | 4 | Enough depth to compose character n-grams into word-level structure; deeper overfits a 100K corpus faster than it helps |
| Attention heads | 4 (64 dims each) | 64 dims per head is the smallest width that keeps the dot-product scale meaningful; 4 heads let separate heads track position, word boundaries and recent-token identity |
| Embedding width | 256 | Sets total capacity at 3.26M params - large relative to a 100K-character corpus, which the 0.247 generalization gap reflects |
| Positional embedding | learned | Sequences are fixed at 256 and never extrapolate beyond it, so the extrapolation advantage of sinusoidal encodings buys nothing here |
| Dropout | 0.1 | Light, because the run is short (1,440 steps); heavier dropout slowed convergence without closing the gap |
| Context length | 256 chars | ~1-2 sentences of TinyStories. Longer context costs attention time quadratically for history the model has too little data to exploit |
| Normalisation | pre-LN | Keeps the residual path unnormalised so gradients reach early blocks; makes the model trainable with a short 150-step warm-up |
| Parameter count | **3,262,464** | |

Hand-written components - no `nn.Transformer*`, no `nn.MultiheadAttention`, no fused SDPA
(see [src/model.py](src/model.py)):

- [x] scaled dot-product attention with explicit causal mask (`CausalSelfAttention.forward`)
- [x] multi-head projection, split and merge (single fused QKV linear, then `view`/`transpose`)
- [x] pre-LN placement, residual connections (`Block.forward`)
- [x] position-wise feed-forward network, 4x inner width, GELU
- [x] learnable token and positional embedding tables
- [x] LM head projecting to vocabulary size

## 2. Data

- TinyStories, streamed and shuffled (seed 1337): **100,000** training characters / **10,000** validation
- Vocabulary: **65** characters, built from the **training split only**; validation characters
  unseen in training map to an explicit `UNK` slot rather than silently extending the vocab
- Targets are inputs shifted right by one; preprocessing artifacts in [data_processed/](data_processed/)

**Epoch definition (stated because epoch count is graded):** a non-overlapping pass over 100K
characters is only ~6 batches, which is far too few optimiser steps. An epoch here is defined
explicitly as **120 randomly-offset batches** of 32×256 tokens = 983K tokens ≈ 9.8 passes over
the corpus. 12 epochs = **1,440 optimiser steps**.

## 3. Training

| Hyperparameter | Value | Why |
|---|---|---|
| Epochs | 12 | Above the required minimum of 10; validation loss was still falling at 12 |
| Batch size | 32 | 32×256 = 8,192 tokens per step, the largest that kept MPS memory near 2.3 GB |
| Optimiser | AdamW, β=(0.9, 0.95), wd=0.1 | β₂=0.95 over the default 0.999 for stability at small batch |
| LR schedule | 3e-4, 150-step warm-up → cosine → 3e-5 | Warm-up prevents the early large-gradient step from destabilising the untrained LayerNorms |
| Grad clip | 1.0 | Max observed grad norm was 9.82, so clipping was active and did real work |

Curves: [outputs/plots/loss_curves_t1_baseline_20260919-232507.png](outputs/plots/loss_curves_t1_baseline_20260919-232507.png)
· gap: [outputs/plots/generalization_gap_t1_baseline_20260919-232507.png](outputs/plots/generalization_gap_t1_baseline_20260919-232507.png)

## 4. Correctness checks

- **Causal mask probe (behavioural, not visual).** Perturbed the token at position j=128 across
  5 random sequences and measured logit movement at every position:
  `max_logit_delta_before_j = 0.0` (bit-identical), `min_logit_delta_from_j = 0.7062` (positions
  at and after j do move). A non-zero delta below j would mean the mask leaks future information.
  Implemented in [src/evaluate.py](src/evaluate.py) `causal_mask_probe`; the training run aborts
  if it fails.
- **Loss at initialisation.** Expected `ln(65) = 4.174` for a uniform distribution over 65
  characters; observed 4.23 at step 0 - the model starts correctly uninformed.

## 5. Metrics

Full row in [metrics_report.csv](metrics_report.csv).

| Metric | Value |
|---|---|
| Training cross-entropy | 1.17207 |
| Validation cross-entropy | 1.41941 |
| Perplexity | 4.1347 |
| Bits-per-character | 2.04778 |
| Generalization gap | 0.24734 |
| Top-1 next-character accuracy | 0.58264 |
| Distinct-1 / 2 / 3 | 0.01386 / 0.10770 / 0.29493 |
| Repeated 4-gram rate | 0.53188 |
| Gradient norm mean / max | 1.38882 / 9.81623 |
| Loss spikes / NaN count | 0 / 0 |
| Parameter count | 3,262,464 |
| Training throughput | 56,194 tokens/sec |
| Generation throughput | 95.3 tokens/sec |
| Peak memory | 2.345 GB |
| Total training time | 209.92 s |
| Device | MPS (Apple M5, arm64) |

Training was stable: zero NaNs, zero loss spikes, and validation loss decreased monotonically
across all 12 epochs (2.264 → 1.419).

## 6. Generation

Samples: [outputs/samples/samples_t1_baseline_20260919-232507.txt](outputs/samples/samples_t1_baseline_20260919-232507.txt)
- 3 prompts × 3 strategies (greedy, temperature 0.8, temperature 1.2), 400 new characters each.

| Strategy | distinct-1 | distinct-2 | distinct-3 | repeated 4-gram |
|---|---|---|---|---|
| greedy | 0.0287 | 0.1434 | 0.2464 | 0.6824 |
| temperature 0.8 | 0.0342 | 0.2207 | 0.4984 | 0.3081 |
| temperature 1.2 | 0.0383 | 0.2690 | 0.6062 | 0.1894 |

Diversity rises and repetition falls monotonically with temperature, and coherence moves the
opposite way - see [failure_analysis.md](failure_analysis.md).

## 7. Hardware disclosure

Apple M5, arm64, 16 GB unified memory · PyTorch 2.5.1 on MPS · Python 3.9.6 (arm64).
Peak memory is `torch.mps.driver_allocated_memory` sampled at end of training - MPS exposes no
true peak counter, unlike CUDA's `max_memory_allocated`.

## 8. Limitations and what I would try next

1. **Corpus size is the binding constraint.** 100K characters is the assignment's figure, but a
   3.26M-parameter model against it produces a 0.247 generalization gap. First experiment: 1M
   characters, same architecture, and check whether bits-per-character falls below 2.048.
2. **No nucleus (top-p) sampling.** Greedy loops and temperature 1.2 is incoherent; top-p would
   likely dominate both, and the metric harness already measures the difference.
3. **Encoding artifacts survive preprocessing.** The model emits `â€` because mis-decoded UTF-8
   is in the training text. Normalising or filtering non-ASCII before vocabulary construction is
   a one-line fix with a measurable effect on vocabulary size.
4. **Single seed.** All numbers are one run; no variance estimate.

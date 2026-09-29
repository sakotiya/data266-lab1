# Task 1 - Results: GPT-style LM from scratch (zoheb_waghu)

> **Metrics files.** `metrics_report.csv` uses the **team-agreed column schema** so my numbers
> line up with my teammate's. The team header omits some metrics the brief requires, so
> `metrics_report_extended.csv` carries the full set alongside it.
>
> Extra in the extended table: `model_name`, `config_path`, `device`.

Run `t1_baseline_20260928-151435` · config [configs/gpt_baseline.yaml](configs/gpt_baseline.yaml)
· checkpoint `t1_baseline_20260928-151435_best.pt`
· raw log [t1_baseline_20260928-151435.jsonl](../../reproducibility/raw_logs/zoheb_waghu/task1_llm/t1_baseline_20260928-151435.jsonl)

## 1. Architecture

| Component | Choice | Why this choice |
|---|---|---|
| Transformer blocks | 4 | Enough depth to compose characters into word- and clause-level structure. Deliberately shallow-and-wide against my teammate's deep-and-narrow 12x8 model, so the team comparison isolates depth |
| Attention heads | 4 (64 dims each) | 64 dims per head keeps the dot-product scale meaningful; 4 heads let separate heads track position, word boundaries and recent-token identity |
| Embedding width | 256 | 3.27M params. At 25.6M training characters this is ~7.8 characters of data per parameter, which the 0.0038 generalization gap shows is not capacity-starved |
| Positional embedding | learned | Sequences are fixed at 256 and never extrapolate beyond it, so sinusoidal encodings' extrapolation advantage buys nothing |
| Dropout | 0.1 | Light on purpose: with 256x more data than my first run, regularisation is no longer what limits this model |
| Context length | 256 chars | ~1-2 sentences. Longer context costs attention time quadratically; §1.4 Case 3 argues this is now the binding constraint |
| Normalisation | pre-LN | Keeps the residual path unnormalised so gradients reach early blocks |
| Parameter count | **3,274,752** | |

Hand-written components - no `nn.Transformer*`, no `nn.MultiheadAttention`, no fused SDPA
(see [src/model.py](src/model.py)):

- [x] scaled dot-product attention with explicit causal mask (`CausalSelfAttention.forward`)
- [x] multi-head projection, split and merge (fused QKV linear, then `view`/`transpose`)
- [x] pre-LN placement and residual connections (`Block.forward`)
- [x] position-wise feed-forward network, 4x inner width, GELU
- [x] learnable token and positional embedding tables
- [x] LM head projecting to vocabulary size

## 2. Data

Spec 1.1 asks for "training (100K) and validation (10K)". **The team reads those counts as
sequences**, so at `block_size` 256 with non-overlapping windows (`stride` 256):

| | sequences | characters |
|---|---|---|
| Training | 100,000 | 25,600,001 |
| Validation | 10,000 | 2,559,763 |

- Vocabulary: **97** characters, built from the **training split only**. Validation characters
  unseen in training map to an explicit `UNK` slot; measured OOV was **0**.
- Targets are inputs shifted right by one.

**Each member creates their own split.** Both members read one canonical download,
`task1_llm/data/tinystories_raw.txt`, at different offsets: shreya_akotiya reads chars
[0, 34M), mine starts at `member_offset_chars: 34000000`, so the two training slices are
**disjoint** rather than merely differently shuffled.

> **Correction from my first submission.** I originally read "100K/10K" as *characters* and
> trained on 100,000 characters - 256x less text. That run reached validation bits-per-char
> 2.0478 with a 0.2473 generalization gap. The numbers below supersede it entirely; the
> comparison is kept in [failure_analysis.md](failure_analysis.md) because the contrast between
> the two runs is itself evidence about what data scale does and does not fix.

## 3. Training

| Hyperparameter | Value | Why |
|---|---|---|
| Epochs | 10 | Spec minimum. Validation loss was still falling at epoch 10 (see below), so this is a budget limit, not a convergence point |
| Steps | 3,125/epoch, **31,250 total** | One epoch is one genuine pass over all 100K sequences at batch 32 |
| Batch size | 32 | 8,192 tokens/step; differs from my teammate's 64 |
| Optimiser | AdamW, β=(0.9, 0.95), wd=0.1 | β₂=0.95 over the default for stability at small batch |
| LR schedule | 3e-4, 1,000-step warm-up → cosine → 3e-5 | Warm-up is ~1/3 of one epoch, scaled up from the 150 steps that suited my much shorter first run |
| Grad clip | 1.0 | Max observed grad norm 9.74, so clipping was active |

Validation loss decreased **monotonically across all 10 epochs**, 1.0205 → 0.7029
(bits-per-char 1.4722 → 1.0140):

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| val bpc | 1.4722 | 1.2715 | 1.1873 | 1.1321 | 1.0915 | 1.0630 | 1.0456 | 1.0310 | 1.0200 | 1.0140 |

Curves: [outputs/plots/loss_curves_t1_baseline_20260928-151435.png](outputs/plots/loss_curves_t1_baseline_20260928-151435.png)
· gap: [outputs/plots/generalization_gap_t1_baseline_20260928-151435.png](outputs/plots/generalization_gap_t1_baseline_20260928-151435.png)

## 4. Correctness checks

- **Causal mask probe (behavioural, not visual).** Perturbed the token at position j=128 across
  5 random sequences: `max_logit_delta_before_j = 0.0` (bit-identical), `min_logit_delta_from_j
  = 0.8473`. A non-zero delta below j would mean the mask leaks future information. Implemented
  in [src/evaluate.py](src/evaluate.py) `causal_mask_probe`; **the training run aborts if it
  fails**.
- **Loss at initialisation.** Expected `ln(97) = 4.5747` for a uniform distribution over 97
  characters; the run began in that region, confirming the model starts correctly uninformed.

## 5. Metrics

Full row in [metrics_report.csv](metrics_report.csv) (team schema) and
[metrics_report_extended.csv](metrics_report_extended.csv).

| Metric | Value |
|---|---|
| Training cross-entropy | 0.69909 |
| Validation cross-entropy | 0.70286 |
| Perplexity | 2.0195 |
| Bits-per-character | 1.01401 |
| Generalization gap | **0.00377** |
| Top-1 next-character accuracy | 0.77719 |
| Distinct-1 / 2 / 3 | 0.01253 / 0.09571 / 0.26400 |
| Repeated 4-gram rate | 0.58229 |
| Gradient norm mean / max | 0.74907 / 9.74310 |
| Loss spikes / NaN count | 0 / 0 |
| Parameter count | 3,274,752 |
| Training throughput | 37,193 tokens/sec |
| Generation throughput | 114.76 tokens/sec |
| Peak memory | 2.303 GB |
| Total training time | 6,882.9 s (1 h 55 m) |
| Device | MPS (Apple M5, arm64) |

Training was stable: zero NaNs, zero loss spikes, monotonic validation improvement.

The **0.0038 generalization gap** is the headline: train 0.69909 against validation 0.70286
means this model is not memorising its training slice at all. Combined with validation loss
still falling at epoch 10, the evidence says the limit here is compute budget and model
capacity, not data.

## 6. Generation

Samples: [outputs/samples/samples_t1_baseline_20260928-151435.txt](outputs/samples/samples_t1_baseline_20260928-151435.txt)
- 3 prompts × 3 strategies (greedy, temperature 0.8, temperature 1.2), 400 new characters each.

| Strategy | distinct-1 | distinct-2 | distinct-3 | repeated 4-gram |
|---|---|---|---|---|
| greedy | 0.0263 | 0.1442 | 0.2648 | 0.6488 |
| temperature 0.8 | 0.0295 | 0.2016 | 0.4195 | 0.4206 |
| temperature 1.2 | 0.0359 | 0.2346 | 0.5511 | 0.2478 |

Diversity rises and repetition falls monotonically with temperature; coherence moves the
opposite way - see [failure_analysis.md](failure_analysis.md).

## 7. Hardware disclosure

Apple M5, arm64, 16 GB unified memory · PyTorch 2.5.1 · Python 3.9.6 (arm64) · device `mps`.
Peak memory is `torch.mps.driver_allocated_memory` sampled at end of training - MPS exposes no
true peak counter, unlike CUDA's `max_memory_allocated`, so this figure is **not** directly
comparable with a CUDA-trained member's.

## 8. Limitations and what I would try next

1. **Context length is now the binding constraint**, not data. With a 0.0038 gap, all three
   failure cases in §1.4 are failures of state: the model cannot see beyond 256 characters. Next
   experiment: 512 or 1024 context at the same parameter count.
2. **Under-trained, not over-trained.** Validation loss fell every epoch to the last. More
   epochs would still help; 10 was the spec minimum and the wall-clock budget on this machine.
3. **No nucleus (top-p) sampling.** Greedy loops at 0.6488 repeated 4-grams and temperature 1.2
   breaks semantics. Top-p is the obvious next decoding experiment and the harness already
   measures the difference.
4. **Document-boundary masking not implemented.** Sequences can straddle two stories, which
   §1.4 Case 3 shows the model has learned to imitate by restarting mid-generation.
5. **Single seed.** All numbers are one run; no variance estimate.

# DATA 266 Lab 1 — Team 32 Report

**Team members:** Shreya Akotiya (`shreya_akotiya`), Zoheb Waghu (`zoheb_waghu`)  
**Repository:** https://github.com/sakotiya/data266-lab1

## Team ownership statement

Each of us designed, implemented, trained, and evaluated our own models in our own folders. We
kept our code and experiment results separate, then compared the results after both members had
finished their runs.

Before training, we agreed on the main shared decisions. We interpreted 100K/10K as the number
of sequences, used non-overlapping data slices for Task 1, built vocabularies from training data
only, and used the same `metrics_report.csv` format. This made it easier to place our results
side by side. The comparison sections below summarize what we found together.

### Individual contributions

**Shreya's contribution**

I built and trained the 12-layer deep-narrow character GPT for Task 1. I also trained three Yelp
sentiment models for Task 2: a mean-pooling baseline, a TextCNN, and a BiLSTM. For Task 3, I
trained a CycleGAN with UNet generators and PatchGAN discriminators. My work is
organized around configuration-driven notebooks, and I prepared the preprocessing, training
logs, metrics, generated samples, and failure analyses for my models.

**Zoheb's contribution**

I built and trained the 4-layer shallow-wide character GPT for Task 1. For Task 2, I trained a
BiLSTM mean-pooling baseline, a TextCNN, and an attention-based BiLSTM. For Task 3, I trained a
CycleGAN with ResNet-9 generators and 70 × 70 PatchGAN discriminators. I organized my work using
Python source modules and notebooks, and prepared the checkpoints, metrics, plots, logs, and
error reviews for my models.

We agreed to use the same main evaluation format so that the results could be compared fairly.

---

## Task 1 — GPT-style character language model from scratch

### 1.1 Shared setup

Both models follow the same rules, so differences between them come from the design choices in
§1.2.

- **Data.** TinyStories, character-level. Both members read the same raw file at **different,
  non-overlapping offsets**: Shreya chars `[0, 34M)`, Zoheb chars from `34M`. Each member
  therefore has their own train/validation split.
- **Split size.** "Training (100K) and validation (10K)" is read as **sequences** of 256
  characters: 100,000 train sequences (25.6M characters) and 10,000 validation sequences
  (2.56M characters). Targets are inputs shifted by one.
- **Vocabulary.** Own `char_to_idx` / `idx_to_char`, built from the training split only, with an
  `<unk>` slot for unseen characters.
- **No prebuilt Transformer modules.** Attention (Q/K/V projections, scaled dot product,
  causal mask, softmax, head split/merge), LayerNorm placement, feed-forward, residuals, token and
  positional embeddings and the LM head are all hand-written. Neither code base uses
  `nn.Transformer*`, `nn.MultiheadAttention` or fused scaled-dot-product attention.
- **Causal mask verified behaviourally.** Both members changed the token at position j and
  confirmed that the logits at every position before j are bit-identical, while logits from j
  onward change.
- **Training.** Cross-entropy loss, AdamW (β = 0.9, 0.95, weight decay 0.1), linear warm-up
  followed by cosine decay, gradient clipping at 1.0, 10 epochs.
- **Generation.** Greedy decoding and temperature sampling from the prompt "Once upon a time".

### 1.2 Model comparison — architecture and hyperparameters

The two models are the team's depth-vs-width comparison: a deep-narrow model against a
shallow-wide one at the same embedding width and the same number of training tokens.

| | **shreya_akotiya** | **zoheb_waghu** |
|---|---|---|
| Design idea | deep-narrow | shallow-wide |
| Transformer blocks | **12** | **4** |
| Attention heads (dims/head) | 8 (32) | 4 (64) |
| Embedding width `d_model` | 256 | 256 |
| Feed-forward width | 1024 (4×) | 1024 (4×), GELU |
| Normalisation | pre-LN | pre-LN |
| Positional encoding | learned | learned |
| Context length | 256 | 256 |
| Vocabulary size | 78 | 97 |
| Dropout | 0.05 | 0.1 |
| **Parameters** | **9,570,816** | **3,274,752** |
| Batch size (tokens/step) | 64 (16,384) | 32 (8,192) |
| Steps (per epoch / total) | 1,562 / 15,620 | 3,125 / 31,250 |
| Peak → min learning rate | 2.5e-4 → 2.5e-5 | 3e-4 → 3e-5 |
| Warm-up | 781 steps (5%) | 1,000 steps (~3%) |
| Epochs | 10 | 10 |
| Training tokens seen | 256M | 256M |
| Seed | 1337 | 1337 |
| Hardware | NVIDIA Tesla T4 16 GB (Google Colab), CUDA, PyTorch 2.11 | NVIDIA GeForce RTX 4090 24 GB, AMD Ryzen 9 7950X, CUDA, PyTorch 2.5.1 |

The two models also differ in dropout, batch size, learning rate and head size, not only in
depth. The comparison is therefore **deep-narrow vs shallow-wide as complete designs**, not a
controlled ablation of depth alone.

### 1.3 Metrics — side by side

All numbers are from each member's `metrics_report.csv` (team schema) and
`metrics_report_extended.csv`. Cross-entropy is in nats per character on each member's own
10K-sequence validation split.

| Metric | **shreya_akotiya** | **zoheb_waghu** |
|---|---|---|
| Training cross-entropy (eval mode, best checkpoint) † | 0.6038 | 0.6982 |
| Validation cross-entropy (best) | **0.6327** | 0.7019 |
| Perplexity | **1.883** | 2.018 |
| Bits-per-character | **0.913** | 1.013 |
| Generalization gap (val − train) | 0.0289 | **0.0037** |
| Top-1 next-character accuracy | **79.9%** | 77.7% |
| Distinct-1 / 2 / 3 (character) ‡ | 0.0029 / 0.0286 / 0.1247 | 0.0125 / 0.0997 / 0.2832 |
| Repeated 4-gram rate (character) ‡ | 0.721 | 0.559 |
| Gradient norm mean / max | 0.687 / 17.79 | 0.744 / 9.77 |
| Loss spikes † | 22 | 0 |
| NaN / Inf events | 0 | 0 |
| Parameter count | 9,570,816 | 3,274,752 |
| Training throughput (tokens/sec) ¶ | 24,209 (T4) | 663,173 (RTX 4090) |
| Generation throughput (tokens/sec) ¶ | 87.6 | 507.6 |
| Peak GPU memory (`torch.cuda.max_memory_allocated`) | 8.06 GB | 0.99 GB |
| Total training time ¶ | 176.2 min | 6.4 min |
| Best epoch | 10 (last) | 10 (last) |

**Comparability notes**

- † **Training cross-entropy is not the number the loss curves end on.** Both members' values
  here are measured in **eval mode** (dropout off) on the best checkpoint, which is what
  `metrics_report.csv` stores. The per-epoch training curve in §1.4 is a running mean of
  minibatch losses *during* the epoch with dropout **active** and weights still changing, so it
  ends higher - Zoheb's curve ends at 0.749 against 0.698 here. Measured on his checkpoint over
  the same 300 training batches, changing only `model.train()` vs `model.eval()`, gives 0.733 vs
  0.684: dropout accounts for 0.0496 of the 0.0511 difference and within-epoch improvement for
  the remaining 0.0015. Validation agrees exactly between curve and table (0.7019), which rules
  out a plotting error. The generalization gap row compares eval mode with eval mode.
- ‡ **Diversity metrics are not directly comparable.** Both use the same formula (unique
  n-grams ÷ total n-grams over characters), but Shreya's are measured on 20,000 generated
  characters at T = 1.0 and Zoheb's on ~3,600 characters pooled over greedy, T = 0.8 and T = 1.2.
  Distinct-n falls and repeated-n-gram rate rises as the text gets longer, so the gap between the
  columns mostly reflects sample length, not model quality.
- † **Loss spikes use different detectors.** Shreya flags a step whose loss exceeds the rolling
  mean by 3σ (100-step window); her 22 spikes are each about 7% above the local mean. Zoheb flags
  a loss above 2× the running median, a much looser threshold that none of Shreya's spikes would
  meet. Both runs are stable by either definition: no NaNs, and validation loss improved every
  epoch.
- ¶ **Speed is not comparable.** Shreya trained on a Tesla T4 and Zoheb on an RTX 4090, a much
  faster GPU, so throughput and training time mostly reflect the hardware, not the models.
- **Peak memory is comparable.** Both runs measure it with `torch.cuda.max_memory_allocated`.
  The 8× difference comes from the deeper model (12 vs 4 blocks of stored activations) and the
  larger batch (64 vs 32).
- The two validation sets are different slices of TinyStories, so the loss comparison is close
  but not paired.

### 1.4 Training curves

**shreya_akotiya (12 × 256):**

![](../task1_llm/shreya_akotiya/outputs/plots/loss_curves.png)

**zoheb_waghu (4 × 256):**

![](../task1_llm/zoheb_waghu/outputs/plots/loss_curves_t1_baseline_20260929-213500.png)

Validation loss per epoch:

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| shreya — val bpc | 1.431 | 1.153 | 1.066 | 1.014 | 0.982 | 0.956 | 0.939 | 0.924 | 0.917 | **0.913** |
| zoheb — val bpc | 1.470 | 1.267 | 1.171 | 1.119 | 1.083 | 1.060 | 1.042 | 1.030 | 1.019 | **1.013** |

Both curves fall at every epoch and both models reach their best validation loss at the last
epoch. The deeper model is ahead from epoch 1, and the gap between the two models grows from
0.04 bpc at epoch 1 to about 0.10 bpc by epoch 10.

### 1.5 Joint analysis

**What we noticed from the comparison**

1. **The deeper model performed better on next-character prediction.** The 12-layer model reached
   0.913 bits per character and 79.9% next-character accuracy, compared with 1.013 bits per
   character and 77.7% accuracy for the 4-layer model. The difference makes sense for this
   task because a character model has to build characters into words, words into phrases, and
   phrases into sentences. The extra layers give it more steps to do that.
2. **The deeper model also costs more.** It has about 2.9 times as many parameters and used about
   eight times more peak GPU memory. The two training times cannot be compared directly,
   because Shreya's run used a Tesla T4 while Zoheb's used an RTX 4090.
3. **The smaller model had a smaller generalization gap, but that does not mean it learned more.**
   Its gap was 0.0037 compared with 0.0289 for the deeper model. The smaller gap mostly reflects the lower
   capacity of the shallow model: it fit the training data less closely as well as the validation
   data.
4. **Neither model had fully converged after 10 epochs.** Validation loss was still improving at
   the end of both runs, so the final epoch was still the best checkpoint. Ten epochs was the
   training limit, not the point where either model had completely finished learning.
5. **Both runs were stable.** Neither model produced NaN or Inf values. The largest gradient norm,
   17.8, occurred at the beginning of the deeper model's training. After warm-up, its maximum was
   2.9, and no step after step 1,910 needed clipping.

**Strengths**

- Both implementations pass a behavioural causal-mask test, not only a visual one.
- Both produce fluent, correctly spelled English at low temperature, with correct word
  boundaries, punctuation, quotation marks and the typical TinyStories structure.
- Disjoint data slices give each member a genuinely separate train/validation split.

**Weaknesses and shared failure modes**

The two models fail in the same ways, which suggests the failures come from the shared setup
(character-level next-token prediction, 256-character context, small model, no decoding
constraints) rather than from either design.

| Failure type | shreya_akotiya example | zoheb_waghu example |
|---|---|---|
| Repetition under greedy decoding | repeats "Once upon a time, there was a little girl named Lily." | "The box was so happy and the box was so happy to see the box." |
| Semantic contradiction / impossible meaning | "Lily was happy to hear that her mom was sad." | "Her pocket seemed sad to melt." |
| Losing track of who is who | "Lily didn't want to share his toy friends" | restarts a new story mid-generation ("Once upon a time…") |
| Non-words at high temperature | "strets", "pumpins", "knowled" (3.0% non-words at T = 1.2) | "aboven", "MPleaf", "spreadying" at T = 1.2 |

In both models, temperature moves the problem from one type of failure to another. Greedy and
low-temperature decoding usually give cleaner spelling, but the model can repeat common phrases.
Higher temperatures reduce repetition, but they also increase spelling and meaning errors. In Shreya's
temperature sweep, T = 0.8 gave the best balance: the non-word rate was only 0.42% and no repeated
phrase was detected. Zoheb saw the same general pattern: his repeated 4-gram rate decreased as
the temperature increased, but the semantic problems became more noticeable at T = 1.2.

**Limitations**

- Single seed per model, so there is no variance estimate.
- The two models differ in several hyperparameters at once (see §1.2), so the quality gap cannot
  be attributed to depth alone.
- The diversity and loss-spike metrics were computed with different sample sizes and detectors
  (see §1.3).
- The 256-character context is shorter than most stories, so neither model can keep a story
  consistent from start to finish.
- Generation uses no KV cache, so throughput figures understate what the models could do.

**What the team would try next**

1. **Train longer.** Both curves were still falling. We expect 15–20 epochs to lower validation
   loss further for both, with the deep model's gap growing first.
2. **Controlled depth ablation.** Train 256 × 4 and 256 × 12 with the same batch size, learning
   rate, dropout and seed, to isolate the effect of depth.
3. **Top-k / nucleus sampling (p ≈ 0.9).** Cutting the low-probability tail should keep the
   diversity of higher temperatures while avoiding their non-words, and should break greedy
   loops.
4. **Longer context (512–1024)** and masking at story boundaries, targeting the coherence and
   story-restart failures that decoding changes cannot fix.
5. **Unified evaluation script.** Re-score both models' diversity on the same number of
   characters at the same temperature, with one loss-spike definition.

### 1.6 Individual failure analyses

The generated-text snippets for every case are reproduced in **Appendix A**. Full write-ups are in
each member's folder:

- `task1_llm/shreya_akotiya/failure_analysis.md`:
    1. semantic contradiction (greedy);
    2. pronoun and reference confusion (T = 0.5);
    3. non-words at high temperature (T = 1.0);
    4. the temperature trade-off across greedy to T = 1.2.
- `task1_llm/zoheb_waghu/failure_analysis.md`:
    1. phrase-level repetition loop (greedy);
    2. invalid words and semantic breakdown at high temperature (T = 1.2);
    3. story-boundary confusion: restarting a new story mid-generation (greedy).

### 1.7 Evidence

| | shreya_akotiya | zoheb_waghu |
|---|---|---|
| Run ID | `task1_shreya_deep_narrow_20260920_194002` | `t1_baseline_20260929-213500` |
| Config | `task1_llm/shreya_akotiya/config.yaml` | `task1_llm/zoheb_waghu/configs/gpt_baseline.yaml` |
| Checkpoint | `task1_llm/shreya_akotiya/checkpoints/task1_shreya_deep_narrow_20260920_194002_best.pt` (sha256 `f540786b…`) | `task1_llm/zoheb_waghu/checkpoints/t1_baseline_20260929-213500_best.pt` (sha256 `6b7ecfe1…`) |
| Raw log | `reproducibility/raw_logs/shreya_akotiya/task1_llm/task1_shreya_deep_narrow_20260920_194002.log` | `reproducibility/raw_logs/zoheb_waghu/task1_llm/t1_baseline_20260929-213500.log` (+ `.jsonl`) |
| Manifest | `reproducibility/manifests/shreya_akotiya/task1_shreya_deep_narrow_20260920_194002.json` | `reproducibility/manifests/zoheb_waghu/task1_llm_manifest.md` |
| Loss curves | `outputs/plots/loss_curves.png` | `outputs/plots/loss_curves_t1_baseline_20260929-213500.png` |
| Samples | `outputs/samples/{greedy,T0.5,T0.8,T1.0,T1.2}.txt` | `outputs/samples/samples_t1_baseline_20260929-213500.txt` |
| Causal-mask check | `outputs/plots/causal_mask_check.png` | `src/evaluate.py::causal_mask_probe` (logged in run) |
| Metrics | `metrics_report.csv`, `metrics_report_extended.csv` | `metrics_report.csv`, `metrics_report_extended.csv` |

---

## Task 2 — Yelp Polarity sentiment classification

### 2.1 Shared setup

- **Dataset: Yelp Polarity** (confirmed), with 560,000 training and 38,000 test reviews,
  binary and exactly balanced.
- **Same test set.** Both members evaluate on the **full official 38,000-review test split**,
  touched once, at the end. Because the test set is identical, the test metrics below are
  directly comparable across members, unlike Task 1, where each member had their own
  validation slice.
- **Validation is held out from the training split** in both pipelines. Each member's three
  models share one split and one vocabulary, so the paired McNemar tests within a member are
  valid.
- **No pretrained embeddings or language models.** Every model uses an `nn.Embedding`
  initialised randomly and trained end to end.
- **Negation words are kept.** Both members remove stopwords but exempt negation words (`not`,
  `no`, `never`, `n't` forms), because removing them would turn "not good" into "good".
- **Same metric set.** Both pipelines report accuracy, precision/recall/F1 (macro, micro and
  weighted), confusion matrix, ROC-AUC, PR-AUC, MCC, Brier score, ECE, 95% bootstrap CIs, paired
  McNemar against the member's baseline, per-slice macro-F1 and error rate, parameter count,
  training time, examples/sec and peak GPU memory.

### 2.2 Preprocessing — side by side

| | **shreya_akotiya** | **zoheb_waghu** |
|---|---|---|
| Training data used | **all 540,000** (+ 20,000 validation) | **all 539,947** (+ 20,000 validation) |
| Test | full 38,000 | full 38,000 |
| Malformed rows dropped | 0 train / 0 test (none found; 35 reviews empty after cleaning, kept as padding) | 53 train / 0 test |
| Lowercase, strip punctuation/special characters | yes | yes |
| Contractions | apostrophe removed (`don't` → `dont`), kept as a negation word | expanded before stopword removal (`don't` → `do not`) |
| Stopword list | sklearn (318); 13 negation words exempted → 305 removed | NLTK (198); 39 negation words exempted → 158 removed |
| Word normalisation | light suffix stemming (`-s`, `-ed`, `-ing`, `-ly`, …) | WordNet lemmatisation |
| Vocabulary | 48,436 words (min frequency 5, cap 50K), 99.47% token coverage | 30,000 words (min frequency 2, cap 30K), 1.15% test OOV |
| Max length (tokens) | 200: keeps 96.6% of reviews whole | 256: keeps 97.9% of reviews whole |
| Embedding dimension | 200 (all three models) | 128 (M1, M2) / 256 (M3) |

EDA findings both members recorded: the data is perfectly balanced, and review length is
right-skewed (a long tail of reviews over 500 words). Shreya also found that **negative reviews
are longer** (median 112 vs 84 words).

### 2.3 Models — architecture and hyperparameters

Each member trained one baseline and two experimental models. No two of the six share an
architecture and hyperparameter set.

| | Model | Encoder | Pooling | Params | Dropout | Optimiser / LR | Batch / epochs |
|---|---|---|---|---|---|---|---|
| **Shreya** | **Baseline** — mean-pool | none (bag of embeddings) | masked mean | 9,687,602 | 0.3 | Adam 1e-3 | 256 / 3 |
| | Exp. 1 — TextCNN | Conv1d, widths 3/4/5 × 100 filters | max over time | 9,928,102 | 0.5 | Adam 1e-3 | 256 / 3 |
| | Exp. 2 — BiLSTM | 1-layer BiLSTM, hidden 128 | masked max over time | 10,025,634 | 0.3 | Adam 1e-3 | 256 / 3 |
| **Zoheb** | **Baseline** — BiLSTM-mean | 1-layer BiLSTM, hidden 128 | masked mean | 4,104,449 | 0.3 | Adam 1e-3 | 64 / 8 (early stop) |
| | Exp. 1 — TextCNN | Conv1d, widths 2/3/4/5 × 128 filters | max over time | 4,135,681 | 0.5 | Adam 1e-3, wd 1e-5 | 64 / 8 (early stop) |
| | Exp. 2 — BiLSTM-attention | 2-layer BiLSTM, hidden 256 | additive (Bahdanau) attention | 10,441,217 | 0.4 | AdamW 2e-3, cosine, 300-step warm-up | 32 / 12 (early stop) |

**Design logic.**

- **Shreya's set adds sequence information step by step at a fixed embedding size:** no word
  order (mean-pool), then local n-grams (TextCNN), then full-sequence state (BiLSTM).
- **Zoheb's set starts from a sequential baseline** and tests two changes: replacing recurrence
  with convolutions (TextCNN), and replacing mean pooling with learned attention (BiLSTM-attn).
- **Model selection.** Both members keep the checkpoint with the best validation score. Every
  model peaked early: Shreya's at epochs 2–3 of 3, and Zoheb's early stopping chose epochs 3, 5
  and 3 (BiLSTM-mean, TextCNN, BiLSTM-attn).

### 2.4 Metrics — all six models on the same 38K test set

| Metric | S: mean-pool | S: TextCNN | S: BiLSTM | Z: BiLSTM-mean | Z: TextCNN | Z: BiLSTM-attn |
|---|---|---|---|---|---|---|
| Accuracy ◆ | 0.9314 | 0.9435 | 0.9485 | 0.9535 | 0.9502 | **0.9553** |
| Precision (macro) ◆ | 0.9315 | 0.9435 | 0.9485 | 0.9535 | 0.9502 | **0.9553** |
| Recall (macro) ◆ | 0.9314 | 0.9435 | 0.9485 | 0.9535 | 0.9502 | **0.9553** |
| F1 (macro) ◆ | 0.9314 | 0.9435 | 0.9485 | 0.9535 | 0.9502 | **0.9553** |
| F1 (micro) ◆ | 0.9314 | 0.9435 | 0.9485 | 0.9535 | 0.9502 | **0.9553** |
| F1 (weighted) ◆ | 0.9314 | 0.9435 | 0.9485 | 0.9535 | 0.9502 | **0.9553** |
| ROC-AUC ◆ | 0.9793 | 0.9863 | 0.9891 | 0.9910 | 0.9896 | **0.9916** |
| PR-AUC ◆ | 0.9787 | 0.9866 | 0.9894 | 0.9912 | 0.9899 | **0.9917** |
| MCC ◆ | 0.8629 | 0.8870 | 0.8970 | 0.9070 | 0.9004 | **0.9106** |
| Brier score ◆ | 0.0516 | 0.0430 | 0.0384 | 0.0350 | 0.0371 | **0.0335** |
| ECE ‡ | 0.0046 | 0.0111 | 0.0086 | 0.0100 | **0.0044** | 0.0053 |
| Confusion TN / FP | 17,795 / 1,205 | 17,850 / 1,150 | 18,091 / 909 | 18,042 / 958 | 18,009 / 991 | 18,064 / 936 |
| Confusion FN / TP | 1,401 / 17,599 | 997 / 18,003 | 1,048 / 17,952 | 809 / 18,191 | 901 / 18,099 | 763 / 18,237 |
| Accuracy 95% CI | [0.9289, 0.9339] | [0.9412, 0.9458] | [0.9463, 0.9507] | [0.9513, 0.9556] | [0.9480, 0.9524] | [0.9531, 0.9573] |
| Macro-F1 95% CI | [0.9289, 0.9339] | [0.9412, 0.9458] | [0.9463, 0.9507] | [0.9513, 0.9556] | [0.9480, 0.9524] | [0.9531, 0.9573] |
| MCC 95% CI | [0.8578, 0.8679] | [0.8824, 0.8915] | [0.8926, 0.9014] | [0.9027, 0.9111] | [0.8960, 0.9048] | [0.9063, 0.9146] |
| McNemar p vs own baseline | – | **1.6e-24** | **9.4e-55** | – | **1.5e-3** (worse) | **0.034** |
| Parameters | 9.69M | 9.93M | 10.03M | 4.10M | 4.14M | 10.44M |
| Training time ¶ | 41 s | 248 s | 350 s | 227 s | 261 s | 3,684 s |
| Examples/sec ¶ | 39,502 | 6,547 | 4,637 | 11,904 | 14,508 | 879 |
| Peak GPU memory | 338 MB | 460 MB | 2,286 MB | 744 MB | 220 MB | 1,529 MB |
| Hardware | Tesla T4 | Tesla T4 | Tesla T4 | A100 | A100 | A100 |

Micro-F1 equals accuracy for single-label classification. Macro and weighted F1 are almost
identical to it because the test set is exactly balanced.

**Comparability notes**

- ◆ **Same training data, different preprocessing.** Both members now train on all ~540,000 Yelp
  training reviews, so training-set size no longer differs. The remaining cross-member
  differences are the preprocessing and input setup: vocabulary (48,436 vs 30,000 words), maximum
  length (200 vs 256 tokens), embedding size, and stemming vs lemmatisation with contraction
  expansion (see §2.2). Within each member's column the three models share one setup and are
  directly comparable; across columns, a gap reflects preprocessing and architecture together.
  A paired McNemar test between the two members' best models would add a paired comparison —
  both scored the same official 38K test set — but it needs per-example predictions from both
  members, which are currently saved for Zoheb only.
- ¶ **Speed is not comparable across members.** Shreya trained on a Tesla T4 and Zoheb on an
  A100 (both Google Colab), and the epoch counts differ. Peak memory is comparable: both use
  `torch.cuda.max_memory_allocated`.
- ‡ **ECE uses different bin counts** (10 bins for Shreya, 15 for Zoheb), so small ECE
  differences across members are not meaningful. Brier score has no such parameter and is
  directly comparable.
- Bootstrap CIs use 1,000 resamples (Shreya) and 2,000 (Zoheb). This affects the CI estimates
  only slightly.

**Confusion matrices (Shreya):**

![](../task2_sentiment/shreya_akotiya/outputs/plots/confusion_matrices.png)

**Confusion matrices and calibration (Zoheb):**

![](../task2_sentiment/zoheb_waghu/outputs/confusion_matrices/confusion_and_calibration.png)

**ROC, PR and calibration curves (Shreya):**

![](../task2_sentiment/shreya_akotiya/outputs/plots/curves.png)

**Model comparison (Zoheb):**

![](../task2_sentiment/zoheb_waghu/outputs/plots/model_comparison.png)

### 2.5 Per-slice robustness

Each member defined their own slices, so the numbers within a member's set are comparable, and
across members only the trends are.

**Shreya** (lengths in processed tokens; negation detected after preprocessing). Each cell is
macro-F1 / error rate.

| Slice (n) | mean-pool | TextCNN | BiLSTM |
|---|---|---|---|
| ≤ 40 tokens (16,731) | 0.931 / 6.8% | 0.943 / 5.6% | 0.949 / 5.0% |
| 41–100 tokens (14,542) | 0.932 / 6.8% | 0.945 / 5.4% | 0.949 / 5.1% |
| > 100 tokens (6,727) | 0.924 / 7.2% | 0.934 / 6.3% | 0.941 / 5.6% |
| has negation (28,544) | 0.924 / 7.4% | 0.940 / 5.9% | 0.945 / 5.3% |
| no negation (9,456) | 0.927 / 5.3% | 0.930 / 5.0% | 0.936 / 4.6% |

**Zoheb** (lengths in tokens; negation detected in the raw review). Each cell is macro-F1 /
error rate.

| Slice (n) | BiLSTM-mean | TextCNN | BiLSTM-attn |
|---|---|---|---|
| short, ≤ 50 tokens (18,860) | 0.953 / 4.6% | 0.951 / 4.8% | 0.954 / 4.5% |
| long, > 200 tokens (1,697) | 0.941 / 5.2% | 0.928 / 6.3% | 0.940 / 5.3% |
| contains negation (22,583) | 0.951 / 4.6% | 0.948 / 4.9% | 0.952 / 4.5% |
| ≥ 3 exclamation marks (6,944) | 0.968 / 3.1% | 0.963 / 3.6% | 0.970 / 2.9% |

**Shared findings.**

- **Long reviews are the weakest slice for all six models.** Long reviews are more often mixed
  (praise and complaint in one review), and a few lose their tail to truncation. Shreya's BiLSTM
  is her most accurate model on long reviews; in Zoheb's set the BiLSTM-mean and BiLSTM-attn are
  level on them (0.941 vs 0.940). In both sets the TextCNN drops most (Shreya: +0.7–0.9 error
  points; Zoheb: 0.951 → 0.928 macro-F1), consistent with fixed-width filters losing long-range
  contrast.
- **Reviews with a negation are harder for every model**, but keeping negation words limits the
  damage. The penalty is largest for the model with no word order: Shreya's mean-pool error
  rate rises 2.1 points from the no-negation to the negation slice, against 0.7–0.8 points for
  her TextCNN and BiLSTM. Zoheb's models lose about 0.3 macro-F1 points on the negation slice
  relative to their overall score.
- **Strongly emotional reviews (≥ 3 exclamation marks) are the easiest slice** in Zoheb's
  analysis (error rate 2.9–3.6%), because they carry unambiguous sentiment.

### 2.6 Joint analysis

**What we learned from the comparison**

1. **Zoheb's BiLSTM-attention had the highest test accuracy at 95.53%.** Its confidence interval
   [0.9531, 0.9573] does not overlap Shreya's best model, the BiLSTM at 94.85% [0.9463, 0.9507].
   Zoheb's BiLSTM-mean baseline (95.35%) is also above it. Both members now train on the same
   540K reviews, so the gap comes from preprocessing and model setup (lemmatisation and contraction
   expansion, a 30K vocabulary, 256-token inputs) rather than training-set size, but these factors
   cannot be separated from each other with the current runs.
2. **More training data clearly helped.** Zoheb retrained the same three models on all 540K
   reviews instead of a 90K subsample, with the same model settings. Accuracy rose by about 1.5–1.9
   points for every model (93.46% → 95.35%, 93.57% → 95.02%, 94.02% → 95.53%). This is the most
   controlled comparison in Task 2, because the models and their hyperparameters were unchanged.
3. **Keeping word order helped.** Shreya's mean-pooling baseline was the weakest of all six models
   at 93.14%, while both her TextCNN and BiLSTM performed better. Zoheb's baseline was already a
   BiLSTM, which reads word order.
4. **TextCNN helps against a bag-of-words model, but not against a BiLSTM.** Shreya's TextCNN
   improved over her mean-pooling baseline by 1.2 points (p = 1.6e-24). Zoheb's TextCNN was
   significantly *worse* than his BiLSTM baseline by 0.3 points (p = 0.0015). Local phrase
   features add a lot when the baseline ignores word order, but a recurrent encoder already
   captures them and more.
5. **Attention pooling gave a small but real gain.** Zoheb's attention BiLSTM beat his mean-pooling
   BiLSTM by 0.18 points (p = 0.034), at about 16 times the training time (3,684 s vs 227 s).
6. **The probability outputs were generally useful.** All models had reasonably low Brier scores
   and ECE values. Within each member's experiments, the model with the best accuracy also had the
   best Brier score.
7. **Most models reached their best validation result early.** The models usually peaked within
   the first few epochs. This suggests that future improvements should focus more on the amount of
   data, input length, and model design than simply adding many more epochs.

**Strengths**

- All six models exceed 93% accuracy with embeddings learned entirely from scratch.
- Every claim is backed by a significance test (CIs plus paired McNemar), not by point
  estimates alone.
- The negation handling is measured: both members report a negation slice, and the
  bag-of-words model's larger negation penalty shows why negations had to be kept.

**Weaknesses and limitations**

- **Cross-member differences have more than one cause.** Training-set size is now the same, but
  vocabulary, maximum length, embedding size, preprocessing (stemming vs lemmatisation) and
  hardware still differ between the two sets, so a cross-member gap cannot be attributed to any
  single factor.
- **Single seed per model**, so there is no estimate of run-to-run variance.
- **Slices were defined independently** (different length cut-offs and negation detectors) and
  ECE bins differ, so those numbers are not fully comparable across members.
- **Label noise.** Both error reviews found reviews whose text clearly contradicts their gold
  label (rating-text mismatch), which caps achievable accuracy.
- **Truncation** at 200 / 256 tokens cuts the end of the ~2–3% longest reviews, where the
  final verdict often sits.

**What the team would try next**

1. **A controlled preprocessing test.** Train Shreya's BiLSTM with Zoheb's preprocessing (and the
   reverse) to measure how much of the gap between the two best models comes from preprocessing.
2. **A paired McNemar test between the two best models.** Both members scored the same official
   38K test reviews in the same order, so the test is valid. `task2_sentiment/cross_member_mcnemar.py`
   performs it; it needs Shreya's best model's per-example test probabilities exported once
   (Zoheb's are already committed as `test_probs_t2_m3_bilstm_attn.npy`). It would answer "is one
   setup better on this test set", not "is one architecture better".
3. **A hierarchical or sentence-level encoder** for long, mixed reviews. Both error reviews
   identify mixed sentiment as a leading error type, and the long-review slice is the weakest
   for all six models.
4. **Shared evaluation code.** Use the same slice definitions, ECE bins and bootstrap count, so
   every robustness number is comparable.
5. **Multiple seeds** (e.g. 3 per model) to put variance bars on the close calls, such as
   Zoheb's TextCNN vs his baseline.

### 2.7 Individual error reviews

Each member manually reviewed 20 errors from one of their own models: 5 confident false
positives, 5 confident false negatives, 5 near-threshold errors and 5 slice-specific errors.
Each error has a type and a testable fix. All 40 reviews, with the review text, are in
**Appendix B**.

| | **shreya_akotiya** | **zoheb_waghu** |
|---|---|---|
| Model reviewed | BiLSTM (best, 94.85%) | BiLSTM-mean baseline (93.46%) |
| Slice used for the last 5 | reviews containing a negation | long reviews (> 200 tokens), the worst slice |
| Most common error types | mixed sentiment / split verdicts, temporal shifts ("EDIT: now much better"), negation scope (negation of a competitor, a requirement or a past state), sarcasm | **mixed sentiment (12 of 20)**, rating-text mismatch (2), domain term (2), length truncation (2), negation (1), sarcasm (1) |
| Proposed fix | phrase- or aspect-level sentiment aggregation; parse negation scope; recency weighting | chunked hierarchical sentence encoder over up to 512 tokens, measured on long-review macro-F1 |
| Full write-up | `task2_sentiment/shreya_akotiya/failure_analysis.md` | `task2_sentiment/zoheb_waghu/failure_analysis.md`, `outputs/error_review_t2_m1_baseline.md` |

Both reviews reach the same conclusion from different models. The remaining errors are mostly
**discourse-level**: mixed or changing sentiment within one review, sarcasm, and negations whose
scope is not the review's subject. A single pooled vector cannot represent these. Both members
also found label noise, which no model can fix.

### 2.8 Evidence

| | shreya_akotiya | zoheb_waghu |
|---|---|---|
| Run IDs | `task2_shreya_main_20260921_222011` (all three models) | `t2_m1_baseline_20261006-210947`, `t2_m2_cnn_20261006-211829`, `t2_m3_bilstm_attn_20261006-212317` |
| Config | `task2_sentiment/shreya_akotiya/config.yaml` | `task2_sentiment/zoheb_waghu/configs/` (`_shared.yaml` + one per model) |
| Checkpoints | `task2_sentiment/shreya_akotiya/checkpoints/{baseline_meanpool,exp_textcnn,exp_bilstm}.pt` | `task2_sentiment/zoheb_waghu/checkpoints/t2_m{1,2,3}_…_best.pt` |
| Raw logs | `reproducibility/raw_logs/shreya_akotiya/task2_sentiment/task2_shreya_main_20260921_222011.log` | `reproducibility/raw_logs/zoheb_waghu/task2_sentiment/t2_m*_20261006-*.log` (+ `.jsonl`) |
| Manifest | `reproducibility/manifests/shreya_akotiya/task2_shreya_main_20260921_222011.json` | `reproducibility/manifests/zoheb_waghu/task2_sentiment_manifest.md` |
| Metrics | `metrics_report.csv`, `metrics_report_extended.csv`, `outputs/slice_metrics.csv`, `outputs/mcnemar.csv` | `metrics_report.csv`, `metrics_report_extended.csv` |
| Plots | `outputs/plots/{eda,confusion_matrices,curves}.png` | `outputs/plots/{eda_overview,model_comparison}.png`, `outputs/confusion_matrices/confusion_and_calibration.png` |
| Error review | `failure_analysis.md`, `outputs/error_review.md` | `failure_analysis.md`, `outputs/error_review_t2_m1_baseline.md` |

---

## Task 3 — CycleGAN image style transfer (Monet ↔ photo)

### 3.1 Shared setup

- **Data.** Kaggle `gan-getting-started`: 300 Monet paintings and 7,038 photographs, unpaired,
  256 × 256. Both members trained at full 256px resolution, with the same augmentation: resize to
  286, random 256 crop, horizontal flip.
- **Direction names.** This report uses the instructor's convention: **A = Monet, B = photo**.
  So **A2B = Monet → photo** and **B2A = photo → Monet** (the Kaggle direction). Shreya's
  notebook internally uses A = photo, so her `outputs/pred_A2B/` folder holds photo → Monet. All
  tables below use the instructor's names.
- **Same CycleGAN recipe.** Both models have two generators and two discriminators, trained with
  an LSGAN adversarial loss, cycle-consistency loss (λ = 10 in both directions), identity loss
  (0.5 × λ = 5), a 50-image fake pool for each discriminator, Adam (lr 2e-4, β = (0.5, 0.999)),
  batch size 1 with instance normalisation, and one epoch = one pass over the 7,038 photos (each
  Monet image is reused about 23 times per epoch).
- **Same evaluation.** All image-quality numbers follow the instructor's
  `Part3_Evaluation_Script.ipynb`: the first 300 sorted images per set, Inception-v3 features,
  FID and MiFID per direction, and submission value = mean of the two directions. Both members
  then computed KID, precision/recall, density/coverage, cycle L1, LPIPS and content cosine on
  the same 300 images, so these numbers are directly comparable.
- **Leaderboard integrity.** Every submitted value comes from each member's own CycleGAN
  inference. Pretrained networks (Inception-v3, AlexNet for LPIPS) are used only to *measure*
  images, never to generate or edit them.

### 3.2 Model comparison — architecture and hyperparameters

The two models differ mainly in the **generator**: Shreya used a UNet with skip connections from
pix2pix, and Zoheb used the ResNet-9 generator from the CycleGAN paper.

| | **shreya_akotiya** | **zoheb_waghu** |
|---|---|---|
| Generator | **UNet**: 7 stride-2 encoder blocks down to a 2 × 2 bottleneck, 6 decoder blocks + output layer, skip connection at every level, dropout 0.5 in 3 decoder blocks | **ResNet-9**: 7 × 7 conv, 2 downsamples, **9 residual blocks** at 64 × 64, 2 upsamples, reflection padding |
| Generator parameters | 41,821,187 each | 11,378,179 each |
| Discriminator | PatchGAN: 3 stride-2 convs (64 → 256) + output conv; 31 × 31 score map | 70 × 70 PatchGAN: 3 stride-2 convs + a stride-1 512 conv; score map |
| Discriminator parameters | 662,593 each | 2,764,737 each |
| **Total parameters** | **84,967,560** | **28,285,832** |
| Epochs (steps) | **80** (563,040) | **40** (281,520) |
| LR schedule | constant to epoch 61, linear decay over epochs 62–80 (ends at 1.9e-5) | constant 20 epochs, linear decay to 0 over 20 |
| Seed | 42 | 1337 |
| Hardware | NVIDIA A100-SXM4-40GB (Google Colab), PyTorch 2.11 | NVIDIA RTX 4090 24 GB, PyTorch 2.5.1 |
| Training time | 9.66 h | 6.69 h |

**Design logic.**

- **Shreya (UNet):** skip connections pass edges and layout straight from the encoder to the
  decoder, so the generator only has to change colour and texture rather than rebuild the scene.
  The expected strength is content preservation; the expected risk is that the skip paths make it
  easy to copy the input or hide information in it.
- **Zoheb (ResNet-9):** nine residual blocks repeatedly transform texture and colour at 64 × 64.
  This is the generator from the original CycleGAN paper, with a larger 70 × 70 patch
  discriminator focused on local brush texture.
- Like Task 1, this compares **two complete designs**. Generator, discriminator, epoch count,
  schedule, seed and GPU all differ, so a difference cannot be credited to the generator alone.

### 3.3 Metrics — side by side (same 300 images per direction)

| Metric | Shreya A2B (Monet → photo) | Shreya B2A (photo → Monet) | Zoheb A2B (Monet → photo) | Zoheb B2A (photo → Monet) |
|---|---:|---:|---:|---:|
| FID ↓ | 102.784 | **97.904** | 103.197 | 98.855 |
| MiFID | 0.4206 | 0.4043 | 0.4181 | 0.4047 |
| KID ↓ | 0.0185 | **0.0069** | 0.0182 | 0.0076 |
| Precision (realism) | **0.730** | 0.540 | 0.703 | 0.507 |
| Recall (diversity) | 0.447 | **0.730** | 0.480 | 0.680 |
| Density | 1.005 | 0.489 | 1.013 | 0.433 |
| Coverage | 0.850 | 0.763 | 0.897 | 0.690 |
| Cycle-reconstruction L1 ↓ ([0, 1] pixels) | **0.0223** | **0.0216** | 0.0332 | 0.0398 |
| LPIPS, input vs translation (how much changed) | 0.244 | 0.304 | 0.354 | 0.378 |
| Content cosine, input vs translation ↑ | **0.861** | **0.821** | 0.797 | 0.772 |
| Human audit (style / content / artifacts) ※ | 3.80 / 4.23 / 4.23 | 4.00 / 3.83 / 4.20 | 3.17 / 4.57 / 4.63 | 3.37 / 4.47 / 4.83 |
| Inter-rater agreement (Cohen's κ) ※ | **0.500** (67.8% exact, 84.4% within 1) | 0.500 (same audit) | 0.149 (63.3% exact, **100%** within 1) | 0.149 (same audit) |

| Submission and training | **shreya_akotiya** | **zoheb_waghu** |
|---|---|---|
| Submission FID / MiFID (mean of both directions) | 100.344 / 0.4124 | 101.026 / 0.4114 |
| **Leaderboard score (FID + MiFID) / 2 ↓** | **50.38** | 50.72 |
| Kaggle score (as displayed) ‖ | **-50.3781** (our final score) | -50.7186 |
| Generator loss (final 10% of steps) | 2.439 | 2.970 |
| Discriminator loss, D_A + D_B (final 10%) | 0.171 | 0.251 |
| Cycle loss, unweighted (final 10%) | 0.080 | 0.132 |
| Identity loss, unweighted (final 10%) | 0.046 | 0.097 |
| Generator gradient norm (mean) † | 38.40 | 23.36 |
| NaN / Inf steps | 0 | 0 |
| Images / sec ¶ | 16.2 | 11.7 |
| Peak GPU memory † | 1.78 GB | 12.78 GB |

**Comparability notes**

- ※ **Human audit — protocol, and why the two κ values are not comparable.** 30 blinded samples
  per member (15 per direction, seed 42, drawn from the same first 300 predictions the
  instructor's evaluator scores), each sheet showing **source | translation** side by side so
  content preservation can be judged. Both members' sheets happen to cover the **same 30 source
  images**, so the per-axis means *are* directly comparable. Two raters, scores 1–5 with **5 best
  on every axis**, rated independently against a shared written rubric
  (`outputs/RATING_GUIDE.md`).
  **Both audits needed two rounds.** The first round of each was run before the rubric existed and
  produced κ = −0.05 (zoheb_waghu) and κ = −0.04 (shreya_akotiya) — agreement *worse than chance*,
  caused by the raters applying different definitions rather than seeing different things. After
  the rubric was written with explicit 1–5 anchors, the same sheets were re-rated, giving the
  figures above.
  **κ is not comparable between the two members, but the raw agreement is.** Same raters, same
  rubric, same source images, yet κ differs threefold. zoheb_waghu's raters agreed more *tightly*
  — never more than one point apart on any sample — but clustered their scores into two adjacent
  categories, which raises chance agreement to 50–61% and deflates κ to 0.149. shreya_akotiya's
  spread across three categories, lowering chance agreement and lifting κ to 0.500 despite a
  *lower* within-1 rate (84.4% vs 100%). κ measures agreement relative to the rating
  distribution, so the honest cross-member comparison is the exact and within-1 percentages, not κ.
- ‖ **Kaggle scoring.** The competition reports a single leaderboard computed on all the test
  data (there is no public/private split), and the score is `-(FID + MiFID) / 2`, so a less
  negative number is better. **Our team's final score is -50.38** (Shreya's model: FID 100.344,
  MiFID 0.4124). Zoheb's model's submission evaluates to -50.72 (FID 101.026, MiFID 0.4114).
- **FID across directions is not comparable.** A2B is compared with real photos and B2A with real
  Monet paintings, so the two directions have different reference sets. Compare each direction
  across the two members, not A2B against B2A.
- † **Gradient norm and peak memory were measured differently.** Zoheb's values come from his
  training run. Shreya's run did not log them, so hers come from **one extra
  diagnostic epoch** (7,038 steps) started from the epoch-80 checkpoint, with the same losses and
  data pipeline (`run1_grad_diagnostic.ipynb`). Her gradient norm therefore describes the trained
  model, not the average over training.
- ¶ **Speed is not comparable** (A100 vs RTX 4090), and the two models also have very different
  parameter counts.
- **Training losses** come from different architectures, so they are context, not a ranking.

### 3.4 Training curves

**shreya_akotiya (UNet, 80 epochs):**

![](../task3_gan/shreya_akotiya/outputs/plots/loss_curves.png)

**zoheb_waghu (ResNet-9, 40 epochs):**

![](../task3_gan/zoheb_waghu/outputs/plots/training_curves_t3_baseline_20260929-235720.png)

**Example translations (zoheb_waghu):** input, translation and cycle reconstruction.

![](../task3_gan/zoheb_waghu/outputs/plots/translation_examples.jpg)

Shreya's failure-case images are in `task3_gan/shreya_akotiya/outputs/plots/failure_candidates_{B2A,A2B}.jpg`
(source, translation, cycle reconstruction for 15 cases per direction).

**What the training shows.**

- **Both runs were numerically stable:** 0 NaN or Inf steps. Zoheb's largest gradient spike
  (527 at epoch 21) recovered by the next logged step without loss divergence.
- **Losses fell fastest early in both runs.** Shreya's cycle loss was flat (about 0.41 weighted)
  from epoch 30. Zoheb's checkpoint score improved mostly up to epoch 20 and then varied by only
  0.46 points to epoch 40.
- **The discriminators gradually got ahead in Shreya's run.** After epoch 5, the photo → Monet
  adversarial loss rose (0.40 → 0.76) while discriminator loss fell (0.23 → 0.08). In Zoheb's run,
  the discriminator loss stayed at 0.245–0.310 from epoch 11, which means the discriminators
  separated real from fake usefully without completely winning.

### 3.5 Joint analysis

**What the comparison shows**

1. **Image quality is effectively tied.** The two models are within about one FID point in each
   direction (B2A 97.9 vs 98.9; A2B 102.8 vs 103.2), and the leaderboard scores are 50.38 vs 50.72.
   That gap is well inside the noise of 300-image FID: Zoheb's FID varied over a range of 4.7–6.5
   points across his checkpoints after epoch 20, and Shreya's v4 run moved 5–8 points between checks 10 epochs
   apart. Neither model is better on FID or KID.
2. **The UNet preserves content more strongly.** Shreya's model has about half the cycle error
   (0.022 vs 0.033–0.040), smaller LPIPS change (0.24–0.30 vs 0.35–0.38) and higher content
   cosine (0.82–0.86 vs 0.77–0.80) in both directions. This is what skip connections are expected
   to do: they carry the scene structure straight to the output.
3. **But low cycle error is not proof of good translation.** Both members found translations
   that look badly broken but still reconstruct almost perfectly. Shreya's night photos become a
   tiled blue-grey pattern yet have the *lowest* cycle errors (0.013–0.021). Zoheb's purple-cloud
   photo becomes a bright paint field, yet the reconstruction restores it. The source information
   is carried in subtle signals a person cannot see ("steganography", Chu et al., 2017). UNet skip
   connections make this easier, so part of the UNet's cycle-error advantage comes from hidden
   information, not from faithful translation.
4. **Both models fail in the same direction-specific way.** **Photo → Monet is diverse but less
   convincing:** recall is higher than precision for both (Shreya 0.73 / 0.54; Zoheb 0.68 / 0.51),
   so the outputs cover much of the Monet style space but many don't look like a real Monet.
   **Monet → photo is convincing but narrow:** precision is higher than recall for both (0.73 /
   0.45; 0.70 / 0.48), so the outputs look photographic but cover only part of the variety of real
   photos. The shared cause is the data: only 300 Monet paintings, against 7,038 photos.
5. **More capacity and training did not buy a better score.** Shreya's model has 3× the
   parameters and trained twice as many epochs, for a 0.34-point lower leaderboard score, which
   is within noise. Zoheb's checkpoint sweep shows the score had nearly plateaued by epoch 20, and
   Shreya's three alternative runs (v2–v4: DiffAugment, lower identity weight, removed outer skip,
   EMA) all scored worse (54.2–55.9). The limiting factors appear to be the small Monet set and
   the noisy evaluation, not model size.

**Strengths**

- Both members built a complete CycleGAN, trained it at full 256px resolution, and produced a
  valid submission from their own model's inference.
- All quality metrics use the instructor's method on the same 300 images, so the side-by-side
  comparison is fair.
- Cycle consistency is verified with measured reconstruction distances, not only assumed from
  the loss term, and both members inspected the cases where low cycle error hides bad output.

**Weaknesses and shared failure modes**

| Failure type | shreya_akotiya example | zoheb_waghu example |
|---|---|---|
| Dark or low-detail scenes break down | night sky → blotchy blue-grey field with a tiled pattern (`09fc404e31`) | dark sunset → structureless pastel field (`02ded12bbd`) |
| Monet → photo becomes too dark / too contrasty | golden Parliament sunset → almost black (`b1ea5d5a7d`) | the same image → black and orange extremes (`b1ea5d5a7d`) |
| Too little change (still looks like the source domain) | office building stays photographic (`063ab57d41`) | the same building, mild purple tint only (`063ab57d41`) |
| The same coastal painting fails (`a619072f82`) | rainbow-like horizontal band across the sky, kept in the reconstruction | becomes saturated and high-contrast; the reconstruction is washed out |
| Colour shifts | sunset loses its red horizon (`02ded12bbd`) | orange clouds turn cyan, storm sky turns teal (`0845e8dc24`, `0254fc91bd`) |

Several of the same images fail for both models (`b1ea5d5a7d`, `a619072f82`, `063ab57d41`,
`02ded12bbd`, plus `6782e7cb2a` and `04f59976b5`). That suggests these images are hard for
CycleGAN in general, not for one architecture. **Specific to the UNet:** the tiled pattern on
night photos and horizontal sky streaks. **Specific to the ResNet-9:** strong hue shifts in
clouds and skies.

**Limitations**

- **Single seed per model.** Run-to-run variance is not measured, and the checkpoint-to-checkpoint
  swings above suggest it is several FID points.
- **300-image FID is noisy and biased upward.** It matches the leaderboard, but small differences
  between models are not meaningful.
- **Not a controlled comparison.** Generator, discriminator, epochs, schedule, seed and GPU all
  differ.
- **Domain imbalance.** With 300 Monet paintings each seen about 23 times per epoch, the Monet
  discriminator can memorise them.

**What the team would try next**

1. **Select checkpoints on a held-out validation set** (FID, KID, precision/recall) rather than
   taking the last epoch, since quality plateaus and fluctuates late in training.
2. **Run three seeds per setting** before treating any FID difference of a few points as real.
3. **Change one thing at a time.** For the UNet: lower only the identity weight (0.5) while keeping
   the outer skip. For the ResNet-9: a discriminator with a larger receptive field, to target low
   B2A precision.
4. **Rebalance the discriminators** (e.g. fewer discriminator updates) where they overpower the
   generator.
5. **Complete the human audit** to check whether people agree with the metric tie, particularly on
   artifacts.

### 3.6 Individual failure analyses

Failure-case images for both members are in **Appendix C**. Full write-ups with sample IDs, LPIPS
and cycle values are in each member's folder:

- `task3_gan/shreya_akotiya/failure_analysis.md`: 14 failure cases (8 photo → Monet, 6 Monet →
  photo), the cycle-consistency check including the steganography finding, training stability,
  shared hard images, and the human-audit protocol. `results.md` also covers the three alternative
  runs (v2–v4).
- `task3_gan/zoheb_waghu/failure_analysis.md`: 18 failure cases (9 per direction: least changed,
  most changed, worst cycle), cycle-consistency verification, training stability including the
  epoch-21 gradient spike, and the human-audit protocol.

### 3.7 Evidence

| | shreya_akotiya | zoheb_waghu |
|---|---|---|
| Run ID | `t3_shreya_unet_128_20261002_023102` (256px; "128" is from an earlier plan) | `t3_baseline_20260929-235720` |
| Config | `task3_gan/shreya_akotiya/config.yaml` | `task3_gan/zoheb_waghu/configs/cyclegan_baseline.yaml` |
| Checkpoint | `checkpoints/t3_shreya_unet_128_20261002_023102_G_AB_fp16.pt`, `…_G_BA_fp16.pt` | `checkpoints/t3_baseline_20260929-235720_generators_fp16.pt` |
| Raw logs | `reproducibility/raw_logs/shreya_akotiya/task3_gan/t3_shreya_unet_128_20261002_023102.log` (+ `_graddiag.{csv,json}`, v2–v4 logs) | `reproducibility/raw_logs/zoheb_waghu/task3_gan/t3_baseline_20260929-235720.{log,jsonl}` |
| Manifest | `reproducibility/manifests/shreya_akotiya/task3_gan_manifest.md`, `t3_shreya_unet_128_20261002_023102.json` | `reproducibility/manifests/zoheb_waghu/task3_gan_manifest.md` |
| Metrics | `metrics_report.csv`, `submission.csv` | `metrics_report.csv`, `full_metrics_report.csv`, `submission.csv`, `outputs/snapshot_metrics.csv` |
| Evaluated images | `outputs/pred_A2B/` (photo → Monet, 300), `outputs/pred_B2A/` (Monet → photo, 300) | `outputs/pred_B2A/` (photo → Monet, 300), `outputs/pred_A2B/` (Monet → photo, 300) |
| Plots | `outputs/plots/loss_curves.png`, `failure_candidates_{B2A,A2B}.jpg` | `outputs/plots/training_curves_…png`, `translation_examples.jpg`, `failure_candidates.jpg` |
| Evaluation code | `src/part3_evaluation_shreya.ipynb`, `src/run1_team_metrics.ipynb`, `src/human_audit.py` | `evaluate_local.py`, `src/snapshot_sweep.py` |

---

## Appendix — Individual failure and error analyses (verbatim excerpts)

Brief §6.6 asks for each member's failure/error analysis with the actual snippets attached. This
appendix reproduces them from each member's `failure_analysis.md` (Tasks 1 and 3) and error review
(Task 2). Review texts are shortened with "…"; full texts and observations are in the linked files.

### A. Task 1 — generated-text failure cases

**shreya_akotiya** (`task1_llm/shreya_akotiya/failure_analysis.md`)

*Case 1 — Semantic contradiction* (greedy decoding). Failure type: Semantic incoherence / contradiction.

```text
Lily said, "Yes, please!" Her mom said, "Yes, I can play with you." Lily was 
happy to hear that her mom was sad.
```

*Case 2 — Pronoun confusion and reference collapse* (T=0.5). Failure type: Pronoun/reference confusion.

```text
At the park, Lily saw a boy crying. She asked him what was wrong. He said he 
was lost and he wanted to play with him. Lily didn't want to share his toy 
friends with her friends. She said he would help him find his toy friends.
```

*Case 3 — Non-word generation at high temperature* (T=1.0). Failure type: Spelling breakdown / non-word generation.

```text
Once upon a time, there was a princess who was very cheerful. She crept for 
quickly and strets, but the pray still wouldn't like something more.
```

**zoheb_waghu** (`task1_llm/zoheb_waghu/failure_analysis.md`)

*Case 1 — Phrase-level repetition loop* (prompt `Once upon a time` (greedy)). Failure type: Repetition at the phrase and clause level.

```text
They saw a big box of cars and a big box. The box was so happy and the box was so happy to see the box.
Lily was so happy that she had been so brave and she was so happy to have her box before. She was happy to have her box before and she was proud of herself.
```

*Case 2 — High-temperature word corruption and semantic breakdown* (prompt `Once upon a time` (temperature 1.2)). Failure type: Invalid words and loss of semantic coherence.

```text
Once upon a time, there was a little girl who loved very melty. She wanted a new hidden aboven the ground and said "Bob! MPleaf!" Her pocket seemed sad to melt. Lily loved to. One day, the sun came across a face of the shiny things. He approached it and cheered. He had the things she, spreadying to eat them.
"Uh-ohw, Lily! Grump! Don't be careful now and hurt!" he screamed.
```

*Case 3 — Story-boundary confusion* (prompt `The little girl` (greedy)). Failure type: Loss of discourse coherence through an unintended story restart.

```text
The little girl was so happy to have her friends to help her friends and they all laughed and smiled. They had lots of fun together and the little girl was happy to have a new friend to play with.

Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she went to the park with her mommy. She saw a big box of colors and she wanted to play with it.
```

### B. Task 2 — manual error reviews (20 errors per member)

**shreya_akotiya** — BiLSTM (`task2_sentiment/shreya_akotiya/failure_analysis.md`); last group = reviews containing a negation

| # | Group | Review text (excerpt) | True | p(pos) | Error type | Proposed fix |
|---|---|---|---|---|---|---|
| 1 | confident FP | Wow love the place and everything is very clean and new! Great place to come and relax worth a try! | neg | 0.999 | Rating-text mismatch | Check for 1-2 star ratings with positive language (sarcasm detection) |
| 2 | confident FP | Though I'm a Copper enthusiast... Maharani was fine enough... The Tikka Masala was spicy and pretty good, but it wasn't as thick and saucy as i like. | neg | 0.999 | Mixed sentiment | Add phrase-level sentiment aggregation; "but" clauses often flip polarity |
| 3 | confident FP | NOTE: This was a 4-star review, but the food quality and ESPECIALLY customer service have gone down the tubes. | neg | 0.999 | Temporal shift | Model sees historical praise, misses "have gone down"; add recency weighting |
| 4 | confident FP | Do you believe in Yin and Yang?... The couple who was seated five minutes after you were. See how they're now eating something? | neg | 0.999 | Sarcasm/irony | Rhetorical questions + comparison to others = complaint; hard to fix without pragmatics |
| 5 | confident FP | Like Clay P... I too love pancakes... they did a respectable job. It is definitely a worthwhile destination for a pancake lover. | neg | 0.998 | Faint praise | "respectable" and "worthwhile" are lukewarm; fine-tune on graded sentiment |
| 6 | confident FN | EDIT: They really did change the service up since I last posted this. Horrible service. Used to be my favorite pizza... | pos | 0.0001 | Temporal shift (update) | "EDIT" signals revision; model fixates on "Horrible service" from old review |
| 7 | confident FN | This place is so much better since they changed owners... It was horrible. Now its much better. | pos | 0.0001 | Negation of past | "was horrible" dominates; model misses "now much better" inversion |
| 8 | confident FN | The food is crap. I'm not trying to be mean, but it really is horrible... Taco Bell's nachos are like manna from heaven compared to the sad mess Barney's serves. | pos | 0.0008 | Label noise | This reads negative; likely mislabeled in dataset |
| 9 | confident FN | Ever wonder what to do if you have lots of extra garbage... This waste facility allows Phoenix residents to dump bulk trash for free once a month. | pos | 0.001 | Domain term | "garbage", "trash", "dump" trigger negative; actually informational positive |
| 10 | confident FN | TERRIBLE SERVICE, RUDE WAITERS WITH A PISS POOR ATTITUDE! WOULD EAT HERE AGAIN! A++++ | pos | 0.001 | Sarcasm | All-caps negative words; the sarcastic "WOULD EAT HERE AGAIN! A++++" is missed |
| 11 | near threshold | It gets the job done. What do you want, it's a Sbarro's... My only beef with Sbarro's is really a beef with the food court | neg | 0.500 | Low-info review | Neutral/functional language; model has no strong signal |
| 12 | near threshold | Well i hate to be the bearer of bad news...but these doughnuts are average at best... I think i will stick to my Krispy Cremes | neg | 0.500 | Comparative | Negative is implicit via comparison to competitor; add comparative features |
| 13 | near threshold | I must have been there on a bad night... there were not actually any people there. Even the free bottle of vodka did not help | neg | 0.501 | Hedged negative | "must have been" hedges; model uncertain |
| 14 | near threshold | This is the new occupant... The beef was pretty good, and so was the noodle... but since it comes mixed with noodles and then a whole bunch of rice, I really felt meat-d… | pos | 0.499 | Mixed with complaint | Positive phrases + "but" clause tips it |
| 15 | near threshold | We orders crepes and cheese fondue... The crepes are nice. I don't like the taste of the cheese fondue | pos | 0.498 | Split verdict | Half positive, half negative; model splits the difference |
| 16 | slice: negation | It was Anniversary time! But we didn't want to spend a ton of money... They do seem treated well but I wonder if they would be happier | pos | – | Negation scope | "didn't want to spend" scopes over intent, not experience; parse negation targets |
| 17 | slice: negation | This review is for the pharmacy only. You do not need to be a member... Cost for a 90 day supply is around $25. Contrast this to $45 at Walmart | pos | – | Informational negation | "do not need" is a benefit, not complaint; distinguish negation of requirement |
| 18 | slice: negation | So, after getting hosed on my room rate last year... Never use i4vegas. Ever... SouthPointe matched the lower price | pos | – | Negation of competitor | Negative is about competitor, not subject; coreference resolution needed |
| 19 | slice: negation | gasp. 1/2 star docked... they don't let you modify any of the standard burgers... they only let you take off toppings | pos | – | Negation signals limitation | Complaint about policy but overall positive; need aspect-level sentiment |
| 20 | slice: negation | I've just been forced to concede that, despite still not digging their ordering process, their food is just too good to disrespect with a 2 star review. | pos | – | Concessive structure | "despite not digging" is subordinate; main clause is positive. Parse syntax |

**zoheb_waghu** — BiLSTM-mean baseline (`task2_sentiment/zoheb_waghu/outputs/error_review_t2_m1_baseline.md`); last group = long reviews (> 200 tokens)

| # | Group | Review text (excerpt) | True | p(pos) | Error type | Proposed fix |
|---|---|---|---|---|---|---|
| 1 | confident FP | Wow love the place and everything is very clean and new! Great place to come and relax worth a try! Cheers, Eric Van Nguyen Visited April 2012 | neg | 1.0000 | rating-text mismatch | Manually audit and correct rating/text mismatches in the training labels, then retrain. |
| 2 | confident FP | What I love about Rubios' is that they always have beer. Always. That is all I love though... | neg | 0.9993 | mixed sentiment | Add sentence-level attention so the final limiting clause ("all I love though") can outweigh the opening prai… |
| 3 | confident FP | This is the neighborhood Foodland that has the bare necessities needed to sustain a pantry or for when the next snowstorm of the century is a day away and you only have… | neg | 0.9991 | mixed sentiment | Encode sentences separately and aggregate their polarity instead of mean-pooling all tokens equally. |
| 4 | confident FP | Updated... they stopped serving Malibu Rum last year so it's no surprise they've closed and changed the theme of the place. MRB REJECT!! Prior Review: Ahhhh. Happy hour… | neg | 0.9991 | mixed sentiment | Preserve paragraph/update boundaries and give the newest review update more weight than the older positive re… |
| 5 | confident FP | Attended the @SpaFitFinder launch party, with @spacephx. I can't say much about the place, other than, even taking into consideration the number of people present, it fe… | neg | 0.9987 | mixed sentiment | Use aspect-level pooling to separate the negative venue assessment from praise of one drink and one employee. |
| 6 | confident FN | EDIT: They really did change the service up since I last posted this. Horrible service. Used to be my favorite pizza in the city (at a reasonable price), but I'm rethink… | pos | 0.0001 | rating-text mismatch | Audit rating/text consistency and train with a noise-robust loss or remove confirmed mismatches. |
| 7 | confident FN | This place is so much better since they changed owners. My wife and I went when it was the old owners, it was terrible. We waited forever and the food never came before… | pos | 0.0004 | mixed sentiment | Add temporal discourse features so "better since they changed owners" outweighs complaints about the former o… |
| 8 | confident FN | I won't say what spilled on my floor carpets, but their vacuums can REALLY suck! Thank goodness because I thought my carpet in my truck was ruined. | pos | 0.0006 | domain term | Train with subword features or character n-grams so idiomatic product praise such as "vacuums can REALLY suck… |
| 9 | confident FN | It was Anniversary time! But we didn't' want to spend a ton of money on food or booze. Plus, were weren't interested in going to a show at the time. So, what to do? Aqua… | pos | 0.0006 | mixed sentiment | Use sentence-level attention trained to emphasize the concluding recommendation over descriptive complaints. |
| 10 | confident FN | its an enjoyable atmosphere for all 21+ (: The beer is Delicious and so is the food - However I unfortunately, can not say the same about the HELP.The service was terrib… | pos | 0.0007 | mixed sentiment | Add aspect-aware aggregation so positive food/atmosphere evidence can be evaluated separately from negative s… |
| 11 | near threshold | great cheap gas station!! I'm always putting in gas for my road trips downtown! :) they even have cones that separate lines so that cars don't get into crazy turning acc… | pos | 0.4998 | domain term | Add character/subword n-gram features for sparse venue terms such as Costco, gas, cones, and road-trip langua… |
| 12 | near threshold | I came here with my boyfriend's family (who are Mauritian) on a Sunday night for his mom's birthday dinner. The decor is decent, nothing too outstanding. Although the pl… | neg | 0.5003 | mixed sentiment | Replace mean pooling with sentence-level attention that can emphasize the repeated service failures over neut… |
| 13 | near threshold | I called \""Anyime Garage Doors\"" because there is a man in my gated community who works there and I always see his truck. I'm real big on supporting local business's.… | neg | 0.5006 | mixed sentiment | Use hierarchical pooling to emphasize the final complaint and quoted-price reversal over the positive local-b… |
| 14 | near threshold | I wish I could give this place Minus 5 stars! This place was a huge waste of time. It's basically a bar in a large freezer. And minus 5 degrees is actually measured in C… | neg | 0.5008 | sarcasm/irony | Preserve punctuation and rating expressions such as "minus 5 stars" as explicit features. |
| 15 | near threshold | This company was great! I was given a 3hr time frame and they showed up in less than 2hrs! I wasn't there when they got there because I had no way to get there and they… | pos | 0.4989 | mixed sentiment | Add contrast-aware sentence aggregation so the positive service assessment outweighs unrelated negative event… |
| 16 | slice: long review | Port Authority (formerly known as PATransit, or \""PAT\"") operates a fairly extensive network of buses and (in the South Hills) light rail. Instead of running school bu… | pos | 0.3450 | length truncation | Raise `max_len` from 256 to 512 and retrain while holding all other settings fixed. |
| 17 | slice: long review | I don't much like the look of this place - I never would have ventured in were it not for the positive yelp reviews - but they serve some pretty good pizza. I have eaten… | pos | 0.1887 | mixed sentiment | Use sentence-level attention to discount the negative appearance/opening clause after the review pivots to pr… |
| 18 | slice: long review | Thoroughly impressed with this airport. Not that I'm some great world traveler, but all the more reason. See, when I booked my virgin transatlantic flight (yes, virgin w… | pos | 0.0960 | negation | Add an explicit negation-scope feature so phrases such as "not that" and "wouldn't drive me" are not treated… |
| 19 | slice: long review | I booked a stay at Hotel San Carlos for one night through Groupon for $76 after taxes. Most other hotels in the area go for $120-300, so finding this deal was awesome. M… | pos | 0.3064 | mixed sentiment | Use hierarchical sentence pooling so the overall stay assessment is not diluted by individual complaints in a… |
| 20 | slice: long review | What era is this? That was my first thought as I stepped into this restaurant. Mirrored ceilings, velvet upholstered chairs, cheetah print fabric -- it was like I was in… | neg | 0.9505 | length truncation | Raise `max_len` to 512 or use chunked hierarchical encoding so the closing negative verdict is retained. |

### C. Task 3 — image failure cases

Each row shows **source | translation | cycle reconstruction**, with LPIPS (how much the image
changed) and cycle L1. Full grids (15 cases per direction) are in each member's `outputs/plots/`.

**shreya_akotiya — photo → Monet, "most changed" cases.** Night and low-light photos collapse into a
repeated blue-grey tiled pattern, yet the reconstruction restores them almost exactly (cycle L1
0.013–0.021): the hidden-information ("steganography") behaviour discussed in §3.5.

<img src="figures/t3_shreya_most_changed_B2A.jpg" style="width:62%">

**shreya_akotiya — Monet → photo, "most changed" cases.** Hazy paintings turn very dark or
high-contrast, and flat skies pick up horizontal streak artifacts.

<img src="figures/t3_shreya_most_changed_A2B.jpg" style="width:62%">

**zoheb_waghu — failure candidates (both directions).** Least changed, most changed and worst-cycle
cases, including the hue shifts (orange clouds → cyan, storm sky → teal) specific to the ResNet-9.

<img src="../task3_gan/zoheb_waghu/outputs/plots/failure_candidates.jpg" style="width:80%">

---

## References

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., &
   Polosukhin, I. (2017). *Attention Is All You Need.* Advances in Neural Information Processing
   Systems 30.
2. Eldan, R., & Li, Y. (2023). *TinyStories: How Small Can Language Models Be and Still Speak
   Coherent English?* arXiv:2305.07759.
3. Zhang, X., Zhao, J., & LeCun, Y. (2015). *Character-level Convolutional Networks for Text
   Classification.* Advances in Neural Information Processing Systems 28. (Source of the Yelp
   Polarity dataset.)
4. Kim, Y. (2014). *Convolutional Neural Networks for Sentence Classification.* Proceedings of
   EMNLP 2014.
5. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8),
   1735–1780.
6. Bahdanau, D., Cho, K., & Bengio, Y. (2015). *Neural Machine Translation by Jointly Learning to
   Align and Translate.* ICLR 2015.
7. McNemar, Q. (1947). *Note on the sampling error of the difference between correlated
   proportions or percentages.* Psychometrika, 12(2), 153–157.
8. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural
   Networks.* ICML 2017.
9. Zhu, J.-Y., Park, T., Isola, P., & Efros, A. A. (2017). *Unpaired Image-to-Image Translation
   using Cycle-Consistent Adversarial Networks.* ICCV 2017.
10. Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A. (2017). *Image-to-Image Translation with
    Conditional Adversarial Networks.* CVPR 2017. (UNet generator, PatchGAN discriminator.)
11. Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical
    Image Segmentation.* MICCAI 2015.
12. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.*
    CVPR 2016.
13. Mao, X., Li, Q., Xie, H., Lau, R. Y. K., Wang, Z., & Smolley, S. P. (2017). *Least Squares
    Generative Adversarial Networks.* ICCV 2017.
14. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2017). *GANs Trained
    by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.* NeurIPS 2017. (FID.)
15. Bińkowski, M., Sutherland, D. J., Arbel, M., & Gretton, A. (2018). *Demystifying MMD GANs.*
    ICLR 2018. (KID.)
16. Kynkäänniemi, T., Karras, T., Laine, S., Lehtinen, J., & Aila, T. (2019). *Improved Precision
    and Recall Metric for Assessing Generative Models.* NeurIPS 2019.
17. Naeem, M. F., Oh, S. J., Uh, Y., Choi, Y., & Yoo, J. (2020). *Reliable Fidelity and Diversity
    Metrics for Generative Models.* ICML 2020. (Density and coverage.)
18. Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018). *The Unreasonable
    Effectiveness of Deep Features as a Perceptual Metric.* CVPR 2018. (LPIPS.)
19. Chu, C., Zhmoginov, A., & Sandler, M. (2017). *CycleGAN, a Master of Steganography.*
    arXiv:1712.02950.

# Task 1 - GPT From Scratch: Results

**Student:** Shreya Akotiya  
**Run:** `task1_shreya_deep_narrow_20260920_194002`  
**Configuration:** `config.yaml`  
**Notebook:** `src/task1_char_gpt.ipynb`

| Evidence | Path |
|---|---|
| Checkpoint | `checkpoints/task1_shreya_deep_narrow_20260920_194002_best.pt` (sha256 `f540786b…`) |
| Raw log | `reproducibility/raw_logs/shreya_akotiya/task1_llm/task1_shreya_deep_narrow_20260920_194002.log` |
| Manifest | `reproducibility/manifests/shreya_akotiya/task1_shreya_deep_narrow_20260920_194002.json` |
| Metrics | `metrics_report.csv`, `metrics_report_extended.csv` |
| Plots / samples | `outputs/plots/`, `outputs/samples/` |

## 1. What I built

For this task, I built a small GPT-style language model from scratch using the TinyStories dataset. The model reads text one character at a time and tries to predict the next character. It has 12 Transformer blocks, a hidden size of 256, and about 9.57 million trainable parameters. I trained it for 10 epochs using 100,000 training sequences and 10,000 validation sequences, with each sequence containing 256 characters.

I wrote the main Transformer parts myself. These include multi-head causal self-attention, LayerNorm, feed-forward layers, residual connections, token embeddings, positional embeddings, and the final language-model head. I did not use a pretrained model or PyTorch's prebuilt Transformer or attention modules. The notebook also checks for banned modules such as `nn.Transformer`, `nn.MultiheadAttention`, and `F.scaled_dot_product_attention`, and the check passed.

### Causal-mask check

The causal mask prevents the model from looking at future characters. To test this, I changed the character at position 192 and compared the model's output before and after the change.

| Positions checked | Largest logit change | Result |
|---|---:|---|
| Before position 192 | 0.000 | Pass - unchanged |
| Position 192 and after | 0.807 | Pass - changed |

This shows that changing a future character did not affect earlier positions. The causal-mask plot is saved as `outputs/plots/causal_mask_check.png`.

## 2. Data preprocessing

I used the first 34 million characters from the shared TinyStories file. The split was made by story, rather than by randomly splitting character windows. This prevents part of the same story from appearing in both training and validation.

| Step | Result |
|---|---|
| Stories used | 197,250 total: 177,525 for training and 19,725 for validation |
| Characters | 30,567,642 training and 3,432,263 validation characters |
| Vocabulary | 78 character symbols, including `<unk>` |
| Training sequences | 100,000 sequences of length 256 |
| Validation sequences | 10,000 sequences of length 256 |
| Sequence format | The target is the input shifted one character to the right |
| Rare-character rate | About 0.0003% in both splits |

I normalized characters such as smart quotes and long dashes before building the vocabulary. The vocabulary was created from the training split only. Characters that appeared fewer than 50 times were mapped to `<unk>`, while common letters, numbers, spaces, punctuation, and other important characters were kept.

I also checked that the input and target sequences were aligned correctly and that every character ID was valid. For example, the input starts with `One day...`, while the target starts with `ne day...`, which confirms that the target is shifted by one character.

## 3. Model architecture and design choices

| Component | Choice | Reason |
|---|---|---|
| Tokenization | Character-level, 78 symbols | This was required by the assignment and keeps the vocabulary small. |
| Context length | 256 characters | It covers roughly one short story paragraph and fits on the T4 GPU. |
| Hidden size | 256 | This allowed me to use a deeper model without making it too large. |
| Transformer layers | 12 | More layers give the model more steps to combine characters into words and sentences. |
| Attention heads | 8 | Each head has 32 dimensions and can focus on different local patterns. |
| Feed-forward size | 1,024 | This follows the usual four-times-the-hidden-size setting. |
| Normalization | Pre-norm LayerNorm | This helps keep training stable in a deeper model. |
| Position information | Learned positional embeddings | The context length is fixed at 256, so learned positions are suitable here. |
| Dropout | 0.05 | The earlier reference model appeared to be under-fitting, so I used slightly less dropout. |
| Weight tying | Not used | The character vocabulary is small, so tying the weights would not save much. |

The model has **9,570,816 total parameters**. Of these, **9,485,312** are non-embedding parameters.

### Why I chose a deeper model

Before training, I compared three model shapes with similar sizes:

| Model shape | Parameters | Time per step |
|---|---:|---:|
| 384 hidden size, 6 layers | 10.81M | 169 ms |
| **256 hidden size, 12 layers** | **9.58M** | **225 ms** |
| 512 hidden size, 4 layers | 12.84M | 173 ms |

The 12-layer model takes longer per step, but I chose it because character-level modeling requires several levels of learning. The model first needs to learn character patterns and spelling, then words, phrases, and simple story structure. I expected the extra depth to help with this composition. The trade-off is that the model is slower than the wider, shallower alternatives.

## 4. Training setup

I used cross-entropy loss because the task is to predict the correct next character. The main training settings were:

| Setting | Value | Reason |
|---|---|---|
| Epochs | 10 | This is the minimum required by the assignment. |
| Batch size | 64 sequences, or 16,384 characters per step | Fits within the available GPU memory. |
| Optimizer | AdamW with beta values `(0.9, 0.95)` | Works well for Transformer language models. |
| Learning rate | 0.00025, decreasing to 0.000025 | The lower rate helps stabilize the 12-layer model. |
| Learning-rate schedule | 5% warm-up followed by cosine decay | Warm-up gives the model time to stabilize at the start. |
| Weight decay | 0.1 on weight matrices | Adds regularization without shrinking biases and normalization parameters. |
| Gradient clipping | Maximum norm of 1.0 | Helps prevent unusually large updates. |
| Random seed | 1337 | Makes the run reproducible. |

The model processed about 255.9 million characters during training. The learning rate warmed up for 781 steps and then gradually decreased.

## 5. Hardware

The main training run was completed in Google Colab using an NVIDIA Tesla T4 GPU with 15.64 GB of memory. The system used Python 3.13.15, PyTorch 2.11.0 with CUDA 12.8, and NumPy 2.1.3.

| Measurement | Result |
|---|---:|
| Total training time | 176.2 minutes |
| Training speed | 24,209 characters/second |
| Generation speed | 87.6 characters/second |
| Peak GPU memory | 8,059 MB |
| Peak process memory | 1,963 MB |

The model-shape benchmarks were run separately on an Apple M5 Max with 48 GB of unified memory using MPS.

## 6. Results and metrics

| Metric | Result |
|---|---:|
| Final training cross-entropy | 0.6038 nats/character |
| Best validation cross-entropy | 0.6327 nats/character |
| Validation perplexity | 1.883 |
| Bits per character | 0.913 |
| Generalization gap | 0.0289 nats/character |
| Top-1 next-character accuracy | 79.9% |
| Distinct-1 / 2 / 3, first 2,000 words | 0.351 / 0.814 / 0.950 |
| Repeated word 4-gram rate | 3.8% |
| Average / maximum gradient norm | 0.687 / 17.79 |
| Loss spikes | 22 |
| NaN or Inf events | 0 |
| Number of parameters | 9,570,816 |

The validation loss improved at every epoch. It reached its best value at epoch 10, which means the model was still improving when training stopped. The loss curves are saved in `outputs/plots/loss_curves.png`.

### What the training curves show

The training loss was higher than the validation loss during the first few epochs. This does not indicate data leakage. The training loss is averaged over the whole epoch, including the difficult early steps, while validation is measured at the end of the epoch with dropout turned off. The generalization gap became positive around epoch 6 and reached only 0.0289 by epoch 10.

Overall, the model appears to be under-fitting more than over-fitting. The validation loss was still decreasing, and the final epoch was also the best epoch. Training was stable: there were no NaN or Inf values. The largest gradient norm occurred at the beginning of training, and the gradients became smaller after warm-up. The 22 loss spikes were small and recovered on the next step, so they were more likely caused by difficult batches than by a serious training problem.

## 7. Generated text

The prompt used for generation was **“Once upon a time”**.

### Greedy decoding

```text
Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine.
One day, she saw a big bird flying in the sky. She was so happy and said, "Hi, I am Lily. Do you
want to play with me?"

Lily said, "Yes, please!" Her mom said, "Yes, I can play with you." Lily was happy to hear that
her mom was sad.
```

### Temperature 0.8

```text
Once upon a time, in a big park, there was a little girl named Lily. She loved to play with her
toys and go outside to play. One day, her mommy found a map in the garden and asked her what was
wrong. Lily told her that it was a little expensive music that she loved the music very much.

"Oh no! I want to be happy but I don't like my music. It's too hot!" Ben said.
```

The model learned several surface-level patterns well. It usually produced correctly spelled words at lower temperatures, used spaces and punctuation properly, created dialogue with quotation marks, and followed the common TinyStories pattern of introducing a character and beginning with “One day.”

However, it struggled with meaning across multiple sentences. For example, Lily is described as being happy that her mother was sad. The model also sometimes changed characters, used confusing pronouns, or produced sentences that sounded grammatical but did not make sense. The generated stories also stopped before reaching a real conclusion.

### Temperature comparison

| Setting | Result |
|---|---|
| Greedy | No non-word errors, but repeated the common opening sentence. |
| T = 0.5 | Very readable, but still repeated the same opening pattern. |
| T = 0.8 | Best balance: only 0.42% non-word rate and no repeated phrase detected. |
| T = 1.0 | More variety, but non-word rate increased to 1.29%; examples included “strets” and “shaked.” |
| T = 1.2 | Most varied, but also the least reliable, with a 3.04% non-word rate and examples such as “pumpins” and “knowled.” |

This shows the usual temperature trade-off. Lower temperatures make the output safer but more repetitive. Higher temperatures increase variety but also make spelling and meaning less reliable. For this model, T = 0.8 gave the best overall balance.

## 8. Limitations and future improvements

The main limitation is that the model was trained for only 10 epochs on a limited portion of TinyStories. Since the validation loss was still decreasing, training for 15-20 epochs or using more of the dataset could improve the results.

Other improvements I would try are:

- Add top-k or top-p sampling to reduce unlikely character choices while keeping more variety than greedy decoding.
- Add a key-value cache during generation. This would make generation faster without changing the output quality.
- Compare the 12-layer model with the 6-layer model using the same training time, so the effect of depth can be measured more fairly.
- Try a longer context window, such as 512 characters, to see whether the model can maintain characters and events across more sentences.
- Increase dropout only after longer training, if the generalization gap becomes noticeably larger.

The model completed the required from-scratch GPT implementation and learned basic spelling, punctuation, and story patterns. Its main weakness is long-range coherence, which is expected given the small model size, limited dataset, short context, and character-level prediction approach.

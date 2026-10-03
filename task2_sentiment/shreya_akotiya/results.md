# Task 2 - Yelp Polarity Sentiment Classification

**Student:** Shreya Akotiya  
**Run:** `task2_shreya_main_20260921_222011`  
**Configuration:** `config.yaml`  
**Notebook:** `src/task2_sentiment.ipynb`

| Evidence | Path |
|---|---|
| Checkpoints | `checkpoints/baseline_meanpool.pt`, `checkpoints/exp_textcnn.pt`, `checkpoints/exp_bilstm.pt` |
| Raw log | `reproducibility/raw_logs/shreya_akotiya/task2_sentiment/task2_shreya_main_20260921_222011.log` |
| Manifest | `reproducibility/manifests/shreya_akotiya/task2_shreya_main_20260921_222011.json` |
| Metrics | `metrics_report.csv`, `metrics_report_extended.csv`, `outputs/slice_metrics.csv`, `outputs/mcnemar.csv` |
| Plots | `outputs/plots/eda.png`, `outputs/plots/confusion_matrices.png`, `outputs/plots/curves.png` |
| Error review | `failure_analysis.md`, `outputs/error_review.md` |

I trained and compared three sentiment-classification models on the Yelp Polarity dataset. The models were a simple mean-pooling baseline, a TextCNN, and a BiLSTM. All three models learned their word embeddings during training. I did not use GloVe, word2vec, fastText, a pretrained language model, or any other pretrained embeddings.

A second run with the same configuration and seed produced almost the same results. The baseline matched to four decimal places, and the TextCNN and BiLSTM were within 0.001 validation accuracy. The small difference is due to GPU operations that are not perfectly deterministic.

## 1. Data and preprocessing

The dataset contains 560,000 training reviews and 38,000 test reviews. The classes are perfectly balanced: half of the reviews are positive and half are negative. I held out 20,000 reviews from the training set for validation, leaving 540,000 reviews for training.

The exploratory analysis showed that the average raw review contains 133 words, while the median is 97 words. Negative reviews tend to be longer than positive reviews, probably because people often give more details when explaining a complaint.

There were no missing reviews, empty entries, or invalid labels. The preprocessing code still checks for these problems. Thirty-five reviews became empty after preprocessing, so they were kept as padding-only examples instead of being removed.

The text-processing steps were:

1. Convert text to lowercase.
2. Remove punctuation and special characters. For example, `don't` becomes `dont`.
3. Remove common stopwords, but keep negation words such as `no`, `not`, `never`, `dont`, and `didnt`. This is important because removing the word “not” could change “not good” into “good.”
4. Apply light suffix stemming to combine simple variations such as plural and verb forms. I used light stemming instead of aggressive Porter stemming because the embeddings can learn that related words are similar.
5. Split each review into tokens.
6. Build the vocabulary from the training set only. Words appearing fewer than five times were mapped to `<unk>`.
7. Pad or truncate each review to 200 tokens.

The final vocabulary contained 48,436 words and covered 99.47% of the processed tokens. The unknown-word rate was 0.52%. After preprocessing, reviews contained an average of 63.5 tokens. A length of 200 kept 96.6% of the reviews completely and preserved 96.3% of all tokens.

All models used a randomly initialized 200-dimensional embedding layer. The embedding weights were learned from the Yelp reviews during training.

## 2. Models and design choices

All three models used the same embedding size so that the comparison focused on the model architecture rather than a larger embedding layer.

| Model | Architecture | Parameters | Main idea |
|---|---|---:|---|
| `baseline_meanpool` | Average the word embeddings, then use a linear classifier | 9,687,602 | Simple reference model |
| `exp_textcnn` | Convolution filters with widths 3, 4, and 5, followed by max-pooling | 9,928,102 | Detect useful local phrases |
| `exp_bilstm` | One-layer BiLSTM with 128 hidden units in each direction and masked max-pooling | 10,025,634 | Use information from across the whole review |

### Mean-pooling baseline

The baseline averages all word embeddings, so it does not understand word order. For example, “not good” and “good, not” would look very similar. Even with this limitation, it provides a useful comparison because it is fast and inexpensive.

### TextCNN

The TextCNN uses filters that look at groups of three to five words. This allows it to learn phrases such as “not good at all” or “would not return.” Max-pooling keeps the strongest phrase signal from the review. The limitation is that it mainly sees short local patterns and cannot directly connect information that is far apart.

### BiLSTM

The BiLSTM reads each review in both directions. This allows information from earlier and later parts of a review to influence the representation of each word. I used masked max-pooling instead of only using the final hidden state because the important sentence could appear anywhere in the review. The mask prevents padding tokens from being selected as the strongest feature.

## 3. Training settings

| Setting | Value | Reason |
|---|---|---|
| Epochs | 3 | Validation performance reached its best point by epoch 2 or 3. |
| Batch size | 256 | Fits the available GPU memory. |
| Optimizer | Adam, learning rate 0.001 | A standard starting point for learned text embeddings. |
| Weight decay | None | Dropout was used for regularization. |
| Dropout | 0.3 (baseline), 0.5 (TextCNN), 0.3 (BiLSTM) | TextCNN's 300 concatenated filter features overfit faster, so it gets more dropout. |
| Gradient clipping | Norm 1.0 | Helps prevent unusually large LSTM updates. |
| Embedding dimension | 200 | Kept the same for all three models. |
| Maximum review length | 200 tokens | Keeps almost all reviews intact. |
| Random seed | 1337 | Makes the experiment reproducible. |

Each model saved its best validation checkpoint before being evaluated on the official 38,000-review test set.

## 4. Hardware

All three models were trained in Google Colab on an NVIDIA Tesla T4 GPU with 15.6 GB of memory. The run used PyTorch 2.11.0 with CUDA 12.8 on an x86_64 Linux machine.

## 5. Main results

| Model | Accuracy | Macro-F1 | ROC-AUC | PR-AUC | MCC | Brier score | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mean-pooling baseline | 0.9314 | 0.9314 | 0.9793 | 0.9787 | 0.8629 | 0.0516 | **0.0046** |
| TextCNN | 0.9435 | 0.9435 | 0.9863 | 0.9866 | 0.8870 | 0.0430 | 0.0111 |
| BiLSTM | **0.9485** | **0.9485** | **0.9891** | **0.9894** | **0.8970** | **0.0384** | 0.0086 |

The BiLSTM achieved the best overall classification performance. The TextCNN was second, and the mean-pooling baseline was third. Since the test set is balanced, the macro-F1, weighted-F1, and accuracy values are the same or nearly the same.

| Model | Precision (macro) | Recall (macro) | F1 (macro) | F1 (micro) | F1 (weighted) |
|---|---:|---:|---:|---:|---:|
| Mean-pooling baseline | 0.9315 | 0.9314 | 0.9314 | 0.9314 | 0.9314 |
| TextCNN | 0.9435 | 0.9435 | 0.9435 | 0.9435 | 0.9435 |
| BiLSTM | 0.9485 | 0.9485 | 0.9485 | 0.9485 | 0.9485 |

Micro-F1 is always equal to accuracy for single-label classification.

### Confusion matrices

| Model | True negatives | False positives | False negatives | True positives |
|---|---:|---:|---:|---:|
| Mean-pooling baseline | 17,795 | 1,205 | 1,401 | 17,599 |
| TextCNN | 17,850 | 1,150 | 997 | 18,003 |
| BiLSTM | 18,091 | 909 | 1,048 | 17,952 |

The BiLSTM made the fewest total errors. The confusion matrices are saved in `outputs/plots/confusion_matrices.png`.

### Confidence intervals and statistical comparison

The 95% bootstrap intervals were calculated using 1,000 resamples.

| Model | Accuracy interval | Macro-F1 interval | MCC interval |
|---|---|---|---|
| Mean-pooling baseline | [0.9289, 0.9339] | [0.9289, 0.9339] | [0.8578, 0.8679] |
| TextCNN | [0.9412, 0.9458] | [0.9412, 0.9458] | [0.8824, 0.8915] |
| BiLSTM | [0.9463, 0.9507] | [0.9463, 0.9507] | [0.8926, 0.9014] |

The confidence intervals do not overlap, so the ranking is clearly separated on this test set. I also used paired McNemar tests against the baseline:

| Comparison | Reviews fixed only by baseline | Reviews fixed only by experimental model | p-value |
|---|---:|---:|---:|
| Baseline vs. TextCNN | 775 | 1,234 | 1.6 × 10⁻²⁴ |
| Baseline vs. BiLSTM | 540 | 1,189 | 9.4 × 10⁻⁵⁵ |

Both experimental models corrected more baseline errors than they introduced, and both differences were statistically significant. I did not run a McNemar test directly between TextCNN and BiLSTM, so the difference between those two models should be interpreted more carefully.

### Calibration

The baseline had the best expected calibration error, 0.0046, meaning its confidence scores were the most reliable even though its accuracy was lower. The TextCNN had the highest ECE, 0.0111, which suggests that its max-pooled features sometimes produced overconfident predictions. The BiLSTM had the best Brier score, showing the best combination of accuracy and probability quality.

## 6. Speed and resource use

| Model | Training time for 3 epochs | Examples/second | Peak GPU memory |
|---|---:|---:|---:|
| Mean-pooling baseline | 41 seconds | 39,502 | 338 MB |
| TextCNN | 248 seconds | 6,547 | 460 MB |
| BiLSTM | 350 seconds | 4,637 | 2,286 MB |

The baseline was much faster and used the least memory. The TextCNN improved the accuracy while remaining relatively efficient. The BiLSTM performed best, but it was the slowest and used much more memory because its sequence processing is harder to parallelize.

## 7. Robustness by data slice

I also measured macro-F1 and error rate for short, medium, and long reviews, as well as reviews with and without negation.

| Slice | Number of reviews | Baseline | TextCNN | BiLSTM |
|---|---:|---|---|---|
| 40 tokens or fewer | 16,731 | 0.931 / 6.8% | 0.943 / 5.6% | 0.949 / 5.0% |
| 41-100 tokens | 14,542 | 0.932 / 6.8% | 0.945 / 5.4% | 0.949 / 5.1% |
| More than 100 tokens | 6,727 | 0.924 / 7.2% | 0.934 / 6.3% | 0.941 / 5.6% |
| Contains negation | 28,544 | 0.924 / 7.4% | 0.940 / 5.9% | 0.945 / 5.3% |
| No negation | 9,456 | 0.927 / 5.3% | 0.930 / 5.0% | 0.936 / 4.6% |

Long reviews were the most difficult for every model. Some contain both positive and negative opinions, and the longest reviews may lose information when truncated at 200 tokens. The BiLSTM remained the most accurate model on long reviews (5.6% error), which is consistent with its ability to carry information through the sequence. The baseline's error increased the least (+0.4 points), mainly because it was already weaker on short reviews.

Reviews containing negation were also harder. The baseline's error rate increased by 2.1 percentage points on the negation slice, while the increases for TextCNN and BiLSTM were only 0.8 and 0.7 points. This supports the decision to keep negation words during preprocessing. The sequence models were better at handling phrases such as “not good” than the mean-pooling model.

## 8. Overall comparison

The baseline was a useful starting point because it was fast, memory-efficient, and already reached 93.1% accuracy. Its main weakness was that it ignored word order.

The TextCNN improved accuracy by 1.2 percentage points. It handled local phrases and negation better than the baseline, while using only 460 MB of GPU memory. Its limitation was that it could not easily connect words or ideas that were far apart.

The BiLSTM performed best on the main test metrics and the robustness slices. It had the lowest error rate on long reviews and on reviews containing negation. Its main disadvantages were training time and memory use. It also started to lose validation accuracy by the third epoch, which suggests that longer training would lead to overfitting.

Although the models have different parameter totals, about 97% of each model's parameters are in the shared-size embedding table. Therefore, the accuracy improvement came mainly from adding better sequence processing, while the main cost was additional computation and memory.

## 9. Error review

I manually reviewed 20 BiLSTM errors (5 confident false positives, 5 confident false negatives, 5 near-threshold errors, and 5 errors from reviews containing negation). Each one has an error type and a testable fix in `failure_analysis.md`.

## 10. Limitations and future improvements

The next improvements I would test are:

- Replace BiLSTM max-pooling with attention pooling so the model can learn which positions are most important.
- Use early stopping and learning-rate decay, especially for the BiLSTM, since its validation accuracy peaked before the final epoch.
- Apply temperature scaling to TextCNN probabilities to improve its calibration without changing accuracy.
- Run a direct McNemar test between TextCNN and BiLSTM.
- Increase the maximum review length to 300 to check whether truncation is affecting the long-review results.

Class weighting is not necessary because the dataset is evenly balanced. Overall, the experiment showed that even simple models can perform well on Yelp sentiment, but models that preserve word order and sequence information are more accurate and more robust than a model that only averages word embeddings.

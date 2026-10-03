# Shared raw data — Yelp polarity

Source: https://huggingface.co/datasets/fancyzhx/yelp_polarity (560K train / 38K test, binary)

Raw data is **not committed** (size).

> **Dataset confirmed: Yelp polarity.** Both members train on it and evaluate on the same
> official 38K test split.

## Fetch

Loaded through the HuggingFace `datasets` library and cached under `~/.cache/huggingface/`:

```bash
python -c "from datasets import load_dataset; load_dataset('yelp_polarity')"
```

## Protocol the team must share

- Validation is carved out of the **training** split; the official 38K test split is touched
  **once**, at final evaluation.
- The vocabulary is built from the training split only.
- All of one member's models must consume identical splits and vocabulary, or the paired
  McNemar test between them is invalid.

Do not put preprocessing output here — that goes in your own `<member>/data_processed/`.

## shreya_akotiya

`shreya_akotiya` reads local CSVs. This writes `train.csv` and `test.csv`
here (columns `label,text`, 0 = negative, 1 = positive):

```bash
python task2_sentiment/shreya_akotiya/src/fetch_data.py
```

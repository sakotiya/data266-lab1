# Shared raw data — TinyStories

Source: https://huggingface.co/datasets/roneneldan/TinyStories

Raw data is **not committed** (size).

## Fetch

No manual download needed. `src/data.py` streams the dataset through the HuggingFace
`datasets` library and takes only the characters the config asks for, so a 110K-character
experiment never materialises the full ~2 GB corpus:

```python
load_dataset("roneneldan/TinyStories", split="train", streaming=True).shuffle(seed=SEED, buffer_size=10_000)
```

It caches under `~/.cache/huggingface/` rather than in this folder. To pre-warm the cache:

```bash
python -c "from datasets import load_dataset; load_dataset('roneneldan/TinyStories', split='train')"
```

Both members' configs name the dataset as `data.dataset`, so the source is identical across
the team even though the character counts and splits are per member.

Do not put preprocessing output here — that goes in your own `<member>/data_processed/`,
which is never shared (brief, section 5).

## shreya_akotiya

`shreya_akotiya` reads a local text file instead of streaming. This writes
`tinystories_raw.txt` here, with stories separated by blank lines:

```bash
python task1_llm/shreya_akotiya/src/fetch_data.py --chars 36000000 --stream
```

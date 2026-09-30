# temp/ — shared assignment inputs

Staged here by zoheb_waghu. These are **assignment inputs**, not deliverables — move them
wherever the team prefers.

| File | What it is |
|---|---|
| `DATA266_Lab1_Fall_2026.pdf` | The brief itself. Where any derived checklist disagrees with it, the PDF wins. |
| `Part3_Evaluation_Script.ipynb` | **Instructor-provided CycleGAN evaluator.** Task 3 numbers should come from this. |

## Part3_Evaluation_Script.ipynb — read before writing Task 3 code

It computes **FID and MiFID in both directions** and expects this flat layout:

```
Part 3/Data/
  monet_jpg/   real Monet       = domain A real
  photo_jpg/   real Photo       = domain B real
  pred_A2B/    generated Photo  (Monet -> Photo)
  pred_B2A/    generated Monet  (Photo -> Monet)
```

**Mind the direction convention:** `pred_A2B` is Monet→Photo, i.e. domain A is *Monet*. If your
generator naming assumes A = photo, your two directions are swapped relative to this script.

Both members' Task 3 scaffolds were written before this script was found, so neither matches its
interface yet. `task3_gan/zoheb_waghu/evaluate_local.py` is stubbed for a config-driven layout
and a different metric set (KID, LPIPS, density/coverage rather than MiFID) — reconcile rather
than running both and reporting whichever looks better.

## Two things that could NOT be pushed

**1. The Python environment (983 MB).** Not pushable and not desirable:

- a single file, `torch/lib/libtorch_cpu.dylib`, is **169 MB** — past GitHub's 100 MB hard limit,
  so the push would be rejected
- it is macOS/arm64-only, so it would not work on a CUDA machine or the GPU Lab anyway
- 60+ files under `.venv/bin` contain the absolute path `/Users/zohebw/...`, which the brief
  (section 1) forbids in any committed file

Rebuild it instead — `requirements.txt` at the repo root pins the exact versions used for
zoheb_waghu's reported runs:

```bash
/usr/bin/python3 -m venv .venv          # on Apple Silicon: MUST be this interpreter, see root README
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"   # expect True
```

**2. The Monet/photo dataset (113 MB, 7,338 files).** Deliberately gitignored — the brief keeps
raw datasets out of the repo, and `task3_gan/data/README.md` holds the fetch command:

```bash
kaggle competitions download -c gan-getting-started -p /tmp
unzip -q /tmp/gan-getting-started.zip -d task3_gan/data
# expect: task3_gan/data/monet_jpg 300 files, task3_gan/data/photo_jpg 7038 files
```

You must join the competition on Kaggle first or the download 403s.

The TinyStories pool for Task 1 is fetched the same way — see `task1_llm/data/README.md`.

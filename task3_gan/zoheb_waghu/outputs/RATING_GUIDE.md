# Human audit — rating guide (Task 3)

Read this **before** scoring. Round 1 produced Cohen's κ = −0.05 (worse than chance), with
`artifacts` the worst axis — the two raters were applying different definitions, not seeing
different images. This guide fixes the definitions so the second round measures the model
rather than our disagreement.

## What you are looking at

Each `human_audit/sample_NN.jpg` is **two 256×256 panels side by side**:

```
[ LEFT = the original input ]   [ RIGHT = the model's translation ]
```

30 samples, shuffled, filenames stripped. **Do not open `human_audit_manifest.csv`** until both
raters have finished — it maps each sample back to its source file and direction.

**The target domain is the opposite of the left panel.** If the left panel is a painting, the
right panel is trying to be a photograph; if the left panel is a photograph, the right panel is
trying to be a Monet. You can tell which from the image — that is fine and is not what is blinded.

## Score each sample on three axes, 1–5, whole numbers only

Score the three axes **independently**. A sample can be 5 for style and 1 for content.

### 1. Style — does the right panel belong to the target domain?

Ignore whether it kept the content. Judge only: does it look like the thing it was trying to become?

| | |
|---|---|
| **1** | No style change. The right panel still clearly belongs to the source domain. |
| **2** | Slight shift — a hint of the target's colour or texture, but it still reads as the source. |
| **3** | Mixed. Convincing in parts, unconvincing in others. You could argue either way. |
| **4** | Clearly the target domain, with a few giveaways. |
| **5** | Convincing. You would not question it at a glance. |

### 2. Content — is the scene from the left panel still there?

Ignore how good the style is. Judge only: are the same objects in the same places?

| | |
|---|---|
| **1** | Scene is gone or unrecognisable — objects lost, invented, or rearranged. |
| **2** | Major content loss: a significant object or region is missing or replaced. |
| **3** | Overall layout survives, but details are lost or smeared. |
| **4** | Scene intact; only small details differ. |
| **5** | Every object and boundary from the left panel is present and in place. |

### 3. Artifacts — how clean is the image?

**Score 5 = clean, 1 = badly damaged.** (Round 1's negative κ came from this axis, most likely
because the direction of the scale was not agreed. High is good, as on the other two axes.)

Count only defects that are **not** part of either domain's normal appearance. Monet-like
brushwork in a photo→Monet translation is *style*, not an artifact.

Artifacts are: checkerboard or grid patterns, repeating tiles, colour blotches unrelated to the
scene, halos or ghosting at edges, melted or smeared regions, hard seams, noise speckle.

| | |
|---|---|
| **1** | Severe — artifacts dominate the image. |
| **2** | Obvious at a glance, across much of the image. |
| **3** | Noticeable in a few places once you look. |
| **4** | Minor; you have to look closely. |
| **5** | Clean. None visible. |

## Procedure

1. Score all 30 in order. Do not go back and adjust earlier scores to be consistent.
2. Do not discuss scores with the other rater while rating — that destroys the independence κ measures.
3. Fill `human_audit_ratings.csv`: `rater1` = Zoheb, `rater2` = Shreya. Every cell filled, whole numbers 1–5.
4. When both are done: `python task3_gan/zoheb_waghu/evaluate_local.py audit-score --config task3_gan/zoheb_waghu/configs/cyclegan_baseline.yaml`

## Reporting

Round 1 is kept in `human_audit_ratings_round1.csv` and `human_audit_summary_round1.json`. The
report should state that a first round was run without an agreed rubric, produced κ = −0.05, and
was repeated after this guide was written — that sequence is itself a finding about subjective
evaluation, and the brief grades honesty about shortcomings.

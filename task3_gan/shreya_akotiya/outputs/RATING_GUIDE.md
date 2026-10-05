# Human audit — rating guide (Task 3, shreya_akotiya)

Read this **before** scoring. Same rubric as `task3_gan/zoheb_waghu/outputs/RATING_GUIDE.md`, so
the two members' audits are comparable.

Why it exists: zoheb_waghu's first audit round was run without an agreed rubric and produced
Cohen's κ = −0.05 — agreement *worse than chance* — almost entirely because the two raters scored
the `artifacts` axis in opposite directions. Re-rating the same sheets against these anchors took
κ to +0.149 with 100% of ratings within one point. Use the anchors.

## What you are looking at

Each `human_audit/sample_NN.jpg` is **two 256×256 panels side by side**:

```
[ LEFT = the original input ]   [ RIGHT = the model's translation ]
```

30 samples, shuffled, filenames stripped. **Do not open `human_audit_manifest.csv`** until both
raters have finished — it maps each sample back to its source file and direction.

**The target domain is the opposite of the left panel.** Left is a painting → the right panel is
trying to be a photograph. Left is a photograph → the right panel is trying to be a Monet.

## Score each sample on three axes, 1–5, whole numbers only

**5 is best on all three axes.** Score the axes **independently** — a sample can be 5 for style
and 1 for content.

### 1. Style — does the right panel belong to the target domain?

| | |
|---|---|
| **1** | No style change; still clearly the source domain. |
| **2** | Slight shift, but it still reads as the source. |
| **3** | Mixed — convincing in parts, not in others. |
| **4** | Clearly the target domain, with a few giveaways. |
| **5** | Convincing; you would not question it at a glance. |

### 2. Content — is the scene from the left panel still there?

| | |
|---|---|
| **1** | Scene gone or unrecognisable. |
| **2** | A significant object or region missing or replaced. |
| **3** | Layout survives, details lost or smeared. |
| **4** | Scene intact, small details differ. |
| **5** | Every object and boundary preserved. |

### 3. Artifacts — how clean is the image? **5 = clean, 1 = badly damaged**

This is the axis that broke round 1 of the other audit. It is scored in the **same direction** as
the others: high is good.

Count only defects belonging to neither domain — checkerboard or grid patterns, repeating tiles,
colour blotches unrelated to the scene, halos or ghosting at edges, melted or smeared regions,
hard seams, noise speckle. **Monet-like brushwork in a photo→Monet translation is *style*, not an
artifact.**

| | |
|---|---|
| **1** | Severe — artifacts dominate. |
| **2** | Obvious at a glance across much of the image. |
| **3** | Noticeable in a few places once you look. |
| **4** | Minor; you have to look closely. |
| **5** | Clean; none visible. |

## Procedure

1. Score all 30 in order; do not revise earlier scores for consistency.
2. **Do not discuss scores with the other rater while rating** — that destroys the independence κ measures.
3. Fill `human_audit_ratings.csv`. Decide and record which person is `rater1` and which is `rater2`.
4. Then: `python task3_gan/shreya_akotiya/src/human_audit.py score`

## One caveat worth knowing

These 30 sheets use the **same 30 source images, in the same order**, as zoheb_waghu's audit —
both scripts sample with seed 42 from the first 300 sorted predictions. That makes the two
members' per-axis means directly comparable, which is useful. But if you rate this set straight
after the other one, you will recognise the scenes and may anchor to the scores you just gave.
The models differ, so the right panels differ; judge what is in front of you. If you would rather
remove the risk entirely, change `SEED` in `src/human_audit.py` and regenerate — the sheets will
be a different sample, at the cost of the two audits no longer sharing source images.

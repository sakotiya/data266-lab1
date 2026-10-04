# Human audit — rating guide

Each `sample_XX.jpg` shows the **original image on the left** and the **model's translation on the
right**. Some translations turn a photo into a Monet-style painting, others turn a Monet painting
into a photo. You are not told which model made them; rate each image on its own.

Fill in `human_audit_ratings.csv`, only your own rows (`rater1` or `rater2`). Use whole numbers 1–5.
Rate alone and don't discuss scores with the other rater until you have both finished.

| Score | Style — does the right image look like the target domain (a real Monet painting, or a real photo)? | Content — are the left image's scene, objects and layout preserved? | Artifacts — how clean is the right image? |
|---|---|---|---|
| 5 | Convincing; could pass as real | Everything important is kept | No visible artifacts |
| 4 | Mostly convincing, small giveaways | Small details lost or changed | Minor, only on close look |
| 3 | Partly there | Main subject kept, noticeable changes | Noticeable but not distracting |
| 2 | Weak; mostly still looks like the source | Layout or main objects distorted | Distracting (blotches, smears, checkerboard, colour patches) |
| 1 | No style change, or wrong domain | Scene unrecognisable | Severe; image is broken |

For artifacts, higher is better (5 = clean).

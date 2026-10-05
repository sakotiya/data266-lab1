"""Compute human-audit means and inter-rater Cohen's kappa for §3.3.

Reads human_audit_ratings.csv (rater1, rater2) + human_audit_manifest.csv
(sample -> direction), then reports, split by direction (A2B / B2A):
  - mean style / content / artifacts (averaged over both raters)
  - Cohen's kappa between the two raters, per dimension and pooled,
    both unweighted and quadratic-weighted (ratings are ordinal 1-5).

Audit covers Zoheb's model only, so results fill the two "Zoheb" columns.
"""
import pandas as pd
from sklearn.metrics import cohen_kappa_score

here = __file__.rsplit("/", 1)[0]
ratings = pd.read_csv(f"{here}/human_audit_ratings.csv")
manifest = pd.read_csv(f"{here}/human_audit_manifest.csv")[["sample_id", "direction"]]

df = ratings.merge(manifest, on="sample_id")
dims = ["style", "content", "artifacts"]
r1 = df[df.rater_id == "rater1"].set_index("sample_id")
r2 = df[df.rater_id == "rater2"].set_index("sample_id")
dirn = manifest.set_index("sample_id")["direction"]


def kappa(a, b, weights=None):
    return cohen_kappa_score(a, b, labels=[1, 2, 3, 4, 5], weights=weights)


for direction in ["A2B", "B2A"]:
    ids = dirn[dirn == direction].index
    sub = df[df.sample_id.isin(ids)]
    print(f"\n=== {direction}  (n={len(ids)} images) ===")
    means = sub.groupby("rater_id")[dims].mean()
    overall = sub[dims].mean()
    print("  mean per rater:")
    print(means.to_string().replace("\n", "\n    "))
    print("  mean (both raters): " +
          ", ".join(f"{d}={overall[d]:.2f}" for d in dims))

    a1, a2 = r1.loc[ids], r2.loc[ids]
    print("  Cohen's kappa (rater1 vs rater2):")
    for d in dims:
        k = kappa(a1[d], a2[d])
        kw = kappa(a1[d], a2[d], weights="quadratic")
        print(f"    {d:<9} unweighted={k:+.3f}   quadratic-weighted={kw:+.3f}")
    # pooled across the 3 dimensions (3*n paired observations)
    pa = pd.concat([a1[d] for d in dims]).values
    pb = pd.concat([a2[d] for d in dims]).values
    print(f"    POOLED    unweighted={kappa(pa, pb):+.3f}   "
          f"quadratic-weighted={kappa(pa, pb, weights='quadratic'):+.3f}")

# overall (all 30, pooled dimensions)
pa = pd.concat([r1[d] for d in dims]).values
pb = pd.concat([r2[d] for d in dims]).values
print(f"\n=== ALL 30 images, pooled dimensions (n=90 paired) ===")
print(f"  unweighted={kappa(pa, pb):+.3f}   "
      f"quadratic-weighted={kappa(pa, pb, weights='quadratic'):+.3f}")

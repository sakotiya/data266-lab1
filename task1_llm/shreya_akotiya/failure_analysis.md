# Task 1 — failure analysis (shreya_akotiya)

## Case 1: Semantic contradiction

**Generated snippet** (greedy decoding)

```text
Lily said, "Yes, please!" Her mom said, "Yes, I can play with you." Lily was 
happy to hear that her mom was sad.
```

**Failure type:** Semantic incoherence / contradiction

**Observation:** The model generates grammatically correct text but produces a logical contradiction — Lily is "happy to hear" something negative ("her mom was sad"). This happens because the character-level model predicts tokens based on local patterns without maintaining semantic consistency. The phrase "happy to hear" is commonly followed by positive content in training data, but the model fails to enforce that constraint across the sentence boundary.

---

## Case 2: Pronoun confusion and reference collapse

**Generated snippet** (T=0.5)

```text
At the park, Lily saw a boy crying. She asked him what was wrong. He said he 
was lost and he wanted to play with him. Lily didn't want to share his toy 
friends with her friends. She said he would help him find his toy friends.
```

**Failure type:** Pronoun/reference confusion

**Observation:** The model loses track of who "he", "him", and "his" refer to. In one sentence Lily "didn't want to share his toy" — but "his" should be "her" since Lily is the subject. The phrase "he wanted to play with him" is self-referential nonsense. This is a known failure mode in autoregressive LMs: without explicit coreference tracking, the model samples pronouns that fit local n-gram statistics but violate discourse-level consistency.

---

## Case 3: Non-word generation at high temperature

**Generated snippet** (T=1.0)

```text
Once upon a time, there was a princess who was very cheerful. She crept for 
quickly and strets, but the pray still wouldn't like something more.
```

**Failure type:** Spelling breakdown / non-word generation

**Observation:** At temperature 1.0, the model produces non-words like "strets" (likely intended "streets") and uses "pray" where "prey" or another word was intended. The non-word rate jumps from 0% at greedy/T=0.5 to 1.29% at T=1.0 and 3.04% at T=1.2. This reflects the exploration–exploitation tradeoff: higher temperature samples from the tail of the character distribution, occasionally producing plausible-looking but invalid character sequences. The model has learned English spelling patterns but not perfectly — rare bigrams get sampled under high temperature.

---

## Cross-cutting observations: temperature trade-off

The temperature comparison shows a clear trade-off. Greedy decoding and T=0.5 produced no non-word errors, but both repeated the phrase "Once upon a time..." and the sentence about Lily. T=0.8 gave the best balance between variety and readability, with only a 0.42% non-word rate and no detected repeated phrases. At T=1.0 and T=1.2, repetition decreased, but spelling and coherence problems increased, with non-word rates of 1.29% and 3.04%. This shows that higher temperature improves diversity but makes the generated text less reliable.

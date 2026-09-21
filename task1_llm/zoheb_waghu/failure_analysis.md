# Task 1 - Sequence model failure analysis (2 marks)

Run: `t1_baseline_20260919-232507` · Checkpoint: `t1_baseline_20260919-232507_best.pt`
Model: 4-layer / 4-head / 256-dim char GPT, 3,262,464 params, 12 epochs on 100K TinyStories characters.
Samples: [outputs/samples/samples_t1_baseline_20260919-232507.txt](outputs/samples/samples_t1_baseline_20260919-232507.txt)

Diversity measured per decoding strategy (all three prompts pooled):

| Strategy | distinct-1 | distinct-2 | distinct-3 | repeated 4-gram rate |
|---|---|---|---|---|
| greedy | 0.0287 | 0.1434 | 0.2464 | **0.6824** |
| temperature 0.8 | 0.0342 | 0.2207 | 0.4984 | 0.3081 |
| temperature 1.2 | 0.0383 | 0.2690 | 0.6062 | 0.1894 |

---

## Case 1 - Degenerate repetition loop (greedy decoding)

**Prompt:** `Once upon a time`

**Generated (verbatim):**
```
Once upon a time, there was a little girl named Lily. She loved to play and said, "What is a
looked around and said, "I will you can be starte."

The little girl named Lily was so happy. They loved to the bird and said, "I will you you can
to the careful and said, "I will you you can to the careful and said, "I will you you can to
the care and said, "I will you you can
```

**Failure type:** Repetition (degenerate loop).

**Observation:** The opening clause is clean - the model has genuinely learned TinyStories'
formulaic opening. It collapses the moment it leaves that pattern. Greedy decoding takes the
argmax at every step, so once the model enters the high-probability state `and said, "I will`
it re-enters the same state deterministically: the continuation is a fixed point. The
measured repeated-4-gram rate of **0.6824** under greedy against **0.1894** at temperature 1.2
confirms this is a decoding property, not a weights property - the same checkpoint escapes the
loop as soon as sampling injects noise. The underlying cause is that 100K characters is too
small a corpus to teach long-range discourse structure: with a 256-character context the model
sees roughly one sentence of history, so nothing in its input distinguishes the third repetition
from the first.

---

## Case 2 - Broken morphology: fluent-looking non-words (temperature 0.8)

**Prompt:** `The little girl`

**Generated (verbatim):**
```
The little girl not her day. They happy it wing in the phatt. It was a like angrion.

"Can I good to the coop it?" asked. They said, "It's sorre to go and behorses."

Lila and Anna with is smaring. It was so havol and head rom it.
```

**Failure type:** Broken grammar / invalid vocabulary (sub-word morphology failure).

**Observation:** `phatt`, `angrion`, `sorre`, `behorses`, `smaring`, `havol` are not English
words, but they are all *phonotactically plausible* English - correct consonant clusters,
plausible endings (`-ion`, `-ing`, `-es`). This is the signature failure of character-level
modelling: the model has learned the conditional distribution over the next *character*, not
over words, so it has no representation that enforces "this sequence of characters must be a
word in the vocabulary". It produces locally valid character transitions that accumulate into
invalid words. Top-1 next-character accuracy is 58.3%, which is high enough for convincing
letter-by-letter texture and far too low to keep a 6-character word on track. A word-level or
BPE tokeniser would remove this failure class outright, at the cost of an out-of-vocabulary
problem the character model does not have.

---

## Case 3 - Loss of coherence and reproduced encoding artifacts (temperature 1.2)

**Prompt:** `The little girl`

**Generated (verbatim):**
```
The little girl very slovemped to hat were flop outside owor and peplen. He dhot himsey
looked afwit. She tey their not yare so run and cared him the birds. One day, a bocky Thimmy
mun run't off cly. Some, this mommy with all of fun it?â€ She had Sam ws heas to she didn't
dowed agch Tim.
```

**Failure type:** Loss of coherence, plus hallucinated data artifacts.

**Observation:** Two separate things fail here. First, at temperature 1.2 the flattened
distribution lets low-probability characters through at every step and coherence collapses
entirely - referents switch (`She` → `He` → `She`), and the sentence never resolves. Second,
and more interesting, the model emits **`â€`** - a mojibake fragment produced when a UTF-8
curly quote is mis-decoded as Latin-1. The model did not invent this; it is reproducing a
corruption present in the TinyStories text I ingested. That is a data-quality finding, not a
modelling one: my preprocessing does not normalise or strip mis-encoded byte sequences, so they
entered the vocabulary as legitimate characters and the model learned to emit them. It is the
clearest evidence in these samples that generated-text inspection catches preprocessing bugs
that aggregate loss metrics hide - validation loss is 1.4194 either way.

---

## Cross-case note

Cases 1 and 3 are the two ends of one axis, and the metrics table above makes that explicit:
as temperature rises, repeated-4-gram rate falls 0.68 → 0.19 while distinct-3 rises 0.25 → 0.61,
and coherence degrades monotonically in the opposite direction. Neither end is good, which says
the useful operating point is between them (temperature ~0.8, or nucleus sampling, which this
run does not implement). Case 2 is orthogonal - it is a tokenisation consequence and no decoding
temperature fixes it.

**The one change I would test first:** train on more data rather than tune decoding. The
generalization gap is 0.2473 (train 1.1721 vs val 1.4194), so the model is already fitting the
100K-character corpus harder than it generalises; increasing the corpus to 1M characters should
close the gap and improve all three failure modes at once. Predicted effect if I am right:
validation bits-per-character drops below the current 2.0478 and the greedy repeated-4-gram rate
falls without any change to decoding.

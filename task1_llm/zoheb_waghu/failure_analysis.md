# Task 1 - Sequence model failure analysis (2 marks)

Run: `t1_baseline_20260928-151435` · Checkpoint: `t1_baseline_20260928-151435_best.pt`
Model: 4-layer / 4-head / 256-dim char GPT, 3,274,752 params, 10 epochs over
**100,000 training sequences** (25,600,001 characters, chars [34M, 59.6M) of the shared pool).
Samples: [outputs/samples/samples_t1_baseline_20260928-151435.txt](outputs/samples/samples_t1_baseline_20260928-151435.txt)

Diversity measured per decoding strategy (all three prompts pooled):

| Strategy | distinct-1 | distinct-2 | distinct-3 | repeated 4-gram rate |
|---|---|---|---|---|
| greedy | 0.0263 | 0.1442 | 0.2648 | **0.6488** |
| temperature 0.8 | 0.0295 | 0.2016 | 0.4195 | 0.4206 |
| temperature 1.2 | 0.0359 | 0.2346 | 0.5511 | 0.2478 |

> **What changed against my earlier 100K-*character* run.** Under the team's corrected reading
> of spec 1.1 this model saw 256x more text (25.6M vs 100K characters). The
> **invalid-word failure mode essentially disappeared** - the old run emitted `phatt`,
> `angrion`, `smaring`, `havol`; this one produces almost entirely real English. So did the
> `â€` mojibake, because the shared pre-joined corpus is clean where my earlier streamed slice
> was not. What is left is harder: the text is now locally fluent and fails at the level of
> *meaning* and *discourse* instead of spelling.

---

## Case 1 - Phrase-level repetition loop (greedy decoding)

**Prompt:** `Once upon a time`

**Generated (verbatim):**
```
Once upon a time, there was a little girl named Lily. She loved to play outside in the
sunshine and explore the world. One day, she went to the park with her mommy and daddy. Lily
was so excited to see her mommy and daddy.
As they were walking, Lily saw a big box of colors and she saw a big box of colors. She
wanted to see what was inside. She wanted to see what was inside. She took a big bite of the
colors and
```

**Failure type:** Repetition (degenerate loop), at clause level rather than character level.

**Observation:** The repeated-4-gram rate under greedy is **0.6488**, which is *higher* than
the weaker 100K-character model's 0.5319 - more data made this failure worse, not better. The
reason is that the loop is now made of well-formed clauses: `she saw a big box of colors` and
`She wanted to see what was inside.` are each repeated verbatim. A better-trained model assigns
sharper probability to the single most likely continuation, and greedy decoding takes that
argmax every time, so a confident model is *more* prone to entering a fixed point, not less.
Note also `Lily was so excited to see her mommy and daddy` immediately after her parents were
already introduced - the model has no mechanism for tracking what it has already said beyond
its 256-character window. This is a decoding failure, not a weights failure: the identical
checkpoint at temperature 1.2 scores 0.2478 on the same metric.

---

## Case 2 - Locally fluent, semantically impossible

**Prompt:** `Once upon a time` (temperature 0.8) and `The little girl` (temperature 1.2)

**Generated (verbatim, three instances):**
```
Timmy loved to play in the snow, eat cookies for his long teddy every day.
...
Timmy put the bitter in a blue frog and he never forgot the frog.
```
```
He had lots of spicy fox one knife. One day he heard a snow, and he saw a picture of up in
their garden. Eighten, he took another picture clothes for him.
```
```
So one day her stomach to a key and sneeze. First that day on, from the girl still had been
heroes and gold!
```

**Failure type:** Loss of coherence - specifically semantic and selectional-preference
violation, with syntax largely intact.

**Observation:** Every word here except `Eighten` is a real English word, and the local syntax
is mostly grammatical - which is exactly what makes the failure visible. `eat cookies for his
long teddy`, `put the bitter in a blue frog`, `lots of spicy fox one knife`, `her stomach to a
key and sneeze`: the model has learned which characters follow which, and enough word
statistics to spell reliably, but it has no representation of what nouns can be eaten, what
adjectives apply to what, or that `bitter` is not a thing you put somewhere. This is the
predictable consequence of a 3.27M-parameter model with a 256-character context trained purely
on next-character likelihood - nothing in the objective rewards semantic consistency, only
local predictability. `Eighten` shows the old morphology failure is suppressed rather than
eliminated: it still surfaces at high temperature.

---

## Case 3 - Story-boundary confusion: restarting mid-generation

**Prompt:** `The little girl` (greedy)

**Generated (verbatim):**
```
The little girl was so happy that she had a great time and the little girl was so happy to
have her friends to play with her toys.

Once upon a time, there was a little girl named Lily. She loved to play outside in the park
with her friends. One day, she went to the park with her mommy and daddy.
```

**Failure type:** Loss of coherence at the discourse level - the model abandons the current
story and begins a new one.

**Observation:** Mid-generation the model emits a blank line and then `Once upon a time, there
was a little girl named Lily` - a fresh story opening, from a prompt that was already
mid-narrative. This is learned behaviour, not noise: in the shared corpus stories are separated
by a blank line, so `\n\n` followed by `Once upon a time` is one of the highest-probability
sequences in the training distribution. Having reached a plausible ending (`was so happy`), the
model does what the data says usually happens next, which is to start another story. It is the
clearest evidence in these samples that the model has learned *corpus* structure rather than
*narrative* structure, and it also explains part of Case 1's high repetition: Lily, the park and
her mommy and daddy recur because they are the corpus's modal story, and the model keeps
regenerating it.

---

## Cross-case note

The three cases are ordered by scope: a loop inside a clause (1), a broken relation inside a
sentence (2), and an abandoned story across paragraphs (3). All three are failures of
*state* - the model has a 256-character window and no memory beyond it, so it cannot know what
it has already said, what it has already claimed about the world, or that it is in the middle of
a story.

Critically, more data did not fix any of them. Comparing the two runs of the same architecture:

| | 100K chars | 25.6M chars |
|---|---|---|
| Validation bits-per-char | 2.0478 | **1.0140** |
| Top-1 next-character accuracy | 58.26% | **77.72%** |
| Generalization gap | 0.2473 | **0.0038** |
| Greedy repeated-4-gram rate | 0.5319 | 0.6488 (worse) |
| Invalid words | pervasive | rare |

Data scale fixed spelling and closed the generalization gap almost entirely; it left repetition
untouched and arguably worse, because those are properties of the decoding rule and the context
length, not of how well the model fits the data.

**The one change I would test first:** nucleus (top-p) sampling, p ~0.9. It is the cheapest
intervention that targets Case 1 directly - it truncates the sharp argmax tail that greedy
follows into loops while keeping the low-temperature coherence that temperature 1.2 destroys.
Predicted effect if I am right: greedy's 0.6488 repeated-4-gram rate falls toward the ~0.25 seen
at temperature 1.2 *without* the semantic breakdown of Case 2, and distinct-3 lands between
0.2648 and 0.5511. Cases 2 and 3 would be unaffected - those need a longer context, a larger
model, or document-boundary masking, none of which are decoding changes.

# Task 1 - Sequence model failure analysis (2 marks)

Run: `t1_baseline_20260929-213500` · Checkpoint: `t1_baseline_20260929-213500_best.pt`
Model: 4-layer / 4-head / 256-dim char GPT, 3,274,752 parameters, trained for 10 epochs over
100,000 sequences. All quotations are from the RTX 4090 run's
[sample file](outputs/samples/samples_t1_baseline_20260929-213500.txt).

Diversity measured per decoding strategy across all three prompts:

| Strategy | distinct-1 | distinct-2 | distinct-3 | repeated 4-gram rate |
|---|---:|---:|---:|---:|
| greedy | 0.0263 | 0.1570 | 0.2919 | **0.6129** |
| temperature 0.8 | 0.0303 | 0.2056 | 0.4545 | 0.3743 |
| temperature 1.2 | 0.0367 | 0.2474 | 0.5751 | 0.2262 |

## Case 1 - Phrase-level repetition loop

**Prompt:** `Once upon a time` (greedy)

**Generated (verbatim):**

```
They saw a big box of cars and a big box. The box was so happy and the box was so happy to see the box.
Lily was so happy that she had been so brave and she was so happy to have her box before. She was happy to have her box before and she was proud of herself.
```

**Failure type:** Repetition at the phrase and clause level.

**Observation:** `the box was so happy` and `happy to have her box before` recur with only
small changes. Greedy decoding repeatedly takes the highest-probability continuation, allowing
a common phrase to feed back into itself. The effect is visible in greedy's 0.6129 repeated
4-gram rate, compared with 0.3743 at temperature 0.8 and 0.2262 at temperature 1.2. This is
primarily a decoding failure: adding randomness reduces the loop, although Case 2 shows its
quality cost.

## Case 2 - High-temperature word corruption and semantic breakdown

**Prompt:** `Once upon a time` (temperature 1.2)

**Generated (verbatim):**

```
Once upon a time, there was a little girl who loved very melty. She wanted a new hidden aboven the ground and said "Bob! MPleaf!" Her pocket seemed sad to melt. Lily loved to. One day, the sun came across a face of the shiny things. He approached it and cheered. He had the things she, spreadying to eat them.
"Uh-ohw, Lily! Grump! Don't be careful now and hurt!" he screamed.
```

**Failure type:** Invalid words and loss of semantic coherence.

**Observation:** `aboven`, `MPleaf`, `spreadying`, and `Uh-ohw` are character-plausible but
invalid forms. Phrases such as `loved very melty`, `Her pocket seemed sad to melt`, and
`Don't be careful now and hurt` are also semantically or pragmatically inconsistent.
Temperature 1.2 improves measured diversity by admitting lower-probability continuations, but
those choices compound until both word formation and meaning fail.

## Case 3 - Story-boundary confusion

**Prompt:** `The little girl` (greedy)

**Generated (verbatim):**

```
The little girl was so happy to have her friends to help her friends and they all laughed and smiled. They had lots of fun together and the little girl was happy to have a new friend to play with.

Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she went to the park with her mommy. She saw a big box of colors and she wanted to play with it.
```

**Failure type:** Loss of discourse coherence through an unintended story restart.

**Observation:** After reaching a plausible ending, the model emits a blank line and begins a
new `Once upon a time` story rather than continuing the prompted narrative. Blank lines followed
by standard openings are common corpus boundaries, so the model has learned the surface pattern
between documents without representing that generation should remain within one story. Explicit
document-boundary handling during training, or a longer context that retains more narrative
state, would directly target this failure.

## Cross-case conclusion

The failures expose a trade-off rather than one universally best decoding strategy. Greedy
decoding is locally fluent but repetitive and prone to high-frequency story templates;
temperature 1.2 is more diverse but damages spelling and meaning. The first experiment I would
run is nucleus (top-p) sampling around `p=0.9`, measuring repeated 4-grams and distinct-3 while
manually checking invalid words. It may retain several plausible choices and interrupt greedy
loops without admitting as much of the low-probability tail as unrestricted temperature 1.2.
Story-boundary confusion is different: sampling alone cannot teach narrative state, so it
requires boundary-aware training or a longer-context model.

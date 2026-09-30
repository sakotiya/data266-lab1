# Task 2 — error review (shreya_akotiya)

Analysis of 20 errors from the BiLSTM model (best performer, 94.85% accuracy).

## Confident false positives (5)

Reviews predicted positive with high confidence but actually negative.

| # | Text snippet | True | Pred | Conf | Error type | Testable fix |
|---|---|---|---|---|---|---|
| 1 | "Wow love the place and everything is very clean and new! Great place to come and relax worth a try!" | neg | pos | 0.999 | Rating-text mismatch | Check for 1-2 star ratings with positive language (sarcasm detection) |
| 2 | "Though I'm a Copper enthusiast... Maharani was fine enough... The Tikka Masala was spicy and pretty good, but it wasn't as thick and saucy as i like." | neg | pos | 0.999 | Mixed sentiment | Add phrase-level sentiment aggregation; "but" clauses often flip polarity |
| 3 | "NOTE: This was a 4-star review, but the food quality and ESPECIALLY customer service have gone down the tubes." | neg | pos | 0.999 | Temporal shift | Model sees historical praise, misses "have gone down"; add recency weighting |
| 4 | "Do you believe in Yin and Yang?... The couple who was seated five minutes after you were. See how they're now eating something?" | neg | pos | 0.999 | Sarcasm/irony | Rhetorical questions + comparison to others = complaint; hard to fix without pragmatics |
| 5 | "Like Clay P... I too love pancakes... they did a respectable job. It is definitely a worthwhile destination for a pancake lover." | neg | pos | 0.998 | Faint praise | "respectable" and "worthwhile" are lukewarm; fine-tune on graded sentiment |

## Confident false negatives (5)

Reviews predicted negative with high confidence but actually positive.

| # | Text snippet | True | Pred | Conf | Error type | Testable fix |
|---|---|---|---|---|---|---|
| 1 | "EDIT: They really did change the service up since I last posted this. Horrible service. Used to be my favorite pizza..." | pos | neg | 0.0001 | Temporal shift (update) | "EDIT" signals revision; model fixates on "Horrible service" from old review |
| 2 | "This place is so much better since they changed owners... It was horrible. Now its much better." | pos | neg | 0.0001 | Negation of past | "was horrible" dominates; model misses "now much better" inversion |
| 3 | "The food is crap. I'm not trying to be mean, but it really is horrible... Taco Bell's nachos are like manna from heaven compared to the sad mess Barney's serves." | pos | neg | 0.0008 | Label noise | This reads negative; likely mislabeled in dataset |
| 4 | "Ever wonder what to do if you have lots of extra garbage... This waste facility allows Phoenix residents to dump bulk trash for free once a month." | pos | neg | 0.001 | Domain term | "garbage", "trash", "dump" trigger negative; actually informational positive |
| 5 | "TERRIBLE SERVICE, RUDE WAITERS WITH A PISS POOR ATTITUDE! WOULD EAT HERE AGAIN! A++++" | pos | neg | 0.001 | Sarcasm | All-caps negative words; the sarcastic "WOULD EAT HERE AGAIN! A++++" is missed |

## Near-threshold errors (5)

Reviews with prediction confidence between 0.45 and 0.55.

| # | Text snippet | True | Pred | Conf | Error type | Testable fix |
|---|---|---|---|---|---|---|
| 1 | "It gets the job done. What do you want, it's a Sbarro's... My only beef with Sbarro's is really a beef with the food court" | neg | pos | 0.500 | Low-info review | Neutral/functional language; model has no strong signal |
| 2 | "Well i hate to be the bearer of bad news...but these doughnuts are average at best... I think i will stick to my Krispy Cremes" | neg | pos | 0.500 | Comparative | Negative is implicit via comparison to competitor; add comparative features |
| 3 | "I must have been there on a bad night... there were not actually any people there. Even the free bottle of vodka did not help" | neg | pos | 0.501 | Hedged negative | "must have been" hedges; model uncertain |
| 4 | "This is the new occupant... The beef was pretty good, and so was the noodle... but since it comes mixed with noodles and then a whole bunch of rice, I really felt meat-deprived." | pos | neg | 0.499 | Mixed with complaint | Positive phrases + "but" clause tips it |
| 5 | "We orders crepes and cheese fondue... The crepes are nice. I don't like the taste of the cheese fondue" | pos | neg | 0.498 | Split verdict | Half positive, half negative; model splits the difference |

## Slice-specific failures (5)

Errors from the "has_negation" slice where negation words are present.

| # | Slice | Text snippet | True | Pred | Error type | Testable fix |
|---|---|---|---|---|---|---|
| 1 | negation | "It was Anniversary time! But we didn't want to spend a ton of money... They do seem treated well but I wonder if they would be happier" | pos | neg | Negation scope | "didn't want to spend" scopes over intent, not experience; parse negation targets |
| 2 | negation | "This review is for the pharmacy only. You do not need to be a member... Cost for a 90 day supply is around $25. Contrast this to $45 at Walmart" | pos | neg | Informational negation | "do not need" is a benefit, not complaint; distinguish negation of requirement |
| 3 | negation | "So, after getting hosed on my room rate last year... Never use i4vegas. Ever... SouthPointe matched the lower price" | pos | neg | Negation of competitor | Negative is about competitor, not subject; coreference resolution needed |
| 4 | negation | "gasp. 1/2 star docked... they don't let you modify any of the standard burgers... they only let you take off toppings" | pos | neg | Negation signals limitation | Complaint about policy but overall positive; need aspect-level sentiment |
| 5 | negation | "I've just been forced to concede that, despite still not digging their ordering process, their food is just too good to disrespect with a 2 star review." | pos | neg | Concessive structure | "despite not digging" is subordinate; main clause is positive. Parse syntax |

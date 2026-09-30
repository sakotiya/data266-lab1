# Task 2 - Error review: `t2_m1_baseline`

Test accuracy 0.9346 · macro-F1 0.9346 · MCC 0.8693 · Brier 0.0489

Error type vocabulary: `negation` · `sarcasm/irony` · `mixed sentiment` · `aspect confusion` · `rating-text mismatch` · `domain term` · `length truncation` · `rare vocabulary / OOV` · `label noise` · `other`

## A. Confident false positives (predicted positive, actually negative)

### 1. test index 29330 — p(pos) = 1.0000, true = negative
- tokens after preprocessing: 19 · exclamations: 2 · negation present: no

> Wow love the place and everything is very clean and new! Great place to come and relax worth a try! Cheers, Eric Van Nguyen Visited April 2012

- **Error type:** `rating-text mismatch`
- **Testable fix:** Manually audit and correct rating/text mismatches in the training labels, then retrain.
- **Metric that should move:** Confident false-positive count should decrease; negative-class precision and macro-F1 should increase.

### 2. test index 3163 — p(pos) = 0.9993, true = negative
- tokens after preprocessing: 7 · exclamations: 0 · negation present: no

> What I love about Rubios' is that they always have beer. Always. That is all I love though...

- **Error type:** `mixed sentiment`
- **Testable fix:** Add sentence-level attention so the final limiting clause ("all I love though") can outweigh the opening praise.
- **Metric that should move:** False positives on mixed-sentiment reviews should decrease and negative-class precision should increase.

### 3. test index 23815 — p(pos) = 0.9991, true = negative
- tokens after preprocessing: 108 · exclamations: 0 · negation present: yes

> This is the neighborhood Foodland that has the bare necessities needed to sustain a pantry or for when the next snowstorm of the century is a day away and you only have minutes to get TP, bread and milk. The bakery here is tops, small selection, but really well made specialities. Huge brownies, iced and moist. They are…

- **Error type:** `mixed sentiment`
- **Testable fix:** Encode sentences separately and aggregate their polarity instead of mean-pooling all tokens equally.
- **Metric that should move:** Macro-F1 on mixed and long reviews should increase; false positives should decrease.

### 4. test index 6186 — p(pos) = 0.9991, true = negative
- tokens after preprocessing: 48 · exclamations: 9 · negation present: yes

> Updated... they stopped serving Malibu Rum last year so it's no surprise they've closed and changed the theme of the place. MRB REJECT!! Prior Review: Ahhhh. Happy hour in the islands. Fruity drinks. Lots of Malibu Rum. Wait!! I'm in Scottsdale?? Darn!! Still a cozy little tiki bar with great happy hour specials!! Grea…

- **Error type:** `mixed sentiment`
- **Testable fix:** Preserve paragraph/update boundaries and give the newest review update more weight than the older positive review.
- **Metric that should move:** False positives on edited reviews should decrease and negative-class recall should increase.

### 5. test index 25815 — p(pos) = 0.9987, true = negative
- tokens after preprocessing: 61 · exclamations: 0 · negation present: yes

> Attended the @SpaFitFinder launch party, with @spacephx. I can't say much about the place, other than, even taking into consideration the number of people present, it felt cramped. The TrimTini (?) I had was really delicious; vodkatini with lemon zest. A very nice gentleman, who works at Urban 7, was very gracious in m…

- **Error type:** `mixed sentiment`
- **Testable fix:** Use aspect-level pooling to separate the negative venue assessment from praise of one drink and one employee.
- **Metric that should move:** False positives on mixed-sentiment reviews should decrease and macro-F1 should increase.

## B. Confident false negatives (predicted negative, actually positive)

### 6. test index 22807 — p(pos) = 0.0001, true = positive
- tokens after preprocessing: 48 · exclamations: 0 · negation present: yes

> EDIT: They really did change the service up since I last posted this. Horrible service. Used to be my favorite pizza in the city (at a reasonable price), but I'm rethinking that. We just had an altercation with a server who refused to split a check when we were paying with cash. He then proceeded to disrespect the part…

- **Error type:** `rating-text mismatch`
- **Testable fix:** Audit rating/text consistency and train with a noise-robust loss or remove confirmed mismatches.
- **Metric that should move:** Confident false-negative count and Brier score should decrease; positive-class recall should increase.

### 7. test index 30793 — p(pos) = 0.0004, true = positive
- tokens after preprocessing: 45 · exclamations: 0 · negation present: yes

> This place is so much better since they changed owners. My wife and I went when it was the old owners, it was terrible. We waited forever and the food never came before we walked out. People were served before us that walked in after and my wife actually got her soup before me and I sat and waited while they \""made mo…

- **Error type:** `mixed sentiment`
- **Testable fix:** Add temporal discourse features so "better since they changed owners" outweighs complaints about the former owners.
- **Metric that should move:** Positive-class recall on edited or contrastive reviews should increase.

### 8. test index 32999 — p(pos) = 0.0006, true = positive
- tokens after preprocessing: 14 · exclamations: 1 · negation present: no

> I won't say what spilled on my floor carpets, but their vacuums can REALLY suck! Thank goodness because I thought my carpet in my truck was ruined.

- **Error type:** `domain term`
- **Testable fix:** Train with subword features or character n-grams so idiomatic product praise such as "vacuums can REALLY suck" is represented in context.
- **Metric that should move:** Positive-class recall on product reviews should increase; confident false negatives should decrease.

### 9. test index 11808 — p(pos) = 0.0006, true = positive
- tokens after preprocessing: 109 · exclamations: 6 · negation present: yes

> It was Anniversary time! But we didn't' want to spend a ton of money on food or booze. Plus, were weren't interested in going to a show at the time. So, what to do? Aquarium! Don't get me wrong: This place is TINY! Narrow walk ways and close quarters in general. Even the exhibits are small. I do hope that all of the an…

- **Error type:** `mixed sentiment`
- **Testable fix:** Use sentence-level attention trained to emphasize the concluding recommendation over descriptive complaints.
- **Metric that should move:** Positive-class recall on mixed-sentiment reviews and macro-F1 should increase.

### 10. test index 6520 — p(pos) = 0.0007, true = positive
- tokens after preprocessing: 17 · exclamations: 0 · negation present: yes

> its an enjoyable atmosphere for all 21+ (: The beer is Delicious and so is the food - However I unfortunately, can not say the same about the HELP.The service was terrible the waitress were rude not only to us but to each other....

- **Error type:** `mixed sentiment`
- **Testable fix:** Add aspect-aware aggregation so positive food/atmosphere evidence can be evaluated separately from negative service evidence.
- **Metric that should move:** Positive-class recall on multi-aspect reviews should increase.

## C. Near-threshold errors (|p - 0.5| <= 0.05)

### 11. test index 33720 — p(pos) = 0.4998, true = positive
- tokens after preprocessing: 31 · exclamations: 5 · negation present: no

> great cheap gas station!! I'm always putting in gas for my road trips downtown! :) they even have cones that separate lines so that cars don't get into crazy turning accidents after gassing up! It's so useful for Costcos to have all these things in one place! I don't put gas anywhere else :)

- **Error type:** `domain term`
- **Testable fix:** Add character/subword n-gram features for sparse venue terms such as Costco, gas, cones, and road-trip language.
- **Metric that should move:** Positive-class recall on rare-domain reviews should increase and OOV-slice macro-F1 should improve.

### 12. test index 29783 — p(pos) = 0.5003, true = negative
- tokens after preprocessing: 234 · exclamations: 0 · negation present: no

> I came here with my boyfriend's family (who are Mauritian) on a Sunday night for his mom's birthday dinner. The decor is decent, nothing too outstanding. Although the place was only 1/3 full, it took the waiters awhile to acknowledge us. Five minutes of waiting around later, we were finally seated, without menus. I had…

- **Error type:** `mixed sentiment`
- **Testable fix:** Replace mean pooling with sentence-level attention that can emphasize the repeated service failures over neutral decor descriptions.
- **Metric that should move:** Negative-class recall and long-review macro-F1 should increase.

### 13. test index 25382 — p(pos) = 0.5006, true = negative
- tokens after preprocessing: 115 · exclamations: 0 · negation present: yes

> I called \""Anyime Garage Doors\"" because there is a man in my gated community who works there and I always see his truck. I'm real big on supporting local business's. All I needed was to have them program a remote for me, I called and they told me 20 dollars. The guy who came out tried to tell me that I needed over 3…

- **Error type:** `mixed sentiment`
- **Testable fix:** Use hierarchical pooling to emphasize the final complaint and quoted-price reversal over the positive local-business preface.
- **Metric that should move:** Negative-class recall on contrastive reviews should increase.

### 14. test index 20711 — p(pos) = 0.5008, true = negative
- tokens after preprocessing: 67 · exclamations: 1 · negation present: yes

> I wish I could give this place Minus 5 stars! This place was a huge waste of time. It's basically a bar in a large freezer. And minus 5 degrees is actually measured in Celsius, so it's really only around 23 degrees Fahrenheit. You're not allowed to take a camera into the bar since they want to earn money from a photogr…

- **Error type:** `sarcasm/irony`
- **Testable fix:** Preserve punctuation and rating expressions such as "minus 5 stars" as explicit features.
- **Metric that should move:** Negative-class recall on sarcasm/rating-language reviews should increase.

### 15. test index 32004 — p(pos) = 0.4989, true = positive
- tokens after preprocessing: 62 · exclamations: 4 · negation present: yes

> This company was great! I was given a 3hr time frame and they showed up in less than 2hrs! I wasn't there when they got there because I had no way to get there and they delivered it to the dealership and the dealership called me when it arrived! Unfortunately I needed a tow truck at the wrong time because a big flood c…

- **Error type:** `mixed sentiment`
- **Testable fix:** Add contrast-aware sentence aggregation so the positive service assessment outweighs unrelated negative event words such as "unfortunately" and "flood."
- **Metric that should move:** Positive-class recall on contrastive reviews should increase and near-threshold errors should decrease.

## D. Worst-slice errors - slice `long_reviews` (macro-F1 0.9151 vs overall 0.9346)

### 16. test index 90 — p(pos) = 0.3450, true = positive
- tokens after preprocessing: 290 · exclamations: 0 · negation present: no

> Port Authority (formerly known as PATransit, or \""PAT\"") operates a fairly extensive network of buses and (in the South Hills) light rail. Instead of running school buses, the school district gave high schoolers bus passes to get to class, and I have used the bus since 1997. I've had a great experience using the bus …

- **Error type:** `length truncation`
- **Testable fix:** Raise `max_len` from 256 to 512 and retrain while holding all other settings fixed.
- **Metric that should move:** `long_reviews` macro-F1 should increase above 0.9151 and positive-class recall should improve.

### 17. test index 363 — p(pos) = 0.1887, true = positive
- tokens after preprocessing: 208 · exclamations: 0 · negation present: yes

> I don't much like the look of this place - I never would have ventured in were it not for the positive yelp reviews - but they serve some pretty good pizza. I have eaten trillions of pizzas. Maybe more, I lost count. And I'm not a a purist or a dogmatic devotee to any particular variety of pizza. I grew up in Chicago, …

- **Error type:** `mixed sentiment`
- **Testable fix:** Use sentence-level attention to discount the negative appearance/opening clause after the review pivots to praise of the pizza.
- **Metric that should move:** Positive-class recall on contrastive reviews should increase.

### 18. test index 439 — p(pos) = 0.0960, true = positive
- tokens after preprocessing: 213 · exclamations: 1 · negation present: yes

> Thoroughly impressed with this airport. Not that I'm some great world traveler, but all the more reason. See, when I booked my virgin transatlantic flight (yes, virgin with a lower case v--as in, I was popping my passport cherry; not capital V, as in going on the airline), I needed an airport that wouldn't drive me bat…

- **Error type:** `negation`
- **Testable fix:** Add an explicit negation-scope feature so phrases such as "not that" and "wouldn't drive me" are not treated as direct negative evidence.
- **Metric that should move:** `contains_negation` macro-F1 should increase above 0.9294 and confident false negatives should decrease.

### 19. test index 770 — p(pos) = 0.3064, true = positive
- tokens after preprocessing: 243 · exclamations: 1 · negation present: yes

> I booked a stay at Hotel San Carlos for one night through Groupon for $76 after taxes. Most other hotels in the area go for $120-300, so finding this deal was awesome. My boyfriend and I live in AZ and he wanted to celebrate his birthday by having a staycation downtown so we could have fun with family and friends witho…

- **Error type:** `mixed sentiment`
- **Testable fix:** Use hierarchical sentence pooling so the overall stay assessment is not diluted by individual complaints in a long review.
- **Metric that should move:** Positive-class recall and `long_reviews` macro-F1 should increase.

### 20. test index 1256 — p(pos) = 0.9505, true = negative
- tokens after preprocessing: 351 · exclamations: 1 · negation present: yes

> What era is this? That was my first thought as I stepped into this restaurant. Mirrored ceilings, velvet upholstered chairs, cheetah print fabric -- it was like I was in the 80's or in the movie \""Scarface.\"" The place was pretty quiet and the atmosphere would be good for maybe a birthday or a graduation dinner. I ca…

- **Error type:** `length truncation`
- **Testable fix:** Raise `max_len` to 512 or use chunked hierarchical encoding so the closing negative verdict is retained.
- **Metric that should move:** `long_reviews` macro-F1 and negative-class recall should increase; confident false positives should decrease.

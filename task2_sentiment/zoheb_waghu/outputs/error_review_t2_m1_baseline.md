# Task 2 - Error review: `t2_m1_baseline`

Test accuracy 0.9331 · macro-F1 0.9331 · MCC 0.8666 · Brier 0.0504

Error type vocabulary: `negation` · `sarcasm/irony` · `mixed sentiment` · `aspect confusion` · `rating-text mismatch` · `domain term` · `length truncation` · `rare vocabulary / OOV` · `label noise` · `other`

## A. Confident false positives (predicted positive, actually negative)

### 1. test index 29330 — p(pos) = 1.0000, true = negative
- tokens after preprocessing: 19 · exclamations: 2 · negation present: no

> Wow love the place and everything is very clean and new! Great place to come and relax worth a try! Cheers, Eric Van Nguyen Visited April 2012

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

### 2. test index 3163 — p(pos) = 0.9995, true = negative
- tokens after preprocessing: 7 · exclamations: 0 · negation present: no

> What I love about Rubios' is that they always have beer. Always. That is all I love though...

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

### 3. test index 6186 — p(pos) = 0.9989, true = negative
- tokens after preprocessing: 48 · exclamations: 9 · negation present: yes

> Updated... they stopped serving Malibu Rum last year so it's no surprise they've closed and changed the theme of the place. MRB REJECT!! Prior Review: Ahhhh. Happy hour in the islands. Fruity drinks. Lots of Malibu Rum. Wait!! I'm in Scottsdale?? Darn!! Still a cozy little tiki bar with great happy hour specials!! Grea…

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

### 4. test index 25701 — p(pos) = 0.9988, true = negative
- tokens after preprocessing: 40 · exclamations: 0 · negation present: yes

> I had both a red tinga as well as a chili verde gordita. I also enjoyed a large cup of the best Horchata in Las Vegas. Fresh and warm, the gorditas were filling and good.....but not great. Good for a quick stop, but there are much better options when looking for a full flavored, spicy, authentic gordita. The service wa…

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

### 5. test index 25815 — p(pos) = 0.9983, true = negative
- tokens after preprocessing: 61 · exclamations: 0 · negation present: yes

> Attended the @SpaFitFinder launch party, with @spacephx. I can't say much about the place, other than, even taking into consideration the number of people present, it felt cramped. The TrimTini (?) I had was really delicious; vodkatini with lemon zest. A very nice gentleman, who works at Urban 7, was very gracious in m…

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

## B. Confident false negatives (predicted negative, actually positive)

### 6. test index 22807 — p(pos) = 0.0001, true = positive
- tokens after preprocessing: 48 · exclamations: 0 · negation present: yes

> EDIT: They really did change the service up since I last posted this. Horrible service. Used to be my favorite pizza in the city (at a reasonable price), but I'm rethinking that. We just had an altercation with a server who refused to split a check when we were paying with cash. He then proceeded to disrespect the part…

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

### 7. test index 30793 — p(pos) = 0.0002, true = positive
- tokens after preprocessing: 45 · exclamations: 0 · negation present: yes

> This place is so much better since they changed owners. My wife and I went when it was the old owners, it was terrible. We waited forever and the food never came before we walked out. People were served before us that walked in after and my wife actually got her soup before me and I sat and waited while they \""made mo…

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

### 8. test index 32999 — p(pos) = 0.0003, true = positive
- tokens after preprocessing: 14 · exclamations: 1 · negation present: no

> I won't say what spilled on my floor carpets, but their vacuums can REALLY suck! Thank goodness because I thought my carpet in my truck was ruined.

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

### 9. test index 11808 — p(pos) = 0.0004, true = positive
- tokens after preprocessing: 109 · exclamations: 6 · negation present: yes

> It was Anniversary time! But we didn't' want to spend a ton of money on food or booze. Plus, were weren't interested in going to a show at the time. So, what to do? Aquarium! Don't get me wrong: This place is TINY! Narrow walk ways and close quarters in general. Even the exhibits are small. I do hope that all of the an…

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

### 10. test index 30354 — p(pos) = 0.0005, true = positive
- tokens after preprocessing: 50 · exclamations: 0 · negation present: no

> Maize is good, but the hype has gotten out of hand. Their burritos are average - I mean, they are above-average for this area, but in general they are just okay. Kind of loosely wrapped, which is disappointing. Chips and salsa are good. Tacos seem fairly standard. Perhaps the issue is that there is so much mediocre Mex…

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

## C. Near-threshold errors (|p - 0.5| <= 0.05)

### 11. test index 21596 — p(pos) = 0.5003, true = negative
- tokens after preprocessing: 63 · exclamations: 2 · negation present: no

> What a let down! The interior of the restaurant looks dingy and our service was good but our waitress was bizarre. I had friends meet me there after being stood up. Didn't really like being asked about it. I don't think she was rude or anything but she was clearly just a kooky kind of girl who make sure everyone in the…

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

### 12. test index 37435 — p(pos) = 0.5008, true = negative
- tokens after preprocessing: 14 · exclamations: 0 · negation present: yes

> Was not able to eat here for lunch, they closed seating. Asked for next day an same thing happened despite perfect planning.

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

### 13. test index 35842 — p(pos) = 0.4984, true = positive
- tokens after preprocessing: 70 · exclamations: 0 · negation present: yes

> Please permit me to stress my experience has been at other multiple locations and not recently. My inclination to feel 5 stars is tempered by my personal short comings. Review is precipitated by how bad a completely different type of business is operated. Guitar Center is a highly complex business. It serves important …

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

### 14. test index 24192 — p(pos) = 0.5022, true = negative
- tokens after preprocessing: 88 · exclamations: 0 · negation present: yes

> Interesting place. Expansive menu. They apparently take pride in the ingredients in their food, which is a good thing. Just don't tell them that you're under any sort of time constraint, as they really don't seem to care, despite the fact that it was mentioned *several* times. I wouldn't recommend it for lunch during t…

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

### 15. test index 33283 — p(pos) = 0.5025, true = negative
- tokens after preprocessing: 209 · exclamations: 1 · negation present: yes

> I was really excited to finally go to Handlebar. I love beer, I love bicycles and I like bar food! We went tonight; a hot, humid summer monsoon evening. They keep all the doors and windows open but as nice as that sounds, that means that it will also be hot and humid inside. I grew up in Arizona, and run long distances…

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

## D. Worst-slice errors - slice `long_reviews` (macro-F1 0.9067 vs overall 0.9331)

### 16. test index 90 — p(pos) = 0.1268, true = positive
- tokens after preprocessing: 290 · exclamations: 0 · negation present: no

> Port Authority (formerly known as PATransit, or \""PAT\"") operates a fairly extensive network of buses and (in the South Hills) light rail. Instead of running school buses, the school district gave high schoolers bus passes to get to class, and I have used the bus since 1997. I've had a great experience using the bus …

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

### 17. test index 363 — p(pos) = 0.1737, true = positive
- tokens after preprocessing: 208 · exclamations: 0 · negation present: yes

> I don't much like the look of this place - I never would have ventured in were it not for the positive yelp reviews - but they serve some pretty good pizza. I have eaten trillions of pizzas. Maybe more, I lost count. And I'm not a a purist or a dogmatic devotee to any particular variety of pizza. I grew up in Chicago, …

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

### 18. test index 439 — p(pos) = 0.1154, true = positive
- tokens after preprocessing: 213 · exclamations: 1 · negation present: yes

> Thoroughly impressed with this airport. Not that I'm some great world traveler, but all the more reason. See, when I booked my virgin transatlantic flight (yes, virgin with a lower case v--as in, I was popping my passport cherry; not capital V, as in going on the airline), I needed an airport that wouldn't drive me bat…

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

### 19. test index 770 — p(pos) = 0.1890, true = positive
- tokens after preprocessing: 243 · exclamations: 1 · negation present: yes

> I booked a stay at Hotel San Carlos for one night through Groupon for $76 after taxes. Most other hotels in the area go for $120-300, so finding this deal was awesome. My boyfriend and I live in AZ and he wanted to celebrate his birthday by having a staycation downtown so we could have fun with family and friends witho…

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

### 20. test index 861 — p(pos) = 0.4913, true = positive
- tokens after preprocessing: 248 · exclamations: 1 · negation present: yes

> Summer time is supposed to be \""healthy time\"", so I am trying to watch what I eat. Now, I will always be a big guy (not \""two fat twins on their matching motorcycles\"" big, but you know, not little) but I definitely am eating more appropriately. However, I work in an office, and around 11:30, people start to doubl…

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

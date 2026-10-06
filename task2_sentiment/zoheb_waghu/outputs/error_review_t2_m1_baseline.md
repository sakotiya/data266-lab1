# Task 2 - Error review: `t2_m1_baseline`

Test accuracy 0.9535 · macro-F1 0.9535 · MCC 0.9070 · Brier 0.0350

Error type vocabulary: `negation` · `sarcasm/irony` · `mixed sentiment` · `aspect confusion` · `rating-text mismatch` · `domain term` · `length truncation` · `rare vocabulary / OOV` · `label noise` · `other`

## A. Confident false positives (predicted positive, actually negative)

### 1. test index 29330 — p(pos) = 1.0000, true = negative
- tokens after preprocessing: 19 · exclamations: 2 · negation present: no

> Wow love the place and everything is very clean and new! Great place to come and relax worth a try! Cheers, Eric Van Nguyen Visited April 2012

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

### 2. test index 5185 — p(pos) = 0.9998, true = negative
- tokens after preprocessing: 37 · exclamations: 0 · negation present: yes

> Like Clay P who posted before me, I too love pancakes. Though I love chocolate chip pancakes. But like Clay I did not love the ones that I got a the Original Pancake House. Typically, restaurants just don't do it the way I like them and I have come to expect that. Grading OPH on those terms, they did a respectable job.…

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

### 3. test index 14825 — p(pos) = 0.9995, true = negative
- tokens after preprocessing: 141 · exclamations: 1 · negation present: yes

> Do you believe in Yin and Yang? The ancient Asian philosophy suggesting polar opposites are interrelated? If you don't, then you best start believing in it if you're headed to Flo's. The manifestation of Yin and Yang is clearly present...like 'in your face' present when you're here. You'll see it from the moment you're…

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

### 4. test index 2818 — p(pos) = 0.9992, true = negative
- tokens after preprocessing: 7 · exclamations: 0 · negation present: no

> House Margs are good cheap and big. Just how I like my men.

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

### 5. test index 27375 — p(pos) = 0.9992, true = negative
- tokens after preprocessing: 47 · exclamations: 0 · negation present: no

> We had dinner here for four. Had an artichoke appetizer with pita chips, club sandwich and shared a huge sundae for dessert. The service was awesome; space impressive. But, I have to say that when a gem from NYC gets transported to Vegas, and gets \""blown up\"" into Vegas standards with all the glam, size, \""in your …

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

### 8. test index 29494 — p(pos) = 0.0003, true = positive
- tokens after preprocessing: 169 · exclamations: 0 · negation present: yes

> UPDATED. My initial very frustrated and dramatic review read as follows: Bililng practices are at best negligent and at worst fraudulent. I started going to the studio per a groupon. I enjoyed the experience so much that I purchased a discounted 20 pack of classes. When I purchased the classes, my credit card was charg…

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

### 9. test index 20061 — p(pos) = 0.0008, true = positive
- tokens after preprocessing: 311 · exclamations: 1 · negation present: yes

> Ever wonder what to do if you have lots of extra garbage or recyclables and either can't fit them all in your bins or missed bulk trash pickup day? Alternatively, are you looking for something just a little bit different to do on a lazy summer day? Ok, ok, all kidding aside - you may find yourselves (as we did yesterda…

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

### 10. test index 264 — p(pos) = 0.0008, true = positive
- tokens after preprocessing: 23 · exclamations: 2 · negation present: yes

> Says they deliver on here... Wrong & wrong again & should not be checked! I like Applebee's & thought the delivery was something new for the Pittsburgh area... Don't know if this is something Yelp does or something someone checked off... But not cool!

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

## C. Near-threshold errors (|p - 0.5| <= 0.05)

### 11. test index 1604 — p(pos) = 0.4998, true = positive
- tokens after preprocessing: 96 · exclamations: 0 · negation present: yes

> I have to like this place.. this is where my engagement ring was bought.. well, it was put together.. because the story Im told by my very creative fiance, is that he wanted a particular princess cut diamond with high clarity..+VS or above... and a particular setting with channel diamonds.. My ring was given in the dar…

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

### 12. test index 7137 — p(pos) = 0.5009, true = negative
- tokens after preprocessing: 26 · exclamations: 1 · negation present: yes

> Whoa...... International??? This tiny, efficient airport does not have much more to offer than that. No restaurants, a few cheap shops. I guess they know all the action is on the outside, so why try??? :) But, it does have slot machines in the middle of the airport - now there's something you don't see every day!

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

### 13. test index 24722 — p(pos) = 0.5011, true = negative
- tokens after preprocessing: 77 · exclamations: 0 · negation present: no

> We went to the 1030 showing, it was pretty good but we expected a lot more out of the Garth and the Wynn for a $143 a ticket. He may have just been tired because it was his second show that night but they could have just had some guy off the street come in and play guitar and sing for two hours and it would have been a…

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

### 14. test index 29611 — p(pos) = 0.5017, true = negative
- tokens after preprocessing: 36 · exclamations: 0 · negation present: yes

> The food here was good, but the prices are outrageous. I am by no means stingy, and I was expecting to pay a little more for a gourmet pizza... But $17 for a 12\"" diameter pizza with a couple pieces of prosciutto on top? Never again. It's a shame, because I did like the pizza and it would be nice to have a good Italia…

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

### 15. test index 439 — p(pos) = 0.4982, true = positive
- tokens after preprocessing: 213 · exclamations: 1 · negation present: yes

> Thoroughly impressed with this airport. Not that I'm some great world traveler, but all the more reason. See, when I booked my virgin transatlantic flight (yes, virgin with a lower case v--as in, I was popping my passport cherry; not capital V, as in going on the airline), I needed an airport that wouldn't drive me bat…

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

## D. Worst-slice errors - slice `long_reviews` (macro-F1 0.9408 vs overall 0.9535)

### 16. test index 439 — p(pos) = 0.4982, true = positive
- tokens after preprocessing: 213 · exclamations: 1 · negation present: yes

> Thoroughly impressed with this airport. Not that I'm some great world traveler, but all the more reason. See, when I booked my virgin transatlantic flight (yes, virgin with a lower case v--as in, I was popping my passport cherry; not capital V, as in going on the airline), I needed an airport that wouldn't drive me bat…

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

### 17. test index 1256 — p(pos) = 0.6371, true = negative
- tokens after preprocessing: 351 · exclamations: 1 · negation present: yes

> What era is this? That was my first thought as I stepped into this restaurant. Mirrored ceilings, velvet upholstered chairs, cheetah print fabric -- it was like I was in the 80's or in the movie \""Scarface.\"" The place was pretty quiet and the atmosphere would be good for maybe a birthday or a graduation dinner. I ca…

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

### 18. test index 2673 — p(pos) = 0.6890, true = negative
- tokens after preprocessing: 473 · exclamations: 6 · negation present: yes

> 12/03/12 Hmm... I could've sworn I had written a review of this place before. At the very least, I uploaded a few pics of our breakfast here many months ago. Took my Mom to try the oven-baked pancakes once. Do yourself a favor and pass on these! Guess the review AND photos were taken down. (Let's see how long this NEW …

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

### 19. test index 3215 — p(pos) = 0.6739, true = negative
- tokens after preprocessing: 336 · exclamations: 9 · negation present: yes

> We went out with the Central AZ Jeepers the other night to do a night run of the Maggie Mine Trail. The plan was to start up at Bumble Bee, come south to Black Canyon City, then finish at Rock Springs Cafe for dinner and pie. First I have to mention we had a party of over 20 Jeeps and the trail is mostly visible from 1…

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

### 20. test index 3755 — p(pos) = 0.4907, true = positive
- tokens after preprocessing: 246 · exclamations: 3 · negation present: yes

> I've been getting used (it takes 21 days for it to become a habit? the experts say) to my new diet now (low-sodium), and have been going to Trader Joe's for the past six or seven straight weekends. While I can't find everything on my list, I can find nearly 90% of what I need for the upcoming week: no salt tortilla chi…

- **Error type:** _<assign one>_
- **Testable fix:** _<one change you could make>_
- **Metric that should move:** _<which number, in which direction>_

# Error review - exp_bilstm (task2_shreya_main_20260921_222011)

## confident_false_positive

**row 29330** true=0 pred=1 p(pos)=0.9993 tokens=18 negation=False

> Wow love the place and everything is very clean and new!\n\nGreat place to come and relax worth a try!\n\nCheers,\n\nEric Van Nguyen\nVisited April 2012

- error type: 
- testable fix: 

**row 408** true=0 pred=1 p(pos)=0.9993 tokens=57 negation=True

> Though I'm a Copper enthusiast when it comes to getting my Indian fix in Charlotte, I'd heard that Maharani was a cheaper but tasty option, so we ordered from there a few nights ago. \n\nCopper is definitely still my place, but Maharani was fine enough. First of all, the food came very quickly, which is rare. Usually indian food, good indian good, takes at least 30-45 minutes. We got our order in like 30, which was great 'cause we were starving. \n\nThe Tikka Masala was spicy and pretty good, but it wasn't as thick and saucy as i like. The salad was just so-so. The Naan were all amazing. Defin

- error type: 
- testable fix: 

**row 5752** true=0 pred=1 p(pos)=0.999 tokens=220 negation=True

> NOTE:  This was a 4-star review, but the food quality and ESPECIALLY customer service have gone down the tubes.  See update below.\nComplaining I can't find a good meatball sub in Phoenix, I was referred to Santisi Brothers.  I was told that the meatballs are still made by the brothers' mother, so I was intrigued. \nSantisi Brothers did not disappoint.  That sub was so freaking good!!!  I got the 1/2 sandwich, which included two meatballs smothered under mozzarella and marinara on a toasted baguette-type roll.  Amazing.  The bread was soft inside with just the right crust, the meatballs were s

- error type: 
- testable fix: 

**row 14825** true=0 pred=1 p(pos)=0.9986 tokens=132 negation=True

> Do you believe in Yin and Yang? The ancient Asian philosophy suggesting polar opposites are interrelated? \nIf you don't, then you best start believing in it if you're headed to Flo's.\n\nThe manifestation of Yin and Yang is clearly present...like 'in your face' present when you're here. You'll see it from the moment you're seated, but you need to be observant...in tune...at one with the universe.\nLook around.\nNotice anything?\n\nLike the couple who was seated five minutes after you were. See how they're now eating something and enjoying their experience?\nAnd what's the opposite manifestati

- error type: 
- testable fix: 

**row 5185** true=0 pred=1 p(pos)=0.9983 tokens=39 negation=True

> Like Clay P who posted before me, I too love pancakes. Though I love chocolate chip pancakes. But like Clay I did not love the ones that I got a the Original Pancake House. Typically, restaurants just don't do it the way I like them and I have come to expect that. Grading OPH on those terms, they did a respectable job. It is definitely a worthwhile destination for a pancake lover.

- error type: 
- testable fix: 

## confident_false_negative

**row 22807** true=1 pred=0 p(pos)=0.0001 tokens=46 negation=True

> EDIT: They really did change the service up since I last posted this.\n\nHorrible service.\n\nUsed to be my favorite pizza in the city (at a reasonable price), but I'm rethinking that. We just had an altercation with a server who refused to split a check when we were paying with cash. He then proceeded to disrespect the party at the table, telling us to 'not give him attitude about it.'\n\nSorry Bella Notte, but we're not children. I don't care if you're working hard - it doesn't give you any excuse to disrespect your paying customers like that.

- error type: 
- testable fix: 

**row 30793** true=1 pred=0 p(pos)=0.0001 tokens=38 negation=True

> This place is so much better since they changed owners.\n\nMy wife and I went when it was the old owners, it was terrible.  We waited forever and the food never came before we walked out.  People were served before us that walked in after and my wife actually got her soup before me and I sat and waited while they \""made more\"".\n\nIt was horrible.\n\nNow its much better.  The staff are very friendly, they treat their customers very well and I have nothing but positive things to now say about this place.  Its much better with the new owners.

- error type: 
- testable fix: 

**row 22451** true=1 pred=0 p(pos)=0.0008 tokens=111 negation=True

> The food is crap.  I'm not trying to be mean, but it really is horrible.  I'd rather eat one of those Tornado things from Circle K for dinner.  Also, I don't appreciate the waitress telling me everything is great when everything is absolutely not great.  What kind of disgusting excuse for food must she live off of if the nachos get her stamp of approval?  They were probably the worst nachos I've ever had.  Taco Bell's nachos are like manna from heaven compared to the sad mess Barney's serves.  She should have just been honest and told me to order fries a la carte, because that was the only thi

- error type: 
- testable fix: 

**row 20061** true=1 pred=0 p(pos)=0.0011 tokens=278 negation=True

> Ever wonder what to do if you have lots of extra garbage or recyclables and either can't fit them all in your bins or missed bulk trash pickup day? Alternatively, are you looking for something just a little bit different to do on a lazy summer day? \n\nOk, ok, all kidding aside - you may find yourselves (as we did yesterday) with more broken-down boxes and odd-shaped trash (old floor lamps, etc.) than can fit in your bins. This waste facility allows Phoenix residents to dump bulk trash and recyclables for free once a month. \n\nTo get to the facility, drive down to 27th Avenue and Buckeye (aka

- error type: 
- testable fix: 

**row 16700** true=1 pred=0 p(pos)=0.0011 tokens=45 negation=True

> TERRIBLE SERVICE, RUDE WAITERS WITH A PISS POOR ATTITUDE! WOULD EAT HERE AGAIN! A++++\n\nThe food here is good but isn't that spectacular, the beer selection is somewhat disappointing if you like a good craft beer. Ask for a craft beer and get an insulting comment from the waitress. \n\nWhat makes this place awesome is the atmosphere, live bands, sports on the TVs, the very obnoxious staff and the hats. Got to love the free hats. \n\nIf you go, ask for a glass of water with your meal. ;)

- error type: 
- testable fix: 

## near_threshold

**row 33696** true=0 pred=1 p(pos)=0.5001 tokens=66 negation=False

> It gets the job done.  What do you want, it's a Sbarro's.  \n\nIf you want something that's different than the traditional pizza, try a stromboli.  Unlike pizza, it's .... okay, it's basically tubular pizza.  But it's good!\n\nMy only beef with Sbarro's is really a beef with the food court it's part of:  The iced tea they serve at the fountain is UNDRINKABLE.  I'm taking off a star for it.  I actually had to pour my tea out and drink water, it was so vile.  There were two fountains, and I tried the tea in both places, and it was equally horrible out of both fountains. \n\nIn an era in which ev

- error type: 
- testable fix: 

**row 21959** true=0 pred=1 p(pos)=0.5001 tokens=42 negation=True

> Well i hate to be the bearer of bad news...but these doughnuts are average at best. My co-workers bring some in to work about twice a week and each time they taste the same to me. The doughnuts have to much dough so to speak...and quite frankly they are really to big to enjoy as a doughnut. The last time i checked doughnuts should be a joy to eat...not a chore. I think i will stick to my Krispy Cremes if you don't mind my fellow yelpers. Eat, Drink, and be Merry my friends!!!!

- error type: 
- testable fix: 

**row 13123** true=0 pred=1 p(pos)=0.5005 tokens=25 negation=True

> I must have been there on a bad night (Sunday) because there were not actually any people there. Even the free bottle of vodka did not help the situation. It did have a good view of the strip, but that only entertained me for about 3 minutes. Drank our vodka and then moved on to Lavo...THAT is where the people were.

- error type: 
- testable fix: 

**row 37350** true=1 pred=0 p(pos)=0.499 tokens=149 negation=True

> This is the new occupant of the space formerly occupied by Ray's something or other blah blah ramen. I never made it out to that place, but it must not have been that good if it closed shop.\n\nMy first visit was a few nights ago. I went with the beef sukiyaki plate for $7.49 ($5.99 bowl cost). The clerk claimed that their plates have a higher portion size than the bowls, but I'm not buying it after seeing my friend's beef sukiyaki bowl. The beef was pretty good, and so was the noodle... but since it comes mixed with noodles and then a whole bunch of rice, I really felt meat-deprived. I did li

- error type: 
- testable fix: 

**row 19447** true=1 pred=0 p(pos)=0.4984 tokens=21 negation=True

> We orders crepes and cheese fondue with sundry tomato. We are so full after all. The crepes are nice. I don't like the taste of the cheese fondue (probably the taste of wine or sundry tomato). I like that they have so many choices on crepes.

- error type: 
- testable fix: 

## slice_negation

**row 11808** true=1 pred=0 p(pos)=0.0011 tokens=99 negation=True

> It was Anniversary time! But we didn't' want to spend a ton of money on food or booze. Plus, were weren't interested in going to a show at the time. So, what to do?\n\nAquarium!\n\nDon't get me wrong: This place is TINY! Narrow walk ways and close quarters in general. Even the exhibits are small. I do hope that all of the animals in there are ok. They do seem treated well but I wonder if they would be happier with larger homes. i think the most open area is at the very end, the shark tank.\n\nSpeaking of said tank, it was TOTALLY the highlight of this place. They built it so it looks like a su

- error type: 
- testable fix: 

**row 24111** true=1 pred=0 p(pos)=0.0013 tokens=46 negation=True

> This review is for the pharmacy only.You do not need to be a member to use the pharmacy. I am a self paying user of generic Plavix...Cost for a 90 day supply is around is about $25. Contrast this to $45 at Walmart,and $200+ at both Target and Shopko. Price difference is unbelievable!! Begs the question of what price are health insurance carriers forced to pay!?? No wonder premiums are high should they be charged the 200-220 prices. It does pay to shop for prices..

- error type: 
- testable fix: 

**row 14906** true=1 pred=0 p(pos)=0.0013 tokens=61 negation=True

> So, after getting hosed on my room rate last year for SuperBowl, I looked around different websites this time.  Never use i4vegas. Ever. 321 days later, I still haven't received my refund.\n\nAnywho, hotels.com was advertising 2 nights for $161. SouthPointe was advertising $212. WTH? Also an issue left over from last year was the fact that since they didn't sell out (and won't this year either), the price dropped in half the week of SuperBowl, but I couldn't take advantage of it.\n\nSo I picked up the phone. SouthPointe matched the lower price and guaranteed that if the price goes down, I'll g

- error type: 
- testable fix: 

**row 28976** true=1 pred=0 p(pos)=0.0017 tokens=327 negation=True

> gasp. 1/2 star docked. like any new restaurant, i guess i should have expected fluctuations in all aspects of the restaurant, food and operation wise. \n\nmy friend and i have made it our goal to come here once a week on a weekend night for half off burgers. so here's the update on business hours of operation: they still offer half off for the same menu items (including the same exclusions), but they don't let you modify any of the standard burgers by adding anything (like lettuce or tomato for example) - they only let you take off toppings.  the half off still takes place during the last hour

- error type: 
- testable fix: 

**row 26683** true=1 pred=0 p(pos)=0.002 tokens=16 negation=True

> I've just been forced to concede that, despite still not digging their ordering process, their food is just too good to disrespect with a 2 star review.

- error type: 
- testable fix: 

"""Generates a SMALL SYNTHETIC demo dataset (data/dataset.csv).

It only exists so the project runs out of the box. Replace data/dataset.csv
with your real dataset (columns: headline,label ; label -1 or 1 = toxic,
0 = non-toxic) and run `python -m ml.train`.
"""
import csv
import random
from pathlib import Path

random.seed(7)

INSULTS = ["idiot", "moron", "loser", "stupid", "dumb", "pathetic", "worthless", "ugly",
           "jerk", "trash", "garbage", "clown", "freak", "creep", "coward", "disgusting",
           "useless", "bastard", "asshole", "bitch", "scumbag", "imbecile", "failure"]
TOXIC = [
    "you are such a {i}", "you are a {i} and a {j}", "shut up you {i}", "nobody likes you, {i}",
    "go away {i}, everyone hates you", "what a {i}, you are worthless", "i hate you so much you {i}",
    "you are so {i} it hurts to look at you", "only a {i} would say that", "get lost you {i}",
    "{i} {j} stop posting your garbage here", "you disgust me, {i}", "everybody laughs at you {i}",
    "you should be ashamed, you {i}", "what is wrong with you, {i}", "stop talking {i}, nobody cares",
    "you are the biggest {i} i have ever seen", "i will find you and make you regret it {i}",
    "you deserve to suffer, {i}", "go kill yourself you {i}", "wow you really are a {i}",
    "{i}!!! nobody wants you here", "u r a {i} lol", "delete your account you {i}",
    "you fat {i}, no wonder nobody loves you", "your whole family is {i}", "shut your mouth {i} {j}",
    "this is why everyone thinks you are a {i}", "you ugly {i}, never show your face again",
    "keep crying {i}, you are a total {j}",
]
TOPICS = ["the history article", "my homework", "the football match", "this recipe", "the new update",
          "the school project", "your photo", "the concert", "this chapter", "the math problem",
          "the group presentation", "the movie", "this song", "the garden", "the science fair"]
NAMES = ["Sam", "Priya", "Alex", "Jordan", "Maria", "Chen", "Ravi", "Fatima", "Tom", "Aisha"]
SAFE = [
    "thanks for helping me with {t}", "i think {t} needs more sources", "great job on {t}",
    "could you explain {t} again please", "{t} was fascinating, i learned a lot",
    "i disagree with your point on {t} but i respect your view", "happy birthday {n}, have a wonderful day",
    "you are so beautiful", "you are amazing and very kind", "you are such a good friend",
    "congratulations {n}, you totally deserve it", "i really enjoyed {t}", "see you at the meeting tomorrow {n}",
    "can anyone recommend something similar to {t}", "well done {n}, keep it up", "this made my day, thank you",
    "i am sorry about earlier, let us talk about {t}", "that exam was a killer but i passed",
    "this workout is killing me, so tired", "the movie was so stupid funny, loved it",
    "please be kind to each other in the comments", "i love my dog, he is the best",
    "what time does {t} start", "your idea about {t} is really clever", "thank you {n}, that was very helpful",
    "nice work, i will review {t} tonight", "have a great weekend everyone", "i am proud of you {n}",
    "the weather is lovely today", "let's study together for {t}", "could someone please fix {t}",
    "no worries {n}, mistakes happen and we all learn", "this discussion is getting heated, let us take a break",
]


def noise(s: str) -> str:
    r = random.random()
    if r < 0.12:
        s = s.upper()
    elif r < 0.25:
        s = s.capitalize()
    if random.random() < 0.12:
        s = f"@user{random.randint(1, 999)} " + s
    if random.random() < 0.15:
        s += random.choice(["!", "!!", "...", " lol", "?"])
    return s


toxic, safe = set(), set()
while len(toxic) < 1500:
    i, j = random.sample(INSULTS, 2)
    toxic.add(noise(random.choice(TOXIC).format(i=i, j=j)))
while len(safe) < 1500:
    safe.add(noise(random.choice(SAFE).format(t=random.choice(TOPICS), n=random.choice(NAMES))))

rows = [(t, -1) for t in toxic] + [(t, 0) for t in safe]  # original data used -1 for toxic
random.shuffle(rows)
out = Path(__file__).parent / "dataset.csv"
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["headline", "label"])
    w.writerows(rows)
print(f"wrote {len(rows)} rows -> {out}")

import numpy as np
from collections import Counter
import sys, os

d = 1024
window = 10      # how many recent words the model can see (6 -> 10 fixed most forgetting)
END = "<end>"
TIE = "<tie>"
USER = "<user>"
BOT = "<bot>"
OOV = "<oov>"
MAX_TOKENS = 50
K = 20           # how many of the closest memories get to vote
T = 8            # how quickly a memory's vote fades as it gets less similar
SAMPLE = 0.3     # 1 = pick by raw odds (very random), 0.2 = almost always the top word
DATASET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_gk.txt")

with open(DATASET_FILE, encoding="utf-8") as f:
    dataset = [line.strip() for line in f if line.strip()]

rng = np.random.default_rng()
translations = {}   # word -> packed 1-bit vector
contexts = {}       # bundled context (bytes) -> Counter of next words

def bundle(words):
    n = len(words)
    total = sum(np.roll(np.unpackbits(translations[w]).astype(np.int8) * 2 - 1, n - 1 - i) for i, w in enumerate(words))
    if n % 2 == 0:
        total = total + np.unpackbits(translations[TIE]).astype(np.int8) * 2 - 1
    return np.packbits(total > 0)

def init(sentences):
    for word in [OOV, USER, BOT, END, TIE]:
        if word not in translations:
            translations[word] = np.packbits(rng.integers(0, 2, size=d))
    for index, sentence in enumerate(sentences):
        if index % 1000 == 0: print(f"assigning sentence: {index}/{len(sentences)}")
        for word in sentence.split():
            if word not in translations:
                translations[word] = np.packbits(rng.integers(0, 2, size=d))

def train(sentences):
    for index, line in enumerate(sentences):
        sentence = line.split() + [END]
        if index % 1000 == 0: print(f"training sentence: {index}/{len(sentences)}")
        for current_idx, current_word in enumerate(sentence):
            if BOT not in sentence[:current_idx]: continue
            start = max(0, current_idx - window)

            for j in range(start, current_idx):
                context = tuple(sentence[j:current_idx])
                contexts.setdefault(bundle(context).tobytes(), Counter())[current_word] += 1

def build():
    global keys, vals, totals
    keys = np.frombuffer(b"".join(contexts), dtype=np.uint8).reshape(-1, d // 8)
    vals = list(contexts.values())
    totals = [sum(v.values()) for v in vals]

def inference(prompt):
    words = [USER] + [w for w in prompt.split() if w in translations] + [BOT]   # unknown words are dropped
    output = []

    for _ in range(MAX_TOKENS):
        vec = bundle(words[-window:])

        diff = np.bitwise_count(keys ^ vec).sum(axis=1).astype(np.int32)
        k = min(K, len(diff))
        nearest = np.argpartition(diff, k - 1)[:k]
        closest = int(diff[nearest].min())

        votes = Counter()
        for i in nearest:
            weight = np.exp(-(int(diff[i]) - closest) / T)
            for word, count in vals[i].items():
                votes[word] += weight * count / totals[i]   # each memory has one vote, split by proportion

        candidates = list(votes)
        p = np.array([votes[w] for w in candidates]) ** (1 / SAMPLE)
        p /= p.sum()
        best_word = candidates[rng.choice(len(candidates), p=p)]
        best_score = 1 - 2 * closest / d

        print(best_word, round(best_score, 3))
        if best_word == END:
            break

        output.append(best_word)
        words.append(best_word)

    return output

def learn(text): # format: /learn <question> > <answer> and converts to <user> <question> <bot> <answer>
    if ">" not in text:
        print("use: /learn question > answer")
        return
    question, answer = text.removeprefix("/learn").lower().split(">", 1)
    line = f"{USER} {question.strip()} {BOT} {answer.strip()}"
    init([line])
    train([line])
    build()
    print(f"learnt: {line}")

init(dataset)
train(dataset)
build()
print(f"ready: {len(contexts)} memories")
print(f"memory of translations: {sys.getsizeof(translations) / 1048576:.2f} MB\nmemory of contexts: {sys.getsizeof(contexts) / 1048576:.2f} MB")

while True:
    prompt = input("\nprompt: ")
    if prompt.startswith("/learn"):
        learn(prompt)
        continue
    print(" ".join(inference(prompt)))
import os, re, time
import numpy as np

d = 256
w = 3
k = 3
END = "<end>"
DATASET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.txt")
rng = np.random.default_rng()
t1 = {}
context_memory = {}
followers = {}
vocab = []
V = None

with open(DATASET_FILE, encoding="utf-8") as f:
    dataset = [line.strip() for line in f if line.strip()]

def tokenize(s):
    return re.findall(r"[a-z'-]+", s.lower())

def build_index():
    global vocab, V
    vocab = list(t1)
    V = np.stack([t1[x] for x in vocab]).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)

def initiate(sentences=None):
    for sentence in sentences or dataset:
        for word in tokenize(sentence) + [END]:
            if word not in t1:
                t1[word] = rng.choice(np.array([-1, 1], dtype=np.int8), size=d)
    build_index()

def train(sentences=None):
    for sentence in sentences or dataset:
        words = tokenize(sentence) + [END]
        for i in range(1, len(words)):
            target = words[i]
            for order in range(1, min(w, i) + 1):
                context = tuple(words[i - order:i])
                followers.setdefault(context, set()).add(target)
                previous = context_memory.get(context)
                if previous is None:
                    context_memory[context] = t1[target].astype(np.int32).copy()
                else:
                    context_memory[context] = previous + t1[target].astype(np.int32)

def binarize():
    word_id = {wd: i for i, wd in enumerate(vocab)}
    for context, v in context_memory.items():
        if len(followers[context]) == 1:
            context_memory[context] = word_id[next(iter(followers[context]))]
        else:
            context_memory[context] = np.packbits(v > 0)

def rank(words, n=10):
    total = np.zeros(len(vocab), dtype=np.float32)
    found = False
    for order in range(1, min(w, len(words)) + 1):
        value = context_memory.get(tuple(words[-order:]))
        if value is None:
            continue
        found = True
        if isinstance(value, int):
            scores = np.zeros(len(vocab), dtype=np.float32)
            scores[value] = 1.0
        else:
            bits = np.unpackbits(value).astype(np.float32) * 2 - 1
            scores = V @ (bits / np.sqrt(d))
        total += order * scores
    if not found:
        return []
    top = np.argsort(-total)[:n]
    return [(vocab[i], float(total[i])) for i in top]

def sims(prompt, n=100):
    start = time.time()
    results = {}
    for i in range(n):
        if i % 10 == 0: print(f"epoch: {i}/{n} | results: {results} | elapsed: {time.time() - start:.2f}s\n")
        t1.clear()
        context_memory.clear()
        followers.clear()
        initiate()
        train()
        binarize()
        result = rank(tokenize(prompt))[0][0]
        if result not in results.keys(): results[result] = 1
        else: results[result] += 1
    return results

def sequence(prompt, max_tokens=50):
    for _ in range(max_tokens):
        ranked = rank(tokenize(prompt))
        if not ranked:
            break
        word = ranked[0][0]
        if word == END:
            break
        prompt += f" {word}"
    return prompt

if __name__ == "__main__":
    start = time.time()
    initiate()
    train()
    binarize()
    print(f"train time:        {time.time() - start:.2f}s")
    for i in rng.choice(len(dataset), size=min(5, len(dataset)), replace=False):
        prompt = " ".join(tokenize(dataset[i])[:2])
        print(f"{prompt!r:30} -> {sequence(prompt)}")
    print(f"{'the golden'!r:30} -> {sequence('the golden')}")
    print(f"words in vocab:    {len(t1)}")
    print(f"stored contexts:   {len(context_memory)}")
    single = sum(isinstance(v, int) for v in context_memory.values())
    multi = len(context_memory) - single
    print(f"single-follower:   {single} / {len(context_memory)}")
    print(f"memory for those:  {(single * 2 + multi * d / 8) / 1e6:.3f} MB")
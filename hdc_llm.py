import os
import re
import time
import numpy as np

DIM = 125000
w = 3
k = 3
END = "<end>"
DATASET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.txt")

t1 = {}
t2 = {}
vocab = []
V = None

with open(DATASET_FILE, encoding="utf-8") as f:
    dataset = [line.strip() for line in f if line.strip()]

class HDC:
    def __init__(self, dim=10000, seed=None):
        self.dim = dim
        self.rng = np.random.default_rng(seed)

    def vector(self):
        return self.rng.choice(np.array([-1, 1], dtype=np.int8), size=self.dim)

    def bind(self, *args):
        if not args:
            raise ValueError("bind() needs at least one hypervector")
        result = np.array(args[0], dtype=np.int8)
        for hv in args[1:]:
            result = result * hv
        return result

    def bundle(self, *args):
        if not args:
            raise ValueError("bundle() needs at least one hypervector")
        total = np.sum(np.stack(args).astype(np.int32), axis=0)
        result = np.sign(total).astype(np.int8)
        ties = result == 0
        if ties.any():
            result[ties] = self.rng.choice(np.array([-1, 1], dtype=np.int8), size=int(ties.sum()))
        return result

    def cos_sim(self, a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

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
                t1[word] = hdc.vector()
    build_index()

def train(sentences=None):
    for sentence in sentences or dataset:
        words = tokenize(sentence) + [END]
        for i in range(1, len(words)):
            for d in range(1, w + 1):
                if i - d < 0:
                    break
                key = (d, words[i - d])
                t2[key] = t2.get(key, 0) + t1[words[i]].astype(np.int32)

def rank(words, n=10):
    total = np.zeros(hdc.dim)
    for d in range(1, w + 1):
        if d > len(words):
            break
        v = t2.get((d, words[-d]))
        if v is not None:
            total += v / np.linalg.norm(v) / d
    norm = np.linalg.norm(total)
    if norm == 0:
        return []
    sims = V @ (total / norm).astype(np.float32)
    top = np.argsort(-sims)[:n]
    return [(vocab[i], float(sims[i])) for i in top]

def inference(prompt):
    return rank(tokenize(prompt))

def generate(prompt, n=20, top_k=5, temp=0.1):
    words = tokenize(prompt)
    for _ in range(n):
        ranked = rank(words, top_k)
        if not ranked:
            break
        names, sims = zip(*ranked)
        p = np.exp((np.array(sims) - max(sims)) / temp)
        p /= p.sum()
        nxt = names[np.random.choice(len(names), p=p)]
        if nxt == END:
            break
        words.append(nxt)
    return " ".join(words)

def fresh(train_set):
    global hdc
    t1.clear()
    t2.clear()
    hdc = HDC(DIM)
    initiate(train_set)
    train(train_set)

def memory():
    import pickle
    t1_file_size_kb = len(pickle.dumps(t1)) / 1024
    t2_file_size_kb = len(pickle.dumps(t2)) / 1024
    unique = len(t1.keys())

    print(f"t1 serialized size: {t1_file_size_kb:.2f} KB")
    print(f"t2 serialized size: {t2_file_size_kb:.2f} KB")
    print(f"unique words: {unique}")
    if unique > 0:
        print(f"KB/word with window as {w}: {t2_file_size_kb / unique:.2f} KB")

def simulations(prompt, epochs, window_range):
    global hdc, w
    final_scores = {w_size: {} for w_size in range(1, window_range + 1)}

    for i in range(epochs):
        for w in range(1, window_range + 1):
            fresh(None)
            result = inference(prompt)[0][0]

            final_scores[w][result] = final_scores[w].get(result, 0) + 1

    return final_scores

if __name__ == "__main__":
    start = time.time()
    hdc = HDC(DIM)
    initiate()
    train()
    print(f"{len(dataset)} sentences, {len(t1)} unique words, trained in {time.time() - start:.1f}s")
    print(inference("A fluffy ragdoll cat")[:k])
    for _ in range(3):
        print(generate("the old cat"))
    memory()

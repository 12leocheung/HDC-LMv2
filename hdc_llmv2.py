import os, re, time
import numpy as np

d = 128        # do NOT drop to 64 for a large corpus: at ~30k vocab the
               # 1/64 score lattice lets random impostors tie/beat the true
               # follower (max impostor score ~ 4.3/sqrt(64) ~ 0.54 >= 0.5),
               # and math answers started leaking into small talk. 128 halves
               # that to ~0.38 (measured via probe).
w = 6          # deep windows: with shared answer frames ('... the sun gives us
k = 3          # ...'), only a 6-token window still sees the question tag
END = "<end>"
DATASET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.txt")
rng = np.random.default_rng()
t1 = {}
context_memory = {}
followers = {}
counts = {}            # how many times each context was seen (used by export.py to prune)
vocab = []
V = None

with open(DATASET_FILE, encoding="utf-8") as f:
    dataset = [line.strip() for line in f if line.strip()]

def tokenize(s):
    # <user>, <bot> and <end> count as single tokens; words may contain digits
    # and arithmetic signs, which GLUE their operands into one token
    # ('9-4', '3*4', '7+4', '12/4') so every fact has a unique context window
    # and the reply can DISPLAY the symbol. Word operators are mapped to
    # symbols ONLY between two digits ('5 minus 8' -> '5-8'), never after a
    # word: otherwise answer text like 'is minus 1' would squeeze into one
    # token 'is-1'. After mapping, spaces around symbols are removed, so typed
    # input like '3 - 4' lands on the trained token too.
    s = s.lower()
    for phrase, sym in (("multiplied by", "*"), ("divided by", "/"),
                        ("plus", "+"), ("minus", "-"), ("add", "+"),
                        ("times", "x")):
        s = re.sub(rf"(?<=\d) {phrase} (?=\d)", f" {sym} ", s)
    s = re.sub(r"\s*([+*/-])\s*", r"\1", s)
    s = re.sub(r"(?<=\d)\s+x\s+(?=\d)", "x", s)
    return re.findall(r"<[a-z]+>|[a-z0-9'\-*+/]+", s)

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
                counts[context] = counts.get(context, 0) + 1
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
    # kind='stable' ties resolve to the LOWEST vocab id, exactly like the
    # board's first-index-wins argmax scan (np default quicksort is not
    # stable, which once handed a tie to a random arithmetic token)
    top = np.argsort(-total, kind="stable")[:n]
    return [(vocab[i], float(total[i])) for i in top]

def sims(prompt, n=100):
    start = time.time()
    results = {}
    for i in range(n):
        if i % 10 == 0: print(f"epoch: {i}/{n} | results: {results} | elapsed: {time.time() - start:.2f}s\n")
        t1.clear()
        context_memory.clear()
        followers.clear()          # <-- the fix: clear this too, or the sets pile up
        counts.clear()
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

FALLBACKS = (
    "i am not sure about that try asking me about numbers animals or space",
    "that one is beyond me for now ask me something simple",
    "hmm i do not know that one yet teach me and i will remember it",
    "i am still learning my favorite topics are math animals and space",
)
fallback_idx = 0


def reply(text, max_tokens=40):
    """Answer one message: feed '<user> ...your words... <bot>' and keep predicting."""
    global fallback_idx
    words = ["<user>"] + [t for t in tokenize(text) if t not in ("<user>", "<bot>", END)] + ["<bot>"]
    # when nothing matched (rank over the generic <bot> average only), say a
    # rotating canned line instead of letting the deterministic argmax collapse
    # every unknown prompt onto one lucky chain
    if len(tokenize(text)) and rank(words, n=1):
        top_word, top_score = rank(words, n=1)[0]
        # a matched 2+-word context scores far above the generic-(<bot>,)-only
        # case; detect it the same way the board does: re-run rank and inspect
        # whether any order-2+ context contributed
        matched = False
        for order in range(2, min(w, len(words)) + 1):
            if tuple(words[-order:]) in context_memory:
                matched = True
                break
        if not matched:
            phrase = FALLBACKS[fallback_idx % len(FALLBACKS)]
            fallback_idx += 1
            return phrase
    out, seen = [], set()
    for _ in range(max_tokens):
        ranked = rank(words, n=1)
        if not ranked:
            break
        word = ranked[0][0]
        if word in (END, "<user>", "<bot>"):
            break
        words.append(word)
        out.append(word)
        trigram = tuple(out[-3:])
        if len(out) >= 3 and trigram in seen:   # stuck in a loop: stop
            break
        seen.add(trigram)
    return " ".join(out)

def chat():
    print("Talk to the model. Type 'exit' to quit.\n")
    while True:
        text = input("you> ").strip()
        if text.lower() in ("exit", "quit"):
            break
        if not tokenize(text):
            continue
        print("bot> " + (reply(text) or "..."))

RUN_CHAT = True    # after training, open the chat prompt (needs a chat dataset)
RUN_SIMS = False   # set True to run the 100-times stability test (takes ~2 minutes)

if __name__ == "__main__":
    start = time.time()
    initiate()
    train()
    binarize()
    print(f"train time:        {time.time() - start:.2f}s")

    # Step 1 check: generate from a few prompts taken from your own dataset,
    # so every prompt is guaranteed to start with words the model has seen.
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

    if RUN_CHAT:
        chat()

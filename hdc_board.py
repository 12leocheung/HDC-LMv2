"""
hdc_board.py  --  runs on the ESP32 (MicroPython). No NumPy needed.

Copy these files to the board first (Thonny: View > Files):
    hdc_board.py, model.idx, model.bits, words.bits, vocab.txt, meta.txt
Run this script, then type a prompt at the  >  prompt.
(Rename it to main.py if you want it to start automatically at boot.)
"""
import os
import struct
from array import array

# These get filled in by load()
d = w = seed = n_words = n_records = word_bytes = 0
end_id = user_id = bot_id = -1
words_file = None       # words.bits stays on flash; read in small chunks while scoring
vocab_path = ""         # vocab.txt stays on flash; we read single lines when needed
idx_file = None         # model.idx stays open; we read 8 bytes at a time
bits_file = None        # model.bits stays open too


def load(folder=""):
    global d, w, seed, n_words, n_records, word_bytes
    global vocab_path, words_file, idx_file, bits_file, end_id, user_id, bot_id

    meta = {}
    with open(folder + "meta.txt") as f:
        for line in f:
            line = line.strip()
            if line:
                key, val = line.split("=")
                meta[key] = int(val)
    d, w, seed, n_words = meta["d"], meta["w"], meta["seed"], meta["words"]
    end_id = meta["end_id"]
    user_id = meta.get("user_id", -1)
    bot_id = meta.get("bot_id", -1)
    word_bytes = d // 8
    vocab_path = folder + "vocab.txt"

    words_file = open(folder + "words.bits", "rb")
    idx_file = open(folder + "model.idx", "rb")
    bits_file = open(folder + "model.bits", "rb")
    n_records = os.stat(folder + "model.idx")[6] // 8


def fnv(text):
    # must be identical to fnv() in export.py
    h = (2166136261 ^ seed) & 0xFFFFFFFF
    for ch in text.encode():
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h


def find(context_words):
    """Binary-search model.idx for this context. Returns its value, or None."""
    target = fnv(" ".join(context_words))
    lo, hi = 0, n_records - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        idx_file.seek(mid * 8)
        h, value = struct.unpack("<II", idx_file.read(8))
        if h == target:
            return value
        if h < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return None


def popcount(x):
    return bin(x).count("1")


def scores(words):
    """Same idea as rank() on the PC, but with whole numbers instead of floats.
    Every score is the PC's score multiplied by d."""
    # array('h') = 2 bytes per word instead of 4+, which matters on the ESP32
    total = array("h", (0 for _ in range(n_words)))
    found = False
    for order in range(1, min(w, len(words)) + 1):
        value = find(words[-order:])
        if value is None:
            continue
        found = True
        if value & 0x80000000:
            # only one word ever followed this context: it gets the full score
            total[value & 0x7FFFFFFF] += order * d
        else:
            # several followers: compare the stored vector to every word fingerprint,
            # reading words.bits from flash 128 words at a time (never all in RAM)
            bits_file.seek(value * word_bytes)
            v = int.from_bytes(bits_file.read(word_bytes), "big")
            words_file.seek(0)
            i = 0
            while i < n_words:
                chunk = words_file.read(word_bytes * 128)
                for off in range(0, len(chunk), word_bytes):
                    word = int.from_bytes(chunk[off:off + word_bytes], "big")
                    total[i] += order * (d - 2 * popcount(v ^ word))
                    i += 1
    return total if found else None


def word_at(i):
    """Read word number i from vocab.txt (saves keeping 779 strings in RAM)."""
    with open(vocab_path) as f:
        for n, line in enumerate(f):
            if n == i:
                return line.strip()
    return ""


def predict(words):
    """Returns the id of the best next word, or None."""
    total = scores(words)
    if total is None:
        return None
    best = 0
    for i in range(1, n_words):
        if total[i] > total[best]:
            best = i
    return best


def tokenize(s):
    # same rule as the PC regex  [a-z'-]+  (re.findall isn't available here)
    words = []
    current = ""
    for ch in s.lower():
        if ("a" <= ch <= "z") or ch == "'" or ch == "-":
            current += ch
        elif current:
            words.append(current)
            current = ""
    if current:
        words.append(current)
    return words


def generate(prompt, max_tokens=20):
    words = tokenize(prompt)
    new = []
    for _ in range(max_tokens):
        best = predict(words)
        if best is None or best == end_id:
            break
        word = word_at(best)
        words.append(word)
        new.append(word)
    return new


def reply(text, max_tokens=30):
    """Chat: feed '<user> your words <bot>' and keep predicting the next word."""
    words = ["<user>"] + tokenize(text) + ["<bot>"]
    out = []
    seen = set()
    for _ in range(max_tokens):
        best = predict(words)
        if best is None or best == end_id or best == user_id or best == bot_id:
            break
        word = word_at(best)
        words.append(word)
        out.append(word)
        if len(out) >= 3:
            trigram = out[-3] + " " + out[-2] + " " + out[-1]
            if trigram in seen:      # stuck repeating itself: stop
                break
            seen.add(trigram)
    return out


def terminal():
    print("Talk to the model. Empty line to quit.")
    while True:
        prompt = input("you> ")
        if not prompt.strip():
            break
        if not tokenize(prompt):
            print("(use some letters)")
            continue
        new = reply(prompt)
        print("bot> " + " ".join(new) if new else "bot> ...")


def serve():
    """Used by ESP_LLM.exe: one prompt per line in, one 'HDC:...' line out."""
    import sys
    import time
    print("HDC_READY")
    while True:
        line = sys.stdin.readline()
        if not line:
            time.sleep(0.05)
            continue
        prompt = line.strip()
        if prompt == "/ping":
            print("HDC:pong")
            continue
        try:
            print("HDC:" + " ".join(reply(prompt)))
        except Exception as e:
            print("HDC_ERR:" + repr(e))


if __name__ == "__main__":
    load()
    terminal()

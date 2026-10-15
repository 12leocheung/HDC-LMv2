"""
export.py  --  run this on your PC (not the ESP32).

It trains the model with hdc_llmv2.py, then writes 5 small files into the
folder  board_files/  . You copy those files onto the ESP32.

    model.idx    one 8-byte record per context:  [hash of the context][value]
                 sorted by hash so the board can binary-search it in the file
                   value with top bit set  -> a single word id (the only follower)
                   value without top bit   -> a slot number in model.bits
    model.bits   the 1-bit vectors (32 bytes each) for contexts with 2+ followers
    words.bits   the 1-bit fingerprint of every word (32 bytes each)
    vocab.txt    one word per line, in the same order as words.bits
    meta.txt     d, w, hash seed, vocab size, id of <end>
"""
import os
import struct
import numpy as np
import hdc_llmv2 as m

OUT = "board_files"
os.makedirs(OUT, exist_ok=True)

# ---- 1. train exactly like the PC script does -------------------------------
m.initiate()
m.train()
m.binarize()

# ---- 2. a simple hash: turns a context like ("the","golden") into a 32-bit number
# The board can't hold 25,000 tuples in RAM, so it looks the number up in the file.
def fnv(text, seed):
    h = (2166136261 ^ seed) & 0xFFFFFFFF
    for ch in text.encode():
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h

# Two different contexts could (rarely) get the same hash. Try seeds until none do.
for seed in range(1000):
    hashes = [fnv(" ".join(c), seed) for c in m.context_memory]
    if len(set(hashes)) == len(hashes):
        break
else:
    raise SystemExit("could not find a collision-free seed")
print(f"hash seed:       {seed}")

# ---- 3. write the context table ---------------------------------------------
records = []          # (hash, value)
multi_blob = []       # packed bit-vectors for multi-follower contexts
for context, v in m.context_memory.items():
    h = fnv(" ".join(context), seed)
    if isinstance(v, int):
        records.append((h, 0x80000000 | v))          # single follower: store its word id
    else:
        records.append((h, len(multi_blob)))         # multi: store which slot in model.bits
        multi_blob.append(v.tobytes())

records.sort()        # sorted by hash = binary search works on the board

with open(f"{OUT}/model.idx", "wb") as f:
    for h, val in records:
        f.write(struct.pack("<II", h, val))
with open(f"{OUT}/model.bits", "wb") as f:
    for b in multi_blob:
        f.write(b)

# ---- 4. write the word fingerprints and vocab -------------------------------
with open(f"{OUT}/words.bits", "wb") as f:
    for word in m.vocab:
        f.write(np.packbits(m.t1[word] > 0).tobytes())
with open(f"{OUT}/vocab.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(m.vocab))

with open(f"{OUT}/meta.txt", "w") as f:
    f.write(f"d={m.d}\nw={m.w}\nseed={seed}\nwords={len(m.vocab)}\nend_id={m.vocab.index(m.END)}\n")

# ---- 5. report sizes --------------------------------------------------------
total = 0
for name in ["model.idx", "model.bits", "words.bits", "vocab.txt", "meta.txt"]:
    size = os.path.getsize(f"{OUT}/{name}")
    total += size
    print(f"{name:12} {size / 1000:8.1f} KB")
print(f"{'total':12} {total / 1000:8.1f} KB")

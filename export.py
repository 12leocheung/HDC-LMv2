"""
export.py  --  run this on your PC (not the ESP32).

It trains the model with hdc_llmv2.py, then writes 6 small files into the
folder  board_files/  . You copy those files onto the ESP32.

    model.idx    all trained contexts, DELTA-ENCODED in hash order:
                 [4B header: record count][block: 5B base hash][64 x record]
                 each record: [3B delta to the base][3B value]
                   value with top bit set  -> the single follower's word id
                   value without top bit   -> a slot number in model.bits
                 The first record of a block IS its 5-byte base hash; every
                 other record stores the CHAIN delta to the PREVIOUS record
                 in its block (slot 0 stores 0), so only consecutive gaps
                 must fit 3 bytes. Every block holds exactly 64 records, and
                 record r sits at block r>>6, slot r&63: binary search runs
                 over global record indices and rebuilds the probe key by
                 prefix-summing the chain up to its slot. This makes a record
                 ~6.1 bytes instead of 8.
    model.bits   the packed multi-follower vectors (d//8 bytes each)
    words.bits   the 1-bit fingerprint of every word (d//8 bytes each)
    vocab.txt    one word per line
    vocab.idx    one 4-byte little-endian offset into vocab.txt per word
    meta.txt     d, w, hash seed, keys width, vocab size, ids of <end>, <user>, <bot>

The context hash is FNV-1a truncated to KEY_BITS (meta.txt: hash_bits),
sized to the corpus -- see section 2 for the two constraints. If a collision
or an oversized in-block delta ever appears, the export rehashes with the
next seed instead of quietly corrupting or re-failing the model.
"""
import os
import struct
import numpy as np
import hdc_llmv2 as m

# ---- reproducible fingerprints (do this BEFORE training) --------------------
# Each word's fingerprint comes from rng.choice, and hdc_llmv2's rng is
# UNSEEDED: every training run invents a new fingerprint table. words.bits and
# model.bits/slot numbers are only valid for the SAME run -- mixing a words.bits
# from one run with a model.bits from another turns every hamming comparison
# into noise, and the board babbles word salad (this actually happened after a
# words.bits from an old run was left on the board). Seeding here makes every
# export reproducible, so re-uploads can never be cross-run mismatches.
m.rng = np.random.default_rng(20261016)

OUT = "board_files"

# ---- size controls (edit these) ---------------------------------------------
MIN_COUNT = 2        # minimum count below which 2+-word contexts are always dropped
                     # (the export then raises it automatically until the files fit)
SIZE_LIMIT_MB = 6.0  # this ESP32 module in fact has 6.3 MB of filesystem
                     # (verified with os.statvfs). DO NOT lower this while the
                     # math families are in the corpus: the arithmetic windows
                     # are seen exactly once (count==1), so auto-raising
                     # MIN_COUNT to fit a tight budget would silently DELETE
                     # the whole math knowledge. The full-keep (MIN_COUNT=1)
                     # size is ~5.45 MB with present vocab: 5.4 was already too
                     # tight and trimmed every 0..99 fact. Keeps ~0.5 MB free.
os.makedirs(OUT, exist_ok=True)

BLOCK = 64            # records per block (power of two: r>>6 / r&63 on the board)
DELTA_MAX = 0xFFFFFE  # largest in-block delta; 0xFFFFFF is reserved padding
REC_BYTES = 6         # 3B delta + 3B value
# BASE_BYTES/BLOCK_BYTES depend on KEY_BITS and are set in section 2 below

# ---- 1. train exactly like the PC script does -------------------------------
m.initiate()
m.train()
m.binarize()

# ---- 1b. size budget: find the biggest pruning threshold that fits -----------
# Raising MIN_COUNT shrinks model.idx (fewer records kept) and model.bits
# (fewer multi-follower vectors). Instead of guessing a threshold by hand we
# measure what each candidate would cost from the trained state, pick the
# smallest one that lands under SIZE_LIMIT_MB, and only then delete anything.
vec_bytes = m.d // 8               # every stored fingerprint is d bits
vocab_bytes = sum(len(w.encode("utf-8")) + 1 for w in m.vocab)     # word + newline each
fixed_bytes = (len(m.vocab) * vec_bytes                                        # words.bits
               + vocab_bytes                                                   # vocab.txt
               + len(m.vocab) * 4                                              # vocab.idx
               + 300)                                                          # meta.txt (padded)

always_k, always_multi = 0, 0      # unigram contexts: kept whatever MIN_COUNT is
rare_hist = {}                     # count -> [num contexts, num multi-follower] for 2+ word contexts
for context, v in m.context_memory.items():
    if not isinstance(v, int):
        multi = 1
    else:
        multi = 0
    if len(context) == 1:
        always_k += 1
        always_multi += multi
    else:
        entry = rare_hist.setdefault(m.counts[context], [0, 0])
        entry[0] += 1
        entry[1] += multi

MIN_COUNT = 1                      # start with no pruning at all
LIMIT_BYTES = SIZE_LIMIT_MB * 1e6  # the report below uses 1000-based KB
while True:
    kept = always_k
    multi = always_multi
    for count, (n, n_multi) in rare_hist.items():
        if count >= MIN_COUNT:
            kept += n
            multi += n_multi
    projected = fixed_bytes + kept * 7 + multi * vec_bytes   # ~6.1 B/real record; padded to 7
    if projected <= LIMIT_BYTES or MIN_COUNT >= 1000:
        break
    MIN_COUNT += 1

before = len(m.context_memory)

# MIN_COUNT prunes RARE NATURAL-LANGUAGE windows (genuine noise), but the
# arithmetic identities are also count-1 multi-word contexts -- each glued
# window ('83-27','is') is seen exactly once. Dropping them would silently
# delete whole math families (this actually happened: MC auto-raised enough
# under a tight budget and the board answered every big subtraction wrong).
# So: rare windows stay in full whenever they or their follower are
# arithmetic -- any token containing a digit, or the 'minus' answer frame --
# and only the non-arithmetic rare contexts obey MIN_COUNT.
def _rare_arith(context, follower_word):
    for tok in context:
        if any(ch.isdigit() for ch in tok) or tok == "minus":
            return True
    return any(ch.isdigit() for ch in follower_word)

dropped = kept_arith = 0
for context in list(m.context_memory):
    if len(context) > 1 and m.counts[context] < MIN_COUNT:
        v = m.context_memory[context]
        # count==1 entries hold the follower id directly; bigger counts hold a
        # follower list that MIN_COUNT would never touch anyway
        follower_word = m.vocab[v] if isinstance(v, int) else ""
        if _rare_arith(context, follower_word):
            kept_arith += 1
        else:
            del m.context_memory[context]
            dropped += 1
print(f"rare windows pruned: {dropped} (non-arithmetic)"
      f" / kept in full: {kept_arith} (arithmetic)")
print(f"auto MIN_COUNT: {MIN_COUNT} (projects {projected / 1e6:.2f} MB against a "
      f"{SIZE_LIMIT_MB} MB limit)")
print(f"contexts kept:   {len(m.context_memory)} of {before}")

# ---- 2. key the contexts with an n-scaled FNV-1a ----------------------------
def fnv_key(text, seed, bits):
    h = (0xCBF29CE484222325 ^ seed) & ((1 << bits) - 1)
    for ch in text.encode():
        h = ((h ^ ch) * 0x100000001B3) & ((1 << bits) - 1)
    return h

# Two constraints size the hash width W for n records (mean gap between
# sorted hashes is 2^W/n):
#   deltas:  each record stores the CHAIN delta to the previous record, so
#            only consecutive gaps must fit 3 bytes -- at W=20+floor(log2 n)
#            the mean gap is always in (2^19, 2^20], so an overflow is
#            ~e^-16 per gap per seed (never)
#   query safety: per-seed a collision must simply not happen; the width
#            n^2/2^(W+1) ~ n/2^21 stays under a few percent for any n the
#            4 MB budget can hold, and if a collision or an oversized gap
#            ever does occur we rehash with the NEXT seed instead of
#            dropping a fact (the chosen seed is written to meta.txt).
entries = []              # ("word ids joined", value)
multi_blob = []           # packed bit-vectors for multi-follower contexts
for context, v in m.context_memory.items():
    if isinstance(v, int):
        entries.append((" ".join(context), 0x800000 | v))   # single: word id
    else:
        entries.append((" ".join(context), len(multi_blob)))  # multi: slot
        multi_blob.append(v.tobytes())

n = len(entries)
KEY_BITS = min(40, 20 + n.bit_length() - 1)          # n >= 1
BASE_BYTES = (KEY_BITS + 7) // 8                     # <= 40 bits -> 5 bytes
BLOCK_BYTES = BASE_BYTES + BLOCK * REC_BYTES

seed = 0
while True:                                           # rehash on any bad gap
    records = [(fnv_key(t, seed, KEY_BITS), val) for t, val in entries]
    records.sort()
    clean = True
    for i in range(1, n):
        # a clash between live OR an in-block delta overflow must never be
        # silently resolved; both are fixed by rehashing with the next seed
        if records[i][0] == records[i - 1][0] or records[i][0] - records[i - 1][0] >= DELTA_MAX:
            clean = False
            break
    if clean:
        break
    seed += 1
    if seed > 1000:
        raise SystemExit("could not find a clean hash seed -- impossible: report this")
print(f"hash bits:        {KEY_BITS} (seed {seed}: collision-free, all gaps in 3 bytes, {n} records)")

# ---- 3. write the delta-encoded context table -------------------------------
# Deltas are CHAINED: each record stores key - previous_key (slot 0 stores 0,
# its key IS the block base). This keeps only consecutive gaps below the
# 3-byte capacity instead of whole-block spans (a 64-record block spans ~63
# gaps -- delta-vs-base at mean gap 2^20 wraps mod 2^24 and silently
# corrupts half the keys, which actually happened here). The board rebuilds
# the probe key by prefix-summing the chain up to its slot: one contiguous
# read per binary-search probe, ~32 integer adds on average.
# every block holds EXACTLY BLOCK records (padding in the last one), so the
# board's record r -> (block r>>6, slot r&63) arithmetic stays exact
with open(f"{OUT}/model.idx", "wb") as f:
    f.write(struct.pack("<I", n))
    for b in range(0, n, BLOCK):
        chunk = records[b:b + BLOCK]
        base = chunk[0][0]
        f.write(struct.pack("<Q", base)[:BASE_BYTES])
        prev = base
        for slot in range(BLOCK):
            if slot < len(chunk):
                delta = chunk[slot][0] - prev
                prev = chunk[slot][0]
                val = chunk[slot][1]
            else:
                delta = DELTA_MAX             # padding outside the search domain
                val = 0
            f.write(struct.pack("<I", delta)[:3])
            f.write(struct.pack("<I", val)[:3])
with open(f"{OUT}/model.bits", "wb") as f:
    for b in multi_blob:
        f.write(b)

# ---- 3b. validate: replay every context through the exact board lookup ------
idx_file = open(f"{OUT}/model.idx", "rb")       # one handle for the whole sweep

def board_find(context):
    """Mirror of the board's find(): binary search + chained-delta decode."""
    target = fnv_key(" ".join(context), seed, KEY_BITS)
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) >> 1
        boff = 4 + (mid >> 6) * BLOCK_BYTES
        idx_file.seek(boff)
        key = int.from_bytes(idx_file.read(BASE_BYTES), "little")
        idx_file.seek(boff + BASE_BYTES)
        n_pairs = (mid & 63) + 1
        blob = idx_file.read(n_pairs * 6)          # [3B delta][3B value] pairs
        for i in range(0, len(blob), 6):           # deltas are the pair's FIRST half
            key += int.from_bytes(blob[i:i + 3], "little")
        if key < target:
            lo = mid + 1
        elif key > target:
            hi = mid
        else:
            return int.from_bytes(blob[-3:], "little")   # the pair's value

val_of = {}            # hash -> value, from the records we actually wrote
for rec_hash, rec_val in records:
    val_of[rec_hash] = rec_val
board_find_decode = None      # debug helper no longer needed
bad = 0
for context, v in m.context_memory.items():
    target = fnv_key(" ".join(context), seed, KEY_BITS)
    got = board_find(context)
    want = val_of[target]
    if got != want:
        bad += 1
        if bad < 4:
            print("MISMATCH:", context, "want", want, "got", got)
print("blocked-lookup check:", "all exact" if bad == 0 else f"{bad} MISMATCHES")
idx_file.close()
if bad:
    raise SystemExit("model.idx validation failed -- refusing to ship")

# ---- 4. write the word fingerprints and vocab -------------------------------
with open(f"{OUT}/words.bits", "wb") as f:
    for word in m.vocab:
        f.write(np.packbits(m.t1[word] > 0).tobytes())
# vocab.txt stays one word per line; vocab.idx adds a 4-byte file offset per
# word (sorted, same order as words.bits), so the board seeks straight to word
# i instead of scanning the file for every generated token (~1.5 s per word).
offsets = bytearray()
pos = 0
with open(f"{OUT}/vocab.txt", "wb") as f:
    for word in m.vocab:
        b = word.encode("utf-8")
        offsets += struct.pack("<I", pos)
        f.write(b + b"\n")
        pos += len(b) + 1
with open(f"{OUT}/vocab.idx", "wb") as f:
    f.write(bytes(offsets))

with open(f"{OUT}/meta.txt", "w", newline="") as f:
    def id_of(word):
        return m.vocab.index(word) if word in m.vocab else -1
    f.write(f"d={m.d}\nw={m.w}\nseed={seed}\nwords={len(m.vocab)}\nend_id={id_of(m.END)}\n"
            f"user_id={id_of('<user>')}\nbot_id={id_of('<bot>')}\nvocab_offsets=1\n"
            f"hash_bits={KEY_BITS}\n")

# ---- 5. report sizes --------------------------------------------------------
total = 0
for name in ["model.idx", "model.bits", "words.bits", "vocab.txt", "vocab.idx", "meta.txt"]:
    size = os.path.getsize(f"{OUT}/{name}")
    total += size
    print(f"{name:12} {size / 1000:8.1f} KB")
print(f"{'total':12} {total / 1000:8.1f} KB")
if total / 1e6 > SIZE_LIMIT_MB:
    print(f"WARNING: {total / 1e6:.2f} MB is still over your {SIZE_LIMIT_MB} MB limit "
          "even at the highest useful MIN_COUNT. "
          "Raise SIZE_LIMIT_MB if the board has room, or shorten the dataset.")

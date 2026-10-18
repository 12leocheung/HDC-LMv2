"""
make_chat_dataset.py -- builds a CONVERSATION dataset.txt you can actually talk to.

Every line is one exchange:
    <user> what is your name <bot> my name is chatterbot

Two real sources, both downloaded from GitHub:
  1. ChatterBot corpus (gunthercox/chatterbot-corpus) - human-written small talk,
     greetings, bot profile, emotions, science, trivia, coding, tech support.
     License: BSD 3-clause.
  2. Stanford Alpaca (tatsu-lab/stanford_alpaca) - questions with AI-written answers.
     Only the first sentence or two of each answer is kept so replies stay short.
     License: CC BY-NC 4.0 (non-commercial use).

Usage:
    python make_chat_dataset.py                       # default mix -> dataset.txt
    python make_chat_dataset.py --alpaca-pairs 50000  # more Q&A
    python make_chat_dataset.py --alpaca-pairs 0      # small talk only (tiny)
    python make_chat_dataset.py --chat-repeat 5       # small talk counts 5x
    python make_chat_dataset.py --bot-name Leo        # rename the bot
"""
import argparse
import io
import json
import os
import random
import re
import urllib.request
import zipfile

CHAT_URL = "https://codeload.github.com/gunthercox/chatterbot-corpus/zip/refs/heads/master"
ALPACA_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"

parser = argparse.ArgumentParser()
parser.add_argument("--alpaca-pairs", type=int, default=30000)
parser.add_argument("--chat-repeat", type=int, default=3,
                    help="repeat the small-talk lines this many times so they aren't drowned out")
parser.add_argument("--bot-name", default="Chatterbot")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="dataset.txt")
args = parser.parse_args()

TAGS = re.compile(r"<[^>]*>")
SPACES = re.compile(r"\s+")
SENT = re.compile(r"(?<=[.!?])\s+")

def clean(text):
    text = TAGS.sub(" ", str(text))          # remove html so it can't fake <user>/<bot>
    text = re.sub(r"\bchatterbot\b", args.bot_name, text, flags=re.I)
    return SPACES.sub(" ", text).strip()

def line(q, a):
    return f"<user> {q} <bot> {a}"

# ---- 1. ChatterBot corpus (tiny hand-rolled parser, no PyYAML needed) --------
def unquote(s):
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        s = s[1:-1]
    return s

def parse_yml(text):
    conversations, current = [], None
    for raw in text.splitlines():
        if raw.startswith("- - "):
            current = [unquote(raw[4:])]
            conversations.append(current)
        elif raw.startswith("  - ") and current is not None:
            current.append(unquote(raw[4:]))
    return conversations

print("downloading ChatterBot corpus...")
data = urllib.request.urlopen(CHAT_URL).read()
chat_lines = []
with zipfile.ZipFile(io.BytesIO(data)) as z:
    for name in z.namelist():
        if "/data/english/" in name and name.endswith(".yml"):
            for conv in parse_yml(z.read(name).decode("utf-8")):
                for q, a in zip(conv, conv[1:]):          # every turn answers the one before
                    q, a = clean(q), clean(a)
                    if q and a and len(q.split()) <= 30 and len(a.split()) <= 60:
                        chat_lines.append(line(q, a))
chat_lines = sorted(set(chat_lines))
print(f"small talk exchanges: {len(chat_lines)}")

# ---- 2. Alpaca questions with short answers ----------------------------------
alpaca_lines = []
if args.alpaca_pairs:
    if not os.path.exists("alpaca_data.json"):
        print("downloading Alpaca (about 22 MB)...")
        urllib.request.urlretrieve(ALPACA_URL, "alpaca_data.json")
    with open("alpaca_data.json", encoding="utf-8") as f:
        alpaca = json.load(f)
    seen = set()
    for ex in alpaca:
        if ex["input"].strip():                 # needs extra text we don't have -> skip
            continue
        q = clean(ex["instruction"])
        if not q or len(q.split()) > 25:
            continue
        sentences = [s for s in SENT.split(clean(ex["output"])) if s]
        if not sentences or len(sentences[0].split()) > 45:
            continue
        answer = sentences[0]
        for s in sentences[1:2]:                # allow a second sentence if short enough
            if len(answer.split()) + len(s.split()) <= 40:
                answer += " " + s
        key = (q.lower(), answer.lower())
        if key not in seen:
            seen.add(key)
            alpaca_lines.append(line(q, answer))
    random.Random(args.seed).shuffle(alpaca_lines)
    alpaca_lines = alpaca_lines[:args.alpaca_pairs]
print(f"Q&A exchanges:        {len(alpaca_lines)}")

# ---- 3. combine and save ------------------------------------------------------
lines = chat_lines * args.chat_repeat + alpaca_lines
random.Random(args.seed).shuffle(lines)
with open(args.out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"wrote {len(lines)} lines to {args.out}")
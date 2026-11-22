"""
tokenize.py
===========
One tokenizer shared by the data generator (gold labels) and the extractor
(prediction time). Using the same tokenizer in both places matters: if
training and inference split text differently, the CRF sees tokens at test
time that never existed during training (e.g. "BREAKING:" vs "BREAKING" ":").

Rules (tuned for social-media text):
  - hashtags stay whole:            "#shutdown"
  - hyphen/apostrophe words whole:  "sit-in", "walk-out", "can't"
  - every other punctuation mark is its own token: "BREAKING", ":"
"""
import re

TOKEN_RE = re.compile(r"#\w+|\w+(?:[-']\w+)*|[^\w\s]")


def tokenize(text: str):
    return TOKEN_RE.findall(text or "")

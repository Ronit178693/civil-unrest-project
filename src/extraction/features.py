"""
features.py
===========
Hand-engineered + lexicon-driven token features for the civil-unrest event
tagger. This feature set is designed specifically for noisy social-media
text (hashtags, elisions, code-mixed tokens, emphatic punctuation) rather
than clean newswire, which is the key adaptation the project proposal calls
for ("adapted for civil-unrest-specific trigger words and arguments").

The CRF trained on these features is the offline-runnable baseline shipped
in this repo. `model.py` documents the drop-in path to a fine-tuned
transformer (e.g. BERTweet / XLM-R) for higher recall once GPU + internet
access to model weights is available -- the feature functions here double
as a sanity-check/ensemble signal even after that upgrade.
"""
import re
from .lexicon import (GAZETTEER, FLAT_TRIGGERS, ACTOR_HINTS, SCALE_WORDS, HEDGE_WORDS,
                      ESCALATION_WORDS, TIME_WORDS)

WORD_SHAPE_RE = re.compile(r"[A-Za-z]")


def word_shape(token: str) -> str:
    shape = []
    for ch in token:
        if ch.isupper():
            shape.append("X")
        elif ch.islower():
            shape.append("x")
        elif ch.isdigit():
            shape.append("d")
        else:
            shape.append(ch)
    # collapse repeats
    collapsed = []
    for c in shape:
        if not collapsed or collapsed[-1] != c:
            collapsed.append(c)
    return "".join(collapsed)


def token_features(tokens, i):
    tok = tokens[i]
    tok_l = tok.lower().strip(",.!?")
    feats = {
        "bias": 1.0,
        "tok.lower": tok_l,
        "tok.isupper": tok.isupper(),
        "tok.istitle": tok.istitle(),
        "tok.isdigit": tok.isdigit(),
        "tok.shape": word_shape(tok),
        "tok.is_hashtag": tok.startswith("#"),
        "tok.is_trigger": tok_l in FLAT_TRIGGERS,
        "tok.trigger_type": FLAT_TRIGGERS.get(tok_l, "NONE"),
        "tok.is_escalation_word": tok_l in ESCALATION_WORDS,
        "tok.is_actor_hint": tok_l in ACTOR_HINTS,
        "tok.is_scale_word": tok_l in SCALE_WORDS,
        "tok.is_time_word": tok_l in TIME_WORDS,
        "tok.is_hedge": tok_l in HEDGE_WORDS,
        "tok.len": len(tok),
        "tok.suffix3": tok_l[-3:],
        "tok.prefix3": tok_l[:3],
        "tok.is_punct": not any(ch.isalnum() for ch in tok),
        "tok.in_gazetteer": tok_l in GAZETTEER,
    }
    # bigram trigger check (e.g. "tear gas", "sit-in", "lathi charge")
    if i < len(tokens) - 1:
        bigram = f"{tok_l} {tokens[i+1].lower().strip(',.!?')}"
        feats["bigram.is_trigger"] = bigram in FLAT_TRIGGERS
        feats["bigram.in_gazetteer"] = bigram in GAZETTEER
        feats["bigram.trigger_type"] = FLAT_TRIGGERS.get(bigram, "NONE")
    else:
        feats["bigram.is_trigger"] = False
        feats["bigram.trigger_type"] = "NONE"
        feats["bigram.in_gazetteer"] = False

    if i > 0:
        prev = tokens[i - 1]
        prev_l = prev.lower().strip(",.!?")
        feats.update({
            "-1:tok.lower": prev_l,
            "-1:tok.istitle": prev.istitle(),
            "-1:tok.is_trigger": prev_l in FLAT_TRIGGERS,
        })
    else:
        feats["BOS"] = True

    if i < len(tokens) - 1:
        nxt = tokens[i + 1]
        nxt_l = nxt.lower().strip(",.!?")
        feats.update({
            "+1:tok.lower": nxt_l,
            "+1:tok.istitle": nxt.istitle(),
            "+1:tok.is_trigger": nxt_l in FLAT_TRIGGERS,
        })
    else:
        feats["EOS"] = True

    # wider context: two tokens either side
    if i > 1:
        feats["-2:tok.lower"] = tokens[i - 2].lower()
    if i < len(tokens) - 2:
        feats["+2:tok.lower"] = tokens[i + 2].lower()

    return feats


def sentence_to_features(tokens):
    return [token_features(tokens, i) for i in range(len(tokens))]

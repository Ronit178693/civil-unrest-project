"""
model.py
========
Civil-unrest event extraction model.

Architecture note (see paper, Section 4 "Methodology"):
---------------------------------------------------------
The project proposal calls for a *fine-tuned transformer-based* event
extraction module. In this environment there is no network path to
download pretrained transformer weights (e.g. HuggingFace Hub), so this
repository ships a fully offline-trainable **linear-chain CRF** over rich
lexicon + orthographic features as the reference baseline (`UnrestCRFTagger`).

A `TransformerTaggerStub` class below documents exactly how to swap in a
fine-tuned transformer (e.g. `vinai/bertweet-base` or `xlm-roberta-base`,
both well-suited to noisy/multilingual social text) once HuggingFace access
is available -- the rest of the pipeline (aggregation, prediction,
evaluation) is agnostic to which tagger produced the BIO labels, so the
swap is a one-file change.
"""
import json
from typing import List, Dict, Tuple

import sklearn_crfsuite
from sklearn_crfsuite import metrics as crf_metrics

from .features import sentence_to_features
from .tokenize import tokenize


LABELS = ["O", "B-TRIGGER", "I-TRIGGER", "B-ACTOR", "I-ACTOR",
          "B-LOCATION", "I-LOCATION", "B-TARGET", "I-TARGET"]


def _span_set(labels_per_sentence):
    spans = set()
    for s_idx, labels in enumerate(labels_per_sentence):
        start, tag = None, None
        for i, lab in enumerate(list(labels) + ["O"]):
            if start is not None and not (lab.startswith("I-") and lab[2:] == tag):
                spans.add((s_idx, start, i, tag))
                start, tag = None, None
            if lab.startswith("B-") or (lab.startswith("I-") and start is None):
                start, tag = i, lab[2:]
    return spans


def span_scores(y_true, y_pred):
    """Entity-level exact-match precision/recall/F1 (a span counts only if
    its start, end and type are all right) -- stricter than per-token F1."""
    gold, pred = _span_set(y_true), _span_set(y_pred)

    def prf(g, p):
        tp = len(g & p)
        prec = tp / len(p) if p else 0.0
        rec = tp / len(g) if g else 0.0
        return prec, rec, (2 * prec * rec / (prec + rec) if prec + rec else 0.0)

    per_type = {}
    for t in sorted({s[3] for s in gold | pred}):
        g = {s for s in gold if s[3] == t}
        p = {s for s in pred if s[3] == t}
        pr, rc, f = prf(g, p)
        per_type[t] = {"precision": pr, "recall": rc, "f1": f, "support": len(g)}
    pr, rc, f = prf(gold, pred)
    return {"precision": pr, "recall": rc, "f1": f, "per_type": per_type}


class UnrestCRFTagger:
    """Linear-chain CRF sequence tagger for civil-unrest event triggers and
    argument roles (ACTOR, LOCATION, TARGET). BIO tagging scheme.
    """

    def __init__(self, c1=0.1, c2=0.1, max_iterations=150):
        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=True,
        )
        self._fitted = False

    def fit(self, X_tokens: List[List[str]], y_labels: List[List[str]]):
        X = [sentence_to_features(toks) for toks in X_tokens]
        self.model.fit(X, y_labels)
        self._fitted = True
        return self

    def predict(self, X_tokens: List[List[str]]) -> List[List[str]]:
        if not self._fitted:
            raise RuntimeError("Call .fit() before .predict().")
        X = [sentence_to_features(toks) for toks in X_tokens]
        return self.model.predict(X)

    def extract_many(self, texts) -> List[Dict]:
        """Batch version of extract_events (much faster for thousands of posts)."""
        token_lists = [tokenize(t) for t in texts]
        non_empty = [i for i, t in enumerate(token_lists) if t]
        preds = self.predict([token_lists[i] for i in non_empty]) if non_empty else []
        frames = [{"trigger": None, "actor": None, "location": None, "target": None}
                  for _ in token_lists]
        for i, labels in zip(non_empty, preds):
            frames[i] = self._bio_to_frame(token_lists[i], labels)
        return frames

    def evaluate(self, X_tokens, y_true):
        y_pred = self.predict(X_tokens)
        labels = [l for l in self.model.classes_ if l != "O"]
        report = crf_metrics.flat_classification_report(
            y_true, y_pred, labels=labels, digits=3, zero_division=0
        )
        f1 = crf_metrics.flat_f1_score(y_true, y_pred, labels=labels,
                                        average="weighted", zero_division=0)
        precision = crf_metrics.flat_precision_score(y_true, y_pred, labels=labels,
                                                       average="weighted", zero_division=0)
        recall = crf_metrics.flat_recall_score(y_true, y_pred, labels=labels,
                                                 average="weighted", zero_division=0)
        span = span_scores(y_true, y_pred)
        return {"report": report, "f1": f1, "precision": precision, "recall": recall,
                "span_precision": span["precision"], "span_recall": span["recall"],
                "span_f1": span["f1"], "span_per_type": span["per_type"]}

    def extract_events(self, text: str) -> Dict:
        """Run the tagger on a single raw text string and return a
        structured event frame: {trigger, trigger_type, actor, location,
        target}. This is the function the real-time pipeline (and the
        prediction module) calls per-post.
        """
        tokens = tokenize(text)
        if not tokens:
            return {}
        labels = self.predict([tokens])[0]
        return self._bio_to_frame(tokens, labels)

    @staticmethod
    def _bio_to_frame(tokens, labels) -> Dict:
        frame = {"trigger": None, "actor": None, "location": None, "target": None}
        cur_tag, cur_tokens = None, []

        def flush():
            if cur_tag and cur_tokens:
                key = cur_tag.lower()
                text = " ".join(cur_tokens)
                if key in frame:
                    frame[key] = text if frame[key] is None else frame[key] + "; " + text

        for tok, lab in zip(tokens, labels):
            if lab.startswith("B-"):
                flush()
                cur_tag, cur_tokens = lab[2:], [tok]
            elif lab.startswith("I-") and cur_tag == lab[2:]:
                cur_tokens.append(tok)
            else:
                flush()
                cur_tag, cur_tokens = None, []
        flush()
        return frame

    def save(self, path: str):
        import joblib
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str):
        import joblib
        inst = cls()
        inst.model = joblib.load(path)
        inst._fitted = True
        return inst


class TransformerTaggerStub:
    """
    Documentation-only stub showing the upgrade path once HuggingFace model
    weights are reachable (e.g. in a GPU training environment with internet
    access). NOT executed by the pipeline; kept here for the paper's
    reproducibility appendix and for future contributors.

    from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    model = AutoModelForTokenClassification.from_pretrained(
        "vinai/bertweet-base", num_labels=len(LABELS)
    )
    # ... tokenize with is_split_into_words=True, align BIO labels to
    # wordpieces, then fine-tune with Trainer on the same (tokens, labels)
    # pairs UnrestCRFTagger.fit() consumes. Decoding and evaluate() keep the
    # same interface so src/pipeline.py does not need to change.
    """
    pass

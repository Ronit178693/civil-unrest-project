"""
stage.py
========
Post-level event-stage classifier: grievance / mobilization / on_ground /
escalation.

The CRF extracts WHO did WHAT, WHERE, against WHAT; it does not say where a
post sits in the unrest process. ("Workers plan a strike tomorrow" and
"Workers are on strike right now" share the same trigger.) This classifier
adds that missing piece so that Stage 2 can be fed entirely from Stage 1's
output instead of from the generator's gold labels.

Model: TF-IDF (word 1-2 grams + character 3-5 grams, robust to spelling noise
and hashtags) -> multinomial logistic regression with balanced class weights
(escalation is the rarest stage, ~8% of posts).
"""
from typing import List

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from .lexicon import FLAT_TRIGGERS
from .tokenize import tokenize

STAGES = ["grievance", "mobilization", "on_ground", "escalation"]
ESCALATION_TYPES = {"RIOT", "CLASH", "REPRESSION"}


FUTURE_CUES = {"tomorrow", "planning", "plan", "planned", "calling", "call", "join", "will",
               "weekend", "monday", "announce", "announced", "upcoming", "organise", "organize",
               "scheduled", "come"}
LIVE_CUES = {"live", "now", "gathered", "gathering", "continues", "ongoing", "streets", "still",
             "visuals", "happening", "currently", "underway", "site", "holding", "tense"}


class CueFeatures(BaseEstimator, TransformerMixin):
    """Small dense block of domain cues that generalize across phrasings:
    trigger presence (grievance posts usually have none), escalation-type
    trigger, and future-vs-live wording (mobilization vs on-ground)."""

    def __init__(self, weight: float = 3.0):
        self.weight = weight

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            toks = [t.lower() for t in tokenize(text)]
            words = set(toks) | {" ".join(p) for p in zip(toks, toks[1:])}
            types = {FLAT_TRIGGERS[w] for w in words if w in FLAT_TRIGGERS}
            rows.append([
                float(bool(types)),
                float(bool(types & ESCALATION_TYPES)),
                float(bool(words & FUTURE_CUES)),
                float(bool(words & LIVE_CUES)),
            ])
        return csr_matrix(np.array(rows) * self.weight)


class StageClassifier:
    def __init__(self, C: float = 1.0):
        self.model = Pipeline([
            ("tfidf", FeatureUnion([
                ("word", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                         lowercase=True, sublinear_tf=True)),
                ("cues", CueFeatures()),
            ])),
            ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced")),
        ])

    def fit(self, texts: List[str], stages: List[str]):
        self.model.fit(list(texts), list(stages))
        return self

    def predict(self, texts: List[str]):
        return self.model.predict(list(texts))

    def evaluate(self, texts, stages):
        pred = self.predict(texts)
        return {
            "accuracy": accuracy_score(stages, pred),
            "macro_f1": f1_score(stages, pred, average="macro"),
            "report": classification_report(stages, pred, labels=STAGES, digits=3, zero_division=0),
        }

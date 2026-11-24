"""
transformer_tagger.py
=====================
Fine-tuned transformer event extractor (the architecture the project brief
asks for). Token classification over the same BIO label set as the CRF:
TRIGGER, ACTOR, LOCATION, TARGET, TIME, SCALE (event type is read off the
trigger, see model.bio_to_frame).

Default encoder: `cardiffnlp/twitter-roberta-base` -- RoBERTa further
pre-trained on ~58M tweets, so it already knows informal social-media
language (hashtags, slang, missing capitals). Fine-tuning adapts it to the
civil-unrest roles.

Same interface as UnrestCRFTagger (fit / predict / extract_many /
extract_events / evaluate), so the rest of the pipeline is unchanged.

Word-to-subword alignment: the label of each word goes on its FIRST subword;
other subwords are ignored in the loss (-100) and at prediction time.
Predicted label sequences are repaired so an I-X never follows O or another
type (it becomes B-X).

Runs on CPU (~1-2 min per epoch for 2-3k short posts on a laptop); uses a GPU
automatically when one is available.
"""
import os
import random
from typing import Dict, List, Optional

import numpy as np

from .model import LABELS, bio_to_frame, empty_frame, span_scores
from .tokenize import tokenize

_LOCAL = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                      "models_cache", "twitter-roberta-base")
# Use a local copy when one exists (models_cache/, for slow or flaky connections),
# otherwise download from the Hugging Face Hub.
DEFAULT_MODEL = _LOCAL if os.path.exists(os.path.join(_LOCAL, "pytorch_model.bin"))     else "cardiffnlp/twitter-roberta-base"
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


def repair_bio(labels: List[str]) -> List[str]:
    out, prev = [], "O"
    for lab in labels:
        if lab.startswith("I-") and not (prev != "O" and prev[2:] == lab[2:]):
            lab = "B-" + lab[2:]
        out.append(lab)
        prev = lab
    return out


class TransformerTagger:
    def __init__(self, model_name: str = DEFAULT_MODEL, epochs: int = 4, lr: float = 3e-5,
                 batch_size: int = 16, max_len: int = 96, seed: int = 42,
                 warmup_share: float = 0.1, threads: Optional[int] = None, verbose: bool = True):
        self.model_name = model_name
        self.epochs, self.lr, self.batch_size, self.max_len = epochs, lr, batch_size, max_len
        self.seed, self.warmup_share, self.threads, self.verbose = seed, warmup_share, threads, verbose
        self.model = self.tokenizer = None
        self._fitted = False

    # ------------------------------------------------------------------ setup
    def _torch(self):
        import torch
        if self.threads:
            torch.set_num_threads(self.threads)
        return torch

    def _device(self):
        torch = self._torch()
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_pretrained(self):
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, add_prefix_space=True)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name, num_labels=len(LABELS),
            id2label={i: l for l, i in LABEL2ID.items()}, label2id=LABEL2ID)

    def _encode(self, X_tokens, y_labels=None):
        enc = self.tokenizer(X_tokens, is_split_into_words=True, truncation=True,
                             max_length=self.max_len, padding=True, return_tensors="pt")
        word_ids = [enc.word_ids(i) for i in range(len(X_tokens))]
        if y_labels is None:
            return enc, word_ids
        aligned = []
        for i, labs in enumerate(y_labels):
            prev, row = None, []
            for w in word_ids[i]:
                row.append(-100 if w is None or w == prev else LABEL2ID[labs[w]])
                prev = w
            aligned.append(row)
        torch = self._torch()
        enc["labels"] = torch.tensor(aligned)
        return enc, word_ids

    # ------------------------------------------------------------------ train
    def fit(self, X_tokens: List[List[str]], y_labels: List[List[str]]):
        torch = self._torch()
        from transformers import get_linear_schedule_with_warmup
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self._load_pretrained()
        device = self._device()
        self.model.to(device).train()

        data = [(t, l) for t, l in zip(X_tokens, y_labels) if t]
        n_steps = self.epochs * ((len(data) + self.batch_size - 1) // self.batch_size)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, int(self.warmup_share * n_steps), n_steps)
        rng = random.Random(self.seed)
        for epoch in range(self.epochs):
            rng.shuffle(data)
            total, n = 0.0, 0
            for b in range(0, len(data), self.batch_size):
                batch = data[b:b + self.batch_size]
                enc, _ = self._encode([t for t, _ in batch], [l for _, l in batch])
                out = self.model(**{k: v.to(device) for k, v in enc.items()})
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                total, n = total + out.loss.item(), n + 1
            if self.verbose:
                print(f"    transformer epoch {epoch + 1}/{self.epochs}: loss {total / n:.4f}",
                      flush=True)
        self.model.eval()
        self._fitted = True
        return self

    # ------------------------------------------------------------------ predict
    def predict(self, X_tokens: List[List[str]], batch_size: int = 64) -> List[List[str]]:
        if not self._fitted:
            raise RuntimeError("Call .fit() before .predict().")
        torch = self._torch()
        device = self._device()
        preds: List[Optional[List[str]]] = [None] * len(X_tokens)
        order = sorted(range(len(X_tokens)), key=lambda i: len(X_tokens[i]))  # less padding
        with torch.no_grad():
            for b in range(0, len(order), batch_size):
                idx = order[b:b + batch_size]
                toks = [X_tokens[i] if X_tokens[i] else ["."] for i in idx]
                enc, word_ids = self._encode(toks)
                logits = self.model(**{k: v.to(device) for k, v in enc.items()}).logits
                best = logits.argmax(-1).cpu().numpy()
                for row, i in enumerate(idx):
                    labs = ["O"] * len(X_tokens[i])
                    prev = None
                    for pos, w in enumerate(word_ids[row]):
                        if w is not None and w != prev and w < len(labs):
                            labs[w] = LABELS[best[row, pos]]
                        prev = w
                    preds[i] = repair_bio(labs)
        return preds

    def extract_many(self, texts) -> List[Dict]:
        token_lists = [tokenize(t) for t in texts]
        labels = self.predict(token_lists)
        return [bio_to_frame(t, l) if t else empty_frame() for t, l in zip(token_lists, labels)]

    def extract_events(self, text: str) -> Dict:
        return self.extract_many([text])[0]

    def evaluate(self, X_tokens, y_true):
        from sklearn_crfsuite import metrics as crf_metrics
        y_pred = self.predict(X_tokens)
        labels = [l for l in LABELS if l != "O"]
        kw = dict(labels=labels, average="weighted", zero_division=0)
        span = span_scores(y_true, y_pred)
        return {
            "report": crf_metrics.flat_classification_report(y_true, y_pred, labels=labels,
                                                             digits=3, zero_division=0),
            "f1": crf_metrics.flat_f1_score(y_true, y_pred, **kw),
            "precision": crf_metrics.flat_precision_score(y_true, y_pred, **kw),
            "recall": crf_metrics.flat_recall_score(y_true, y_pred, **kw),
            "span_precision": span["precision"], "span_recall": span["recall"],
            "span_f1": span["f1"], "span_per_type": span["per_type"],
        }

    # ------------------------------------------------------------------ save / load
    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)

    @classmethod
    def load(cls, directory: str):
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        inst = cls(model_name=directory, verbose=False)
        inst.tokenizer = AutoTokenizer.from_pretrained(directory, add_prefix_space=True)
        inst.model = AutoModelForTokenClassification.from_pretrained(directory).eval()
        inst._fitted = True
        return inst

    def __getstate__(self):
        """Pickled bundles keep only a pointer to the saved weights directory."""
        state = self.__dict__.copy()
        if self.model is not None:
            state["_weights_dir"] = getattr(self, "_weights_dir", None)
            if state["_weights_dir"] is None:
                raise RuntimeError("Call .save(directory) and set ._weights_dir before pickling.")
        state["model"] = state["tokenizer"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        path = state.get("_weights_dir")
        if path:
            if not os.path.isabs(path):   # stored relative to the repo root
                path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__)))), path)
            loaded = TransformerTagger.load(path)
            self.model, self.tokenizer = loaded.model, loaded.tokenizer

"""
model.py (prediction)
======================
Escalation forecasters. All share one interface:

    model.fit(windows_df)              # rows with eligible == True are used as targets
    model.predict_proba(windows_df)    # P(escalation onset in the next `horizon` days), one per row

Models compared in the pipeline:
  * "Volume baseline"     - logistic regression on post-volume features only.
                            Answers: does event extraction add anything over
                            simply counting how much people are posting?
  * "Logistic regression" - all window + trend features, linear.
  * "Gradient boosting"   - all window + trend features, non-linear trees.
  * "GRU"                 - a recurrent network (NumPy, full backpropagation
                            through time, Adam) that reads each region's
                            sequence of windows and learns the temporal
                            pattern itself instead of from hand-made trends.
"""
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .aggregate import BASE_FEATURES, add_trend_features, feature_columns  # noqa: F401

VOLUME_FEATURES = ["n_posts", "n_posts_delta", "n_posts_roll2"]


class TabularModel:
    """A scikit-learn classifier over a fixed list of window feature columns."""

    def __init__(self, estimator, feature_cols=None):
        self.estimator = estimator
        self.feature_cols = feature_cols

    def fit(self, df: pd.DataFrame):
        if self.feature_cols is None:
            self.feature_cols = feature_columns(df)
        train = df[df["eligible"]] if "eligible" in df else df
        self.estimator.fit(train[self.feature_cols].values, train["label"].values)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(df[self.feature_cols].values)[:, 1]

    def feature_importances(self):
        est = self.estimator[-1] if hasattr(self.estimator, "steps") else self.estimator
        if hasattr(est, "feature_importances_"):
            vals = est.feature_importances_
        elif hasattr(est, "coef_"):
            vals = np.abs(est.coef_[0])
        else:
            return []
        return sorted(zip(self.feature_cols, vals), key=lambda t: -t[1])


def EscalationClassifier(**gbc_kwargs):
    """Gradient-boosted trees (kept under its old name for compatibility)."""
    return TabularModel(GradientBoostingClassifier(
        n_estimators=gbc_kwargs.pop("n_estimators", 200),
        max_depth=gbc_kwargs.pop("max_depth", 3),
        learning_rate=gbc_kwargs.pop("learning_rate", 0.05),
        subsample=gbc_kwargs.pop("subsample", 0.8),
        random_state=42, **gbc_kwargs))


class GRUForecaster:
    """Single-layer GRU over each region's window sequence, sigmoid output at
    every step, trained with full backpropagation through time and Adam on a
    class-weighted, masked binary cross-entropy (ineligible steps -- after
    onset -- are read but never scored).

    h_t = (1 - z) * h_{t-1} + z * h~,  z = sig(Wz[h;x]+bz), r = sig(Wr[h;x]+br),
    h~ = tanh(Wh[r*h; x] + bh),        y_t = sig(Wo h_t + bo)
    """

    def __init__(self, feature_cols=None, hidden_dim=16, lr=0.01, epochs=300,
                 l2=1e-4, seed=42):
        self.feature_cols = feature_cols or list(BASE_FEATURES)
        self.H, self.lr, self.epochs, self.l2, self.seed = hidden_dim, lr, epochs, l2, seed
        self.losses: List[float] = []

    # ---- data shaping -------------------------------------------------------
    def _sequences(self, df):
        df = df.sort_values(["region", "window_end_day"])
        regions = list(df["region"].unique())
        steps = sorted(df["window_end_day"].unique())
        T, B, D = len(steps), len(regions), len(self.feature_cols)
        X = np.zeros((T, D, B))
        y = np.zeros((T, 1, B))
        m = np.zeros((T, 1, B))
        pos = {}
        r_idx = {r: i for i, r in enumerate(regions)}
        t_idx = {s: i for i, s in enumerate(steps)}
        feats = df[self.feature_cols].values
        for row, (label_idx, r, s) in enumerate(zip(df.index, df["region"], df["window_end_day"])):
            t, b = t_idx[s], r_idx[r]
            X[t, :, b] = feats[row]
            y[t, 0, b] = df.at[label_idx, "label"] if "label" in df else 0
            m[t, 0, b] = 1.0 if ("eligible" not in df or df.at[label_idx, "eligible"]) else 0.0
            pos[label_idx] = (t, b)
        return X, y, m, pos

    # ---- parameters -----------------------------------------------------------
    def _init(self, D):
        rng = np.random.RandomState(self.seed)
        H = self.H
        s1, s2 = 1 / np.sqrt(H + D), 1 / np.sqrt(H)
        self.p = {
            "Wz": rng.randn(H, H + D) * s1, "bz": np.zeros((H, 1)),
            "Wr": rng.randn(H, H + D) * s1, "br": np.zeros((H, 1)),
            "Wh": rng.randn(H, H + D) * s1, "bh": np.zeros((H, 1)),
            "Wo": rng.randn(1, H) * s2, "bo": np.zeros((1, 1)),
        }

    @staticmethod
    def _sig(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def _forward(self, X):
        p, H = self.p, self.H
        T, _, B = X.shape
        h = np.zeros((H, B))
        cache, out = [], np.zeros((T, 1, B))
        for t in range(T):
            x = X[t]
            c = np.vstack([h, x])
            z = self._sig(p["Wz"] @ c + p["bz"])
            r = self._sig(p["Wr"] @ c + p["br"])
            cr = np.vstack([r * h, x])
            ht = np.tanh(p["Wh"] @ cr + p["bh"])
            hn = (1 - z) * h + z * ht
            out[t] = self._sig(p["Wo"] @ hn + p["bo"])
            cache.append((h, c, z, r, cr, ht, hn))
            h = hn
        return out, cache

    def _loss_and_grads(self, X, y, m, pos_weight):
        p, H = self.p, self.H
        out, cache = self._forward(X)
        w = m * np.where(y == 1, pos_weight, 1.0)
        norm = max(w.sum(), 1e-9)
        eps = 1e-9
        loss = -(w * (y * np.log(out + eps) + (1 - y) * np.log(1 - out + eps))).sum() / norm
        g = {k: np.zeros_like(v) for k, v in p.items()}
        dh_next = np.zeros((H, X.shape[2]))
        for t in reversed(range(X.shape[0])):
            h, c, z, r, cr, ht, hn = cache[t]
            dlogit = w[t] * (out[t] - y[t]) / norm            # BCE + sigmoid
            g["Wo"] += dlogit @ hn.T
            g["bo"] += dlogit.sum(axis=1, keepdims=True)
            dh = p["Wo"].T @ dlogit + dh_next
            dz = dh * (ht - h)
            dht = dh * z
            dh_prev = dh * (1 - z)
            da_h = dht * (1 - ht ** 2)
            g["Wh"] += da_h @ cr.T
            g["bh"] += da_h.sum(axis=1, keepdims=True)
            dcr = p["Wh"].T @ da_h
            drh = dcr[:H]
            dr = drh * h
            dh_prev += drh * r
            da_z = dz * z * (1 - z)
            da_r = dr * r * (1 - r)
            g["Wz"] += da_z @ c.T
            g["bz"] += da_z.sum(axis=1, keepdims=True)
            g["Wr"] += da_r @ c.T
            g["br"] += da_r.sum(axis=1, keepdims=True)
            dc = p["Wz"].T @ da_z + p["Wr"].T @ da_r
            dh_prev += dc[:H]
            dh_next = dh_prev
        for k in ("Wz", "Wr", "Wh", "Wo"):
            loss += 0.5 * self.l2 * (p[k] ** 2).sum()
            g[k] += self.l2 * p[k]
        return loss, g

    # ---- public API -------------------------------------------------------------
    def fit(self, df: pd.DataFrame):
        X, y, m, _ = self._sequences(df)
        self.mu = X.transpose(0, 2, 1).reshape(-1, X.shape[1]).mean(axis=0)[None, :, None]
        self.sd = X.transpose(0, 2, 1).reshape(-1, X.shape[1]).std(axis=0)[None, :, None] + 1e-6
        X = (X - self.mu) / self.sd
        self._init(X.shape[1])
        n_pos = (y * m).sum()
        pos_weight = ((m.sum() - n_pos) / n_pos) if n_pos else 1.0
        mom = {k: np.zeros_like(v) for k, v in self.p.items()}
        vel = {k: np.zeros_like(v) for k, v in self.p.items()}
        b1, b2 = 0.9, 0.999
        self.losses = []
        for step in range(1, self.epochs + 1):
            loss, g = self._loss_and_grads(X, y, m, pos_weight)
            self.losses.append(float(loss))
            for k in self.p:
                gk = np.clip(g[k], -5, 5)
                mom[k] = b1 * mom[k] + (1 - b1) * gk
                vel[k] = b2 * vel[k] + (1 - b2) * gk ** 2
                mhat, vhat = mom[k] / (1 - b1 ** step), vel[k] / (1 - b2 ** step)
                self.p[k] -= self.lr * mhat / (np.sqrt(vhat) + 1e-8)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X, _, _, pos = self._sequences(df)
        out, _ = self._forward((X - self.mu) / self.sd)
        return np.array([out[pos[i][0], 0, pos[i][1]] for i in df.index])


def make_models(base_features=None, volume_features=None) -> Dict[str, object]:
    """Fresh, untrained instances of every model compared in the pipeline.
    base_features: per-window features the GRU reads (default: the 9 Stage-1
    counts); volume_features: what the baseline may use (default: post volume)."""
    return {
        "Volume baseline": TabularModel(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
            feature_cols=list(volume_features or VOLUME_FEATURES)),
        "Logistic regression": TabularModel(
            make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000,
                                                               class_weight="balanced"))),
        "Gradient boosting": EscalationClassifier(),
        "GRU": GRUForecaster(feature_cols=list(base_features) if base_features else None),
    }


# Kept for anyone importing the old name.
SequenceRNN = GRUForecaster

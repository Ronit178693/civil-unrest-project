"""
aggregate.py
============
Turns per-post event information into per-region time windows for the
escalation forecaster.

Input: a "post view" dataframe with one row per post and these columns:
    region, day, stage, actors, targets, has_trigger, has_escalation_trigger
Two post views exist:
    gold_post_view(...)       -> from the generator's true labels (upper bound)
    extracted_post_view(...)  -> from Stage 1's CRF + stage classifier output
                                  (this is the real, connected pipeline)

Forecasting set-up (no leakage)
-------------------------------
A window ending at day `e` covers days [e - window_size, e). At the start of
day e the model raises (or not) an alarm. The label is 1 if the region's
escalation ONSET falls in [e, e + horizon): "escalation starts within the
next `horizon` days". Windows with e > onset are marked `eligible = False`:
the region has already escalated, so there is nothing left to forecast and
the window would contain escalation posts (that was the old leakage).
"""
import json
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.extraction.lexicon import FLAT_TRIGGERS

ESCALATION_TRIGGER_TYPES = {"RIOT", "CLASH", "REPRESSION"}

BASE_FEATURES = ["n_posts", "n_grievance", "n_mobilization", "n_on_ground", "n_escalation",
                 "unique_actors", "unique_targets", "n_trigger_posts", "n_escalation_trigger_posts"]


def trigger_flags(trigger_phrases):
    """(has any trigger, has an escalation-type trigger) for a list of trigger strings."""
    phrases = [p.strip().lower() for p in trigger_phrases if p and p.strip()]
    has_esc = any(FLAT_TRIGGERS.get(p) in ESCALATION_TRIGGER_TYPES for p in phrases)
    return bool(phrases), has_esc


def _spans(tokens, labels, tag):
    out, cur = [], []
    for tok, lab in zip(tokens, labels):
        if lab == f"B-{tag}":
            if cur:
                out.append(" ".join(cur))
            cur = [tok]
        elif lab == f"I-{tag}" and cur:
            cur.append(tok)
        else:
            if cur:
                out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return out


def gold_post_view(events_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tok_s, lab_s in zip(events_df["tokens"], events_df["bio_labels"]):
        toks, labs = json.loads(tok_s), json.loads(lab_s)
        rows.append(trigger_flags(_spans(toks, labs, "TRIGGER")))
    flags = np.array(rows, dtype=bool).reshape(-1, 2)
    return pd.DataFrame({
        "region": events_df["region"].values,
        "day": events_df["day"].values,
        "stage": events_df["stage"].values,
        "actors": events_df["actor"].str.lower().values,
        "targets": events_df["target"].str.lower().values,
        "has_trigger": flags[:, 0],
        "has_escalation_trigger": flags[:, 1],
    })


def extracted_post_view(posts_df: pd.DataFrame, frames, stages) -> pd.DataFrame:
    """posts_df needs region + day; frames = CRF event frames; stages = predicted stage per post."""
    def split(v):
        return [s.strip() for s in v.split(";")] if v else []
    flags = np.array([trigger_flags(split(f.get("trigger"))) for f in frames], dtype=bool).reshape(-1, 2)
    return pd.DataFrame({
        "region": posts_df["region"].values,
        "day": posts_df["day"].values,
        "stage": list(stages),
        "actors": [(f.get("actor") or "").lower() for f in frames],
        "targets": [(f.get("target") or "").lower() for f in frames],
        "has_trigger": flags[:, 0],
        "has_escalation_trigger": flags[:, 1],
    })


def _unique_items(series):
    items = set()
    for v in series:
        if v:
            items.update(s.strip() for s in v.split(";") if s.strip())
    return len(items)


def build_windows(post_view: pd.DataFrame, region_onset_day: Dict[str, Optional[int]],
                  n_days: int, window_size: int = 3, horizon: int = 3) -> pd.DataFrame:
    """One row per (region, window end day e). With horizon=0 (forecasting
    unlabeled new data) windows run up to e = n_days and labels are all 0."""
    rows = []
    for region in sorted(region_onset_day):
        reg = post_view[post_view["region"] == region]
        onset = region_onset_day[region]
        for e in range(window_size, n_days - horizon + 1):
            w = reg[(reg["day"] >= e - window_size) & (reg["day"] < e)]
            stage_counts = w["stage"].value_counts()
            rows.append({
                "region": region,
                "window_start_day": e - window_size,
                "window_end_day": e,
                "n_posts": len(w),
                "n_grievance": int(stage_counts.get("grievance", 0)),
                "n_mobilization": int(stage_counts.get("mobilization", 0)),
                "n_on_ground": int(stage_counts.get("on_ground", 0)),
                "n_escalation": int(stage_counts.get("escalation", 0)),
                "unique_actors": _unique_items(w["actors"]),
                "unique_targets": _unique_items(w["targets"]),
                "n_trigger_posts": int(w["has_trigger"].sum()),
                "n_escalation_trigger_posts": int(w["has_escalation_trigger"].sum()),
                "label": int(onset is not None and e <= onset < e + horizon),
                "eligible": onset is None or e <= onset,
                "onset_day": onset,
            })
    return pd.DataFrame(rows)


def add_trend_features(windows_df: pd.DataFrame, lookback: int = 2) -> pd.DataFrame:
    """Per region: change vs the previous window and a rolling mean, so the
    model sees momentum (a rising mobilization count), not only levels."""
    df = windows_df.sort_values(["region", "window_end_day"]).copy()
    g = df.groupby("region")
    for col in BASE_FEATURES:
        df[f"{col}_delta"] = g[col].diff().fillna(0)
        df[f"{col}_roll{lookback}"] = g[col].transform(
            lambda s: s.rolling(lookback, min_periods=1).mean())
    posts = df["n_posts"].replace(0, np.nan)
    df["mobilization_ratio"] = (df["n_mobilization"] / posts).fillna(0)
    df["on_ground_ratio"] = (df["n_on_ground"] / posts).fillna(0)
    df["escalation_ratio"] = (df["n_escalation"] / posts).fillna(0)
    return df.reset_index(drop=True)


NON_FEATURE_COLS = {"region", "window_start_day", "window_end_day", "label", "eligible", "onset_day"}


def feature_columns(df: pd.DataFrame):
    return [c for c in df.columns if c not in NON_FEATURE_COLS]

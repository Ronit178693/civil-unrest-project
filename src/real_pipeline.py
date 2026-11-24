"""
real_pipeline.py
================
The forecasting experiment on REAL data: GDELT-coded protest events in
India, May-October 2026 (fetch with data/real/fetch_gdelt.py).

Text      Each GDELT event cites a news article. The article's headline is
          recovered from its URL slug ("farmers-block-highway-in-ambala-..."
          -> "farmers block highway in ambala"). Every distinct article is
          one "post"; Stage 1 (the trained extractor + stage classifier)
          runs on these headlines exactly as on social-media posts.
Regions   Indian states/UTs (GDELT ActionGeo ADM1 code).
Truth     GDELT's own CAMEO coding, which our models never see as input:
          a state ESCALATES on a day when at least `--min-violent` distinct
          articles report a violent protest or riot (CAMEO 145x) there, after
          at least `--cooldown` calm days. That is the escalation onset.
Task      Same as the synthetic experiment: at the start of day e, using the
          headlines of days e-3..e-1, will a new escalation start in a state
          within the next 3 days? Windows that already contain an
          escalation day are not forecast (see build_windows_multi).
Split     By time, not at random: train on May-August, test on
          September-October. This is the honest set-up for a forecaster.
Features  (a) "extracted": Stage-1 counts from headlines (our pipeline)
          (b) "gdelt_coded": GDELT's CAMEO counts per window (an alternative
              structured input, for comparison)
          plus the volume baseline (number of protest articles only).

Run:  python -m src.real_pipeline            (needs results/models/ from src.pipeline)
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.evaluation.metrics import prediction_metrics  # noqa: E402
from src.prediction.aggregate import (add_trend_features, build_windows_multi,  # noqa: E402
                                      extracted_post_view)
from src.prediction.model import make_models  # noqa: E402

DATA = os.path.join(ROOT, "data", "real")
OUT = os.path.join(ROOT, "results", "real")
BUNDLE = os.path.join(ROOT, "results", "models", "pipeline.joblib")

# CAMEO 14x: 140 political dissent, 141 demonstrate/rally, 142 hunger strike,
# 143 strike/boycott, 144 obstruct/blockade, 145 protest violently/riot
CODED_FEATURES = ["n_dissent", "n_demonstrate", "n_hunger_strike", "n_strike_boycott",
                  "n_blockade", "n_violent", "n_protest_events"]
CAMEO_GROUP = {"140": "n_dissent", "141": "n_demonstrate", "142": "n_hunger_strike",
               "143": "n_strike_boycott", "144": "n_blockade", "145": "n_violent"}

_ID_RE = re.compile(r"\d{5,}")
_EXT_RE = re.compile(r"\.(cms|html?|php|aspx?|ece|amp)$")
SKIP_WORDS = {"articleshow", "amp", "news", "story", "article", "index", "photos", "videos",
              "liveblog", "live", "updates"}


def headline_from_url(url: str):
    """Best-guess headline from a news URL slug, or None when the URL has no
    readable slug (e.g. /article/123456)."""
    if not isinstance(url, str):
        return None
    best = []
    for seg in urlparse(url).path.split("/"):
        seg = _ID_RE.sub(" ", _EXT_RE.sub("", seg.lower()))
        words = [w for w in re.split(r"[-_+\s]+", seg) if w]
        words = [w for w in words if w.isalpha() or (len(w) <= 4 and w.isalnum()
                                                     and any(c.isalpha() for c in w))]
        if len(words) > len(best):
            best = words
    while best and best[-1] in SKIP_WORDS:
        best = best[:-1]
    return " ".join(best) if len(best) >= 4 else None


def load_gdelt(path=os.path.join(DATA, "gdelt_india_protest_events.csv")):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run: python data/real/fetch_gdelt.py")
    ev = pd.read_csv(path, dtype=str)
    ev = ev[ev["adm1"].notna() & (ev["adm1"].str.len() > 2)]        # drop country-level "IN"
    ev["place_state"] = ev["place"].str.split(",").map(
        lambda p: p[-2].strip() if isinstance(p, list) and len(p) >= 3 else
        (p[0].strip() if isinstance(p, list) else None))
    names = ev.groupby("adm1")["place_state"].agg(lambda s: s.value_counts().index[0])
    ev["region_name"] = ev["adm1"].map(names)
    ev["headline"] = ev["url"].map(headline_from_url)
    return ev


def build_articles(ev: pd.DataFrame, start: pd.Timestamp):
    """One row per (state, day, article). An article that GDELT coded as
    several events keeps the most severe code (145 > 14x)."""
    ev = ev.copy()
    ev["day"] = (pd.to_datetime(ev["date"]) - start).dt.days
    ev["violent"] = ev["event_code"].str.startswith("145")
    ev["group"] = ev["event_code"].str[:3].map(CAMEO_GROUP).fillna("n_dissent")
    art = (ev.sort_values("violent", ascending=False)
             .drop_duplicates(["adm1", "day", "url"]))
    return art


def find_onsets(art: pd.DataFrame, regions, n_days, min_violent, cooldown):
    """Onset = first day of a violent spell: >= min_violent distinct articles
    report violent protest that day, and none of the previous `cooldown`
    days did. Hot days = every day with >= min_violent violent articles."""
    v = art[art["violent"]].groupby(["adm1", "day"]).size()
    onsets, hot = {}, {}
    for r in regions:
        days = sorted(d for (rr, d), n in v.items() if rr == r and n >= min_violent)
        hot[r] = set(days)
        onsets[r] = [d for d in days if not any((d - k) in hot[r] for k in range(1, cooldown + 1))]
        onsets[r] = [d for d in onsets[r] if 0 <= d < n_days]
    return onsets, hot


def coded_windows(art, regions, onsets, hot, n_days, window=3, horizon=3):
    """GDELT CAMEO counts per window, with the same labels as the text windows."""
    counts = art.groupby(["adm1", "day", "group"]).size().unstack(fill_value=0)
    rows = []
    for r in sorted(regions):
        hot_r = hot.get(r, set())
        for e in range(window, n_days - horizon + 1):
            row = {c: 0 for c in CODED_FEATURES}
            for d in range(e - window, e):
                if (r, d) in counts.index:
                    for c, n in counts.loc[(r, d)].items():
                        row[c] += int(n)
            row["n_protest_events"] = sum(row[c] for c in CODED_FEATURES[:-1])
            nxt = next((o for o in onsets[r] if o >= e), None)
            rows.append({"region": r, "window_start_day": e - window, "window_end_day": e, **row,
                         "label": int(nxt is not None and nxt < e + horizon),
                         "eligible": not any(d in hot_r for d in range(e - window, e)),
                         "onset_day": nxt})
    return pd.DataFrame(rows)


def onset_detection(windows: pd.DataFrame, proba: np.ndarray, threshold: float, horizon: int = 3):
    """Per onset: detected if an alarm fired in one of the `horizon` eligible
    windows before it; lead time = onset day - first such alarm day."""
    df = windows.assign(proba=proba)
    rows = []
    for (r, o), g in df[df["label"] == 1].groupby(["region", "onset_day"]):
        alarms = g.loc[g["proba"] >= threshold, "window_end_day"]
        rows.append({"region": r, "onset_day": int(o), "detected": len(alarms) > 0,
                     "lead_days": int(o - alarms.min()) if len(alarms) else None})
    out = pd.DataFrame(rows)
    if out.empty:
        return out, {"n_onsets": 0, "detection_rate": None, "mean_lead_days": None}
    det = out[out["detected"]]
    return out, {"n_onsets": int(len(out)), "detection_rate": float(out["detected"].mean()),
                 "mean_lead_days": float(det["lead_days"].mean()) if len(det) else None}


def evaluate_tables(tables, split_day, threshold, horizon=3):
    rows, preds = [], []
    for source, (w, base) in tables.items():
        train = w[w["window_end_day"] + horizon <= split_day]     # labels end before the split
        test_mask = (w["window_end_day"] >= split_day) & w["eligible"]
        vol = ["n_posts", "n_posts_delta", "n_posts_roll2"] if source == "extracted" else \
              ["n_protest_events", "n_protest_events_delta", "n_protest_events_roll2"]
        for name, model in make_models(base_features=base, volume_features=vol).items():
            if source != "extracted" and name == "Volume baseline":
                continue          # one volume baseline is enough (it reads the same counts)
            model.fit(train)
            p = model.predict_proba(w)[test_mask.values]
            te = w[test_mask]
            y = te["label"].values
            m = prediction_metrics(y, (p >= threshold).astype(int), p)
            _, det = onset_detection(te, p, threshold, horizon)
            rows.append({"features": source, "model": name, "precision": m["precision"],
                         "recall": m["recall"], "f1": m["f1"], "roc_auc": m.get("roc_auc"),
                         "avg_precision": m.get("avg_precision"),
                         "confusion_matrix": m["confusion_matrix"], **det})
            preds.append(te[["region", "window_end_day", "onset_day", "label"]]
                         .assign(features=source, model=name, proba=p))
    return pd.DataFrame(rows), pd.concat(preds)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Escalation forecasting on real GDELT data")
    ap.add_argument("--start", default="2026-05-01")
    ap.add_argument("--end", default="2026-10-31")
    ap.add_argument("--split", default="2026-09-01", help="first day of the test period")
    ap.add_argument("--min-violent", type=int, default=2,
                    help="violent-protest articles in a state-day that count as escalation")
    ap.add_argument("--cooldown", type=int, default=7, help="calm days needed before a new onset")
    ap.add_argument("--min-articles", type=int, default=60,
                    help="drop states with fewer protest articles in the whole period")
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    start, end = pd.Timestamp(a.start), pd.Timestamp(a.end)
    n_days = (end - start).days + 1
    split_day = (pd.Timestamp(a.split) - start).days
    ev = load_gdelt()
    ev = ev[(ev["date"] >= a.start) & (ev["date"] <= a.end)]
    art = build_articles(ev, start)
    per_state = art.groupby("adm1").size()
    regions = sorted(per_state[per_state >= a.min_articles].index)
    art = art[art["adm1"].isin(regions)]
    names = art.groupby("adm1")["region_name"].first().to_dict()
    onsets, hot = find_onsets(art, regions, n_days, a.min_violent, a.cooldown)
    n_on = sum(len(v) for v in onsets.values())
    print(f"GDELT {a.start}..{a.end}: {len(ev)} protest events, {len(art)} state-day articles, "
          f"{len(regions)} states, {n_on} escalation onsets "
          f"(>= {a.min_violent} violent articles/day after {a.cooldown} calm days)")

    # ---- Stage 1 on the headlines
    posts = art[art["headline"].notna()].rename(columns={"adm1": "region"})
    bundle = joblib.load(BUNDLE)
    t1 = time.time()
    uniq = posts["headline"].drop_duplicates().tolist()
    frames = dict(zip(uniq, bundle["tagger"].extract_many(uniq)))
    stages = dict(zip(uniq, bundle["stage_clf"].predict(uniq)))
    print(f"Stage 1 ({bundle.get('tagger_name', 'tagger')}) on {len(uniq)} unique headlines "
          f"({time.time() - t1:.0f}s)")
    view = extracted_post_view(posts, [frames[h] for h in posts["headline"]],
                               [stages[h] for h in posts["headline"]])

    ext = add_trend_features(build_windows_multi(view, onsets, hot, n_days))
    cod = add_trend_features(coded_windows(art, regions, onsets, hot, n_days),
                             base_features=CODED_FEATURES)
    for w in (ext, cod):
        w["region_name"] = w["region"].map(names)
        w["window_end_date"] = (start + pd.to_timedelta(w["window_end_day"], unit="D")).dt.date
        w["split"] = np.where(w["window_end_day"] >= split_day, "test", "train")
    elig = ext[ext["eligible"]]
    print(f"{len(elig)} forecastable state-windows | {int(elig['label'].sum())} positive "
          f"({elig['label'].mean():.1%}) | train {int((elig['split'] == 'train').sum())}, "
          f"test {int((elig['split'] == 'test').sum())}")

    table, preds = evaluate_tables({"extracted": (ext, None), "gdelt_coded": (cod, CODED_FEATURES)},
                                   split_day, a.threshold)
    table.to_csv(os.path.join(OUT, "real_model_comparison.csv"), index=False)
    preds.to_csv(os.path.join(OUT, "real_test_predictions.csv"), index=False)
    ext.to_csv(os.path.join(OUT, "real_windows_extracted.csv"), index=False)
    cod.to_csv(os.path.join(OUT, "real_windows_gdelt_coded.csv"), index=False)

    stage_dist = pd.Series(list(stages.values())).value_counts().to_dict()
    filled = {k: float(np.mean([bool(f.get(k)) for f in frames.values()]))
              for k in ("trigger", "actor", "location", "target", "time", "scale")}
    test_elig = elig[elig["split"] == "test"]
    summary = {
        "args": vars(a), "n_events": int(len(ev)), "n_articles": int(len(art)),
        "n_headlines": int(len(uniq)), "n_states": len(regions),
        "states": {r: names.get(r) for r in regions},
        "n_onsets": n_on, "onsets_by_state": {names.get(r, r): len(v) for r, v in onsets.items() if v},
        "n_windows": int(len(elig)), "n_positive": int(elig["label"].sum()),
        "n_test_windows": int(len(test_elig)), "n_test_positive": int(test_elig["label"].sum()),
        "headline_stage_distribution": stage_dist, "headline_role_fill_rate": filled,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(OUT, "real_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    show = table.drop(columns=["confusion_matrix"]).copy()
    for c in ["precision", "recall", "f1", "roc_auc", "avg_precision", "detection_rate"]:
        show[c] = show[c].map(lambda v: f"{v:.3f}" if v is not None and not pd.isna(v) else "-")
    print(f"\nTest period from {a.split} (threshold {a.threshold}):")
    print(show.to_string(index=False))
    print(f"\n=== DONE in {time.time() - t0:.0f}s. Results in results/real/ ===")


if __name__ == "__main__":
    main()

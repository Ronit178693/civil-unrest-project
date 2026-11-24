"""
pipeline.py
===========
End-to-end run:

  1. Generate the synthetic dataset (several independent episodes).
  2. Stage 1 -- train four extractors and compare them:
       CRF / fine-tuned transformer (twitter-roberta-base), each trained on
       (a) synthetic episode 0 only, or (b) synthetic + the training half of
       the hand-annotated REAL posts (Mastodon + GDELT headlines) = the
       domain-adapted model. Same for the post-stage classifier.
  3. Evaluate Stage 1 on synthetic seen / unseen templates, the hand-written
     challenge set and the held-out half of the real annotated posts.
  4. Run Stage 1 on every post of the forecasting regions (episodes 1..N-1)
     and aggregate its OUTPUT into region windows. (Gold-label windows are
     built too, as an upper bound.)
  5. Stage 2 -- compare four forecasters with region-grouped K-fold
     cross-validation; report window metrics and early-warning metrics.
  6. Train the best forecaster on all forecasting regions and save every
     model to results/models/ for demo.py.

Run:  python -m src.pipeline            (~20-30 min on a laptop CPU: two transformer fine-tunes)
      python -m src.pipeline --tagger crf  (CRF only, ~2 minutes)
      python -m src.pipeline --quick    (CRF only, smaller; ~30 seconds, for checking)
Then: python -m src.real_pipeline       (the real-data forecasting experiment)
"""
import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.generate_synthetic_data import build_dataset  # noqa: E402
from src.extraction.model import UnrestCRFTagger  # noqa: E402
from src.extraction.tokenize import tokenize  # noqa: E402
from src.extraction.stage import StageClassifier  # noqa: E402
from src.prediction.aggregate import (add_trend_features, build_windows,  # noqa: E402
                                      extracted_post_view, gold_post_view)
from src.prediction.model import make_models  # noqa: E402
from src.evaluation.metrics import (lead_time_analysis, prediction_metrics,  # noqa: E402
                                    summarize_lead_time)

RESULTS_DIR = os.path.join(ROOT, "results")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")


def _save_json(obj, name):
    with open(os.path.join(RESULTS_DIR, name), "w") as f:
        json.dump(obj, f, indent=2, default=float)


ROLES = ("trigger", "actor", "location", "target", "time", "scale")
REAL_PATH = os.path.join(ROOT, "data", "real", "annotated_real.json")


def labelled_to_bio(item):
    """{text, trigger: [...], actor: [...], ...} -> (tokens, BIO labels).
    Each listed phrase is matched to its first occurrence that is not
    labelled yet, so a phrase listed twice labels two occurrences."""
    toks = tokenize(item["text"])
    lower = [t.lower() for t in toks]
    labels = ["O"] * len(toks)
    for role in ROLES:
        for phrase in item.get(role, []):
            span = [t.lower() for t in tokenize(phrase)]
            for i in range(len(toks) - len(span) + 1):
                if lower[i:i + len(span)] == span and all(l == "O" for l in labels[i:i + len(span)]):
                    labels[i] = f"B-{role.upper()}"
                    for j in range(1, len(span)):
                        labels[i + j] = f"I-{role.upper()}"
                    break
            else:
                raise ValueError(f"{role} '{phrase}' not found (or overlaps) in: {item['text']}")
    return toks, labels


def load_labelled(path):
    rows = []
    for item in json.load(open(path, encoding="utf-8")):
        toks, labels = labelled_to_bio(item)
        rows.append({"text": item["text"], "stage": item["stage"],
                     "source": item.get("source", "challenge"), "split": item.get("split"),
                     "tokens": json.dumps(toks), "bio_labels": json.dumps(labels)})
    return pd.DataFrame(rows)


def load_challenge_set(path=os.path.join(ROOT, "data", "challenge_set.json")):
    """Hand-written, hand-labelled posts with vocabulary the generator never
    uses (new actors, targets, cities, trigger words). Converted to the same
    tokens/BIO format as the synthetic data."""
    return load_labelled(path)


def load_real_annotated(path=REAL_PATH):
    """Real posts (Mastodon) and real news headlines (GDELT URLs), labelled by
    hand with the same roles and stages. `split` says which half trains the
    domain-adapted models and which half is held out for testing."""
    if not os.path.exists(path):
        return None
    return load_labelled(path)


REAL_OVERSAMPLE = 5   # the real training half is small next to the synthetic set
TRANSFORMER_SYNTH_CAP = 1200   # synthetic posts are template-redundant; caps CPU fine-tuning time


def run_extraction_stage(events_df, tagger_kind="transformer", epochs=3, max_eval_posts=1000,
                         seed=42):
    print("\n=== STAGE 1: Event extraction ===")
    dev = events_df[(events_df["episode"] == 0) & (~events_df["held_out_template"])]
    rest = events_df[events_df["episode"] != 0]
    seen = rest[~rest["held_out_template"]]
    unseen = rest[rest["held_out_template"]]
    seen = seen.sample(min(len(seen), max_eval_posts), random_state=seed)
    unseen = unseen.sample(min(len(unseen), max_eval_posts), random_state=seed)
    challenge = load_challenge_set()
    real = load_real_annotated()
    eval_sets = [("seen_templates", seen), ("unseen_templates", unseen), ("challenge_set", challenge)]
    adapt, real_train = dev, None
    if real is not None:
        real_train, real_test = real[real["split"] == "train"], real[real["split"] == "test"]
        eval_sets += [("real_test", real_test),
                      ("real_test_mastodon", real_test[real_test["source"] == "mastodon"]),
                      ("real_test_headlines", real_test[real_test["source"] == "gdelt"])]
        adapt = pd.concat([dev] + [real_train] * REAL_OVERSAMPLE, ignore_index=True)
        print(f"Real annotated posts: {len(real_train)} train (x{REAL_OVERSAMPLE} when adapting) | "
              f"{len(real_test)} test")
    print(f"Synthetic train: {len(dev)} posts (episode 0, training templates) | "
          f"test: {len(seen)} seen-template + {len(unseen)} unseen-template posts + "
          f"{len(challenge)} challenge posts")

    tok = lambda d: d["tokens"].apply(json.loads).tolist()  # noqa: E731
    lab = lambda d: d["bio_labels"].apply(json.loads).tolist()  # noqa: E731
    variants = [("CRF", "synthetic", dev), ("CRF", "adapted", adapt)]
    if tagger_kind == "transformer":
        dev_t = dev.sample(min(len(dev), TRANSFORMER_SYNTH_CAP), random_state=seed)
        adapt_t = dev_t if real_train is None else             pd.concat([dev_t] + [real_train] * REAL_OVERSAMPLE, ignore_index=True)
        variants += [("Transformer", "synthetic", dev_t), ("Transformer", "adapted", adapt_t)]
    if real is None:
        variants = [v for v in variants if v[1] == "synthetic"]

    taggers, results = {}, {}
    for kind, train_name, train_df in variants:
        name = f"{kind} ({train_name})"
        t0 = time.time()
        print(f"  training {name} on {len(train_df)} posts ...", flush=True)
        if kind == "CRF":
            taggers[name] = UnrestCRFTagger().fit(tok(train_df), lab(train_df))
        else:
            from src.extraction.transformer_tagger import TransformerTagger
            taggers[name] = TransformerTagger(epochs=epochs, seed=seed).fit(tok(train_df),
                                                                            lab(train_df))
        results[name] = {"train_seconds": round(time.time() - t0, 1)}
        for set_name, d in eval_sets:
            if len(d) == 0:
                continue
            m = taggers[name].evaluate(tok(d), lab(d))
            results[name][set_name] = {
                "n_posts": int(len(d)), "token_f1": m["f1"], "span_f1": m["span_f1"],
                "span_precision": m["span_precision"], "span_recall": m["span_recall"],
                "span_per_type": m["span_per_type"]}
            with open(os.path.join(RESULTS_DIR, f"extraction_report_{set_name}.txt"), "a") as f:
                f.write(f"=== {name} on {set_name}\n{m['report']}\n")
        print("    " + " | ".join(f"{k} {v['span_f1']:.3f}" for k, v in results[name].items()
                                   if isinstance(v, dict)), flush=True)

    stage = {"synthetic": StageClassifier().fit(dev["text"], dev["stage"]),
             "adapted": StageClassifier().fit(adapt["text"], adapt["stage"])}
    stage_results = {}
    for train_name, clf in stage.items():
        stage_results[train_name] = {}
        for set_name, d in eval_sets:
            if len(d):
                s = clf.evaluate(d["text"], d["stage"])
                stage_results[train_name][set_name] = {"accuracy": s["accuracy"],
                                                       "macro_f1": s["macro_f1"]}
        print(f"  Stage classifier ({train_name}): " + " | ".join(
            f"{k} {v['accuracy']:.3f}" for k, v in stage_results[train_name].items()))

    final_name = ("Transformer" if tagger_kind == "transformer" else "CRF") + \
        (" (adapted)" if real is not None else " (synthetic)")
    tagger = taggers[final_name]
    stage_clf = stage["adapted" if real is not None else "synthetic"]
    out = {"taggers": results, "stage_classifier": stage_results, "final_tagger": final_name}
    _save_json(out, "extraction_metrics.json")

    print(f"\nFinal extractor: {final_name}. Sample extractions (not in any training set):")
    for post in ["Thousands of farmers are marching towards Jaipur to protest the new farm laws.",
                 "BREAKING: workers planning a strike outside the factory in Noida tomorrow.",
                 "Police used tear gas as the protest in Lucknow turned into a riot last night."]:
        print(f"  {post}\n    -> {tagger.extract_events(post)} | stage: {stage_clf.predict([post])[0]}")
    return tagger, stage_clf, out, final_name


def build_feature_tables(events_df, onset, n_days, tagger, stage_clf):
    fc = events_df[events_df["episode"] != 0].reset_index(drop=True)
    fc_onset = {r: d for r, d in onset.items() if r in set(fc["region"])}
    t0 = time.time()
    frames = tagger.extract_many(fc["text"].tolist())
    stages = stage_clf.predict(fc["text"])
    print(f"\nRan Stage 1 on {len(fc)} forecasting-region posts ({time.time() - t0:.1f}s)")
    tables = {}
    for source, view in [("extracted", extracted_post_view(fc, frames, stages)),
                         ("gold", gold_post_view(fc))]:
        tables[source] = add_trend_features(build_windows(view, fc_onset, n_days))
    return tables


def cross_validate(windows, n_folds, threshold):
    """Region-grouped K-fold. Returns out-of-fold probabilities per model."""
    regions = windows["region"].values
    elig = windows["eligible"].values
    oof = {name: np.full(len(windows), np.nan) for name in make_models()}
    fold_scores = {name: [] for name in oof}
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=n_folds).split(windows, groups=regions)):
        train_df, test_df = windows.iloc[tr], windows.iloc[te]
        for name, model in make_models().items():
            model.fit(train_df)
            proba = model.predict_proba(test_df)
            oof[name][te] = proba
            te_e = elig[te]
            y = test_df["label"].values[te_e]
            m = prediction_metrics(y, (proba[te_e] >= threshold).astype(int), proba[te_e])
            fold_scores[name].append(m)
    return oof, fold_scores


def run_prediction_stage(tables, n_folds, threshold):
    print("\n=== STAGE 2: Escalation forecasting (region-grouped "
          f"{n_folds}-fold cross-validation) ===")
    w = tables["extracted"]
    elig = w[w["eligible"]]
    print(f"{w['region'].nunique()} regions | {len(elig)} forecastable windows | "
          f"{int(elig['label'].sum())} positive ({elig['label'].mean():.1%})")

    rows, oof_frames, lead_frames = [], [], []
    for source, windows in tables.items():
        oof, folds = cross_validate(windows, n_folds, threshold)
        e = windows["eligible"].values
        y = windows["label"].values[e]
        for name, proba in oof.items():
            p = proba[e]
            m = prediction_metrics(y, (p >= threshold).astype(int), p)
            lead_df = lead_time_analysis(windows[e], p, threshold)
            lt = summarize_lead_time(lead_df)
            f1s = [f["f1"] for f in folds[name]]
            aps = [f.get("avg_precision", np.nan) for f in folds[name]]
            rows.append({
                "features": source, "model": name,
                "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                "f1_fold_std": float(np.std(f1s)),
                "roc_auc": m.get("roc_auc"), "avg_precision": m.get("avg_precision"),
                "ap_fold_std": float(np.nanstd(aps)),
                "confusion_matrix": m["confusion_matrix"],
                "detection_rate": lt["detection_rate"],
                "mean_lead_days": lt["mean_lead_time_days"],
                "false_alarm_rate": lt["false_alarm_rate"],
            })
            oof_frames.append(windows[e][["region", "window_end_day", "onset_day", "label"]]
                              .assign(features=source, model=name, proba=p))
            lead_frames.append(lead_df.assign(features=source, model=name))

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
    pd.concat(oof_frames).to_csv(os.path.join(RESULTS_DIR, "oof_predictions.csv"), index=False)
    pd.concat(lead_frames).to_csv(os.path.join(RESULTS_DIR, "lead_time_by_region.csv"), index=False)

    show = table.copy()
    for c in ["precision", "recall", "f1", "roc_auc", "avg_precision", "detection_rate",
              "false_alarm_rate"]:
        show[c] = show[c].map(lambda v: f"{v:.3f}" if v is not None and not pd.isna(v) else "-")
    show["f1"] = show["f1"] + " +/-" + table["f1_fold_std"].map(lambda v: f"{v:.2f}")
    show["mean_lead_days"] = table["mean_lead_days"].map(
        lambda v: f"{v:.1f}" if v is not None and not pd.isna(v) else "-")
    print(f"\nThreshold = {threshold}. F1 '+/-' = std across folds. Lead/false-alarm are per region.")
    print(show[["features", "model", "precision", "recall", "f1", "roc_auc", "avg_precision",
                "detection_rate", "mean_lead_days", "false_alarm_rate"]].to_string(index=False))
    return table


def train_final_models(tables, table, tagger, stage_clf, threshold, tagger_name="CRF"):
    ext = table[table["features"] == "extracted"].sort_values("avg_precision", ascending=False)
    best_name = ext.iloc[0]["model"]
    best = make_models()[best_name].fit(tables["extracted"])
    gbt = make_models()["Gradient boosting"].fit(tables["extracted"])
    importances = [(f, float(v)) for f, v in gbt.feature_importances()]
    _save_json(importances, "feature_importances.json")
    os.makedirs(MODELS_DIR, exist_ok=True)
    if hasattr(tagger, "save") and tagger.__class__.__name__ == "TransformerTagger":
        tagger.save(os.path.join(MODELS_DIR, "transformer_tagger"))
        tagger._weights_dir = os.path.join("results", "models", "transformer_tagger")
    joblib.dump({"tagger": tagger, "stage_clf": stage_clf, "forecaster": best,
                 "forecaster_name": best_name, "threshold": threshold,
                 "tagger_name": tagger_name},
                os.path.join(MODELS_DIR, "pipeline.joblib"))
    print(f"\nBest forecaster on extracted features (by average precision): {best_name}")
    print("Top features (gradient boosting):")
    for f, v in importances[:6]:
        print(f"  {f:32s} {v:.3f}")
    print(f"Saved trained models to {os.path.relpath(MODELS_DIR, ROOT)}{os.sep}pipeline.joblib")
    return best_name


def main(argv=None):
    ap = argparse.ArgumentParser(description="Civil-unrest extraction + forecasting pipeline")
    ap.add_argument("--episodes", type=int, default=5, help="simulation episodes (>= 3)")
    ap.add_argument("--folds", type=int, default=5, help="cross-validation folds")
    ap.add_argument("--threshold", type=float, default=0.5, help="alarm probability threshold")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tagger", choices=["transformer", "crf"], default="transformer",
                    help="final extractor (transformer also trains the CRF for comparison)")
    ap.add_argument("--epochs", type=int, default=3, help="transformer fine-tuning epochs")
    ap.add_argument("--quick", action="store_true", help="3 episodes, 3 folds, CRF (fast check)")
    args = ap.parse_args(argv)
    if args.quick:
        args.episodes, args.folds, args.tagger = 3, 3, "crf"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()
    print("=== Generating synthetic civil-unrest social-media dataset ===", flush=True)
    events_df, onset, start_date, n_days = build_dataset(n_episodes=args.episodes, seed=args.seed)
    events_df.to_csv(os.path.join(RESULTS_DIR, "events.csv"), index=False)
    n_esc = sum(v is not None for v in onset.values())
    print(f"{len(events_df)} posts | {len(onset)} regions ({args.episodes} episodes x 16 cities) | "
          f"{n_esc} regions escalate")

    for f in os.listdir(RESULTS_DIR):          # reports are appended per model
        if f.startswith("extraction_report_"):
            os.remove(os.path.join(RESULTS_DIR, f))
    tagger, stage_clf, ext_results, tagger_name = run_extraction_stage(
        events_df, tagger_kind=args.tagger, epochs=args.epochs, seed=args.seed)
    tables = build_feature_tables(events_df, onset, n_days, tagger, stage_clf)
    tables["extracted"].to_csv(os.path.join(RESULTS_DIR, "windows_extracted.csv"), index=False)
    tables["gold"].to_csv(os.path.join(RESULTS_DIR, "windows_gold.csv"), index=False)
    table = run_prediction_stage(tables, args.folds, args.threshold)
    best = train_final_models(tables, table, tagger, stage_clf, args.threshold, tagger_name)

    _save_json({"args": vars(args), "n_posts": int(len(events_df)), "n_regions": len(onset),
                "extraction": ext_results, "best_forecaster": best,
                "runtime_seconds": round(time.time() - t0, 1)}, "run_summary.json")
    print(f"\n=== DONE in {time.time() - t0:.0f}s. Results written to results/ ===")


if __name__ == "__main__":
    main()

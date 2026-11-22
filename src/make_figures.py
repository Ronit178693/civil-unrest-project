"""
make_figures.py
===============
Figures for the report/slides, drawn from the files a pipeline run saves in
results/ (no retraining). Run AFTER `python -m src.pipeline`.

  fig1_event_stage_timeline.png  posts per day by stage: an escalating vs a fizzling region
  fig2_extraction_scores.png     span F1 per argument role, seen vs unseen templates
  fig3_feature_importance.png    top features of the gradient-boosting forecaster
  fig4_pr_curves.png             precision-recall curves of all forecasters (extracted features)
  fig5_lead_time.png             days of warning per escalating region (best model)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "results")
FIG_DIR = os.path.join(BASE, "paper", "figures")

plt.rcParams.update({"figure.dpi": 150, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
PALETTE = ["#2a5d8f", "#c75c3a", "#3f8f6b", "#8a6bb8", "#999999"]
STAGES = ["grievance", "mobilization", "on_ground", "escalation"]


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name))
    plt.close(fig)


def fig_event_stage_timeline():
    ev = pd.read_csv(os.path.join(RESULTS, "events.csv"))
    ev = ev[ev["episode"] == 1]
    pick = [ev.loc[ev["region_kind"] == k, "region"].iloc[0] for k in ("escalate", "fizzle")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, region, title in zip(axes, pick, ("escalates", "builds up, then fizzles")):
        sub = ev[ev["region"] == region]
        daily = (sub.groupby(["day", "stage"]).size().unstack(fill_value=0)
                 .reindex(index=range(21), columns=STAGES, fill_value=0))
        bottom = np.zeros(len(daily))
        for color, st in zip(PALETTE, STAGES):
            ax.bar(daily.index, daily[st], bottom=bottom, color=color, label=st.replace("_", " "))
            bottom += daily[st].values
        ax.set_title(f"{region.split('-')[0]}: {title}")
        ax.set_xlabel("Day")
    axes[0].set_ylabel("Posts per day")
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    _save(fig, "fig1_event_stage_timeline.png")


def fig_extraction_scores():
    m = json.load(open(os.path.join(RESULTS, "extraction_metrics.json")))
    types = sorted(m["seen_templates"]["span_per_type"])
    x = np.arange(len(types))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    splits = [("seen_templates", "Synthetic, seen templates"),
              ("unseen_templates", "Synthetic, unseen templates"),
              ("challenge_set", "Hand-labelled challenge set")]
    for k, (split, label) in enumerate(splits):
        vals = [m[split]["span_per_type"].get(t, {}).get("f1", 0.0) for t in types]
        bars = ax.bar(x + (k - 1) * 0.27, vals, 0.27, color=PALETTE[k], label=label)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), ha="center",
                        va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Span F1 (exact match)")
    ax.set_title("Event extraction (CRF) by argument role")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8)
    _save(fig, "fig2_extraction_scores.png")


def fig_feature_importance():
    imp = json.load(open(os.path.join(RESULTS, "feature_importances.json")))[:8][::-1]
    feats, vals = zip(*imp)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.barh(feats, vals, color=PALETTE[0])
    ax.set_xlabel("Relative importance")
    ax.set_title("Top forecasting features (gradient boosting)")
    _save(fig, "fig3_feature_importance.png")


def fig_pr_curves():
    oof = pd.read_csv(os.path.join(RESULTS, "oof_predictions.csv"))
    oof = oof[oof["features"] == "extracted"]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for color, (name, g) in zip(PALETTE, oof.groupby("model", sort=False)):
        p, r, _ = precision_recall_curve(g["label"], g["proba"])
        ap = average_precision_score(g["label"], g["proba"])
        ax.plot(r, p, color=color, lw=2, label=f"{name} (AP {ap:.2f})")
    ax.axhline(oof.groupby("model")["label"].mean().iloc[0], color="gray", ls="--", lw=1,
               label="chance")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.05)
    ax.set_title("Escalation forecasting: precision-recall")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    _save(fig, "fig4_pr_curves.png")


def fig_lead_time():
    """Early warning is a trade-off: alarming earlier also alarms more quiet regions."""
    t = pd.read_csv(os.path.join(RESULTS, "model_comparison.csv"))
    t = t[t["features"] == "extracted"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    y = np.arange(len(t))
    panels = [("mean_lead_days", "Mean days of warning before onset", "{:.1f} d"),
              ("false_alarm_rate", "Quiet regions given a false alarm", "{:.0%}")]
    for ax, (col, title, fmt) in zip(axes, panels):
        ax.barh(y, t[col], color=PALETTE[:len(t)])
        for yi, v in zip(y, t[col]):
            ax.annotate(fmt.format(v), (v, yi), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, t[col].max() * 1.25)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(t["model"])
    axes[0].invert_yaxis()
    _save(fig, "fig5_lead_time.png")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_event_stage_timeline()
    fig_extraction_scores()
    fig_feature_importance()
    fig_pr_curves()
    fig_lead_time()
    print("Figures written to", FIG_DIR)

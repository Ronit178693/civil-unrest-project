"""
make_figures.py
===============
Figures for the report/slides, drawn from the files a pipeline run saves in
results/ (no retraining). Run AFTER `python -m src.pipeline`.

  fig1_event_stage_timeline.png  posts per day by stage: an escalating vs a fizzling region
  fig2_extraction_scores.png     span F1 of CRF vs transformer, synthetic vs real test sets
  fig3_feature_importance.png    top features of the gradient-boosting forecaster
  fig4_pr_curves.png             precision-recall curves of all forecasters (extracted features)
  fig5_lead_time.png             days of warning per escalating region (best model)
  fig6_real_pr_curves.png        real GDELT test period: precision-recall of the forecasters
  fig7_real_timeline.png         real data: protest headlines per day and escalation onsets
                                 in the state with the most onsets
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
REAL = os.path.join(RESULTS, "real")
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
    m = json.load(open(os.path.join(RESULTS, "extraction_metrics.json")))["taggers"]
    sets = [("unseen_templates", "Synthetic,\nunseen templates"),
            ("challenge_set", "Challenge set\n(new vocabulary)"),
            ("real_test_mastodon", "Real posts\n(Mastodon)"),
            ("real_test_headlines", "Real headlines\n(GDELT)")]
    sets = [(k, lab) for k, lab in sets if any(k in v for v in m.values())]
    models = list(m)
    x = np.arange(len(sets))
    width = 0.8 / len(models)
    colors = {"CRF (synthetic)": "#b9c7d6", "CRF (adapted)": "#6f8fb0",
              "Transformer (synthetic)": "#e3b49f", "Transformer (adapted)": "#c75c3a"}
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for k, name in enumerate(models):
        vals = [m[name].get(key, {}).get("span_f1", np.nan) for key, _ in sets]
        bars = ax.bar(x + (k - (len(models) - 1) / 2) * width, vals, width,
                      color=colors.get(name, PALETTE[k % len(PALETTE)]), label=name)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), ha="center",
                            va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in sets], fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Span F1 (exact match)")
    ax.set_title("Event extraction: CRF vs fine-tuned transformer")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=8)
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


def fig_real_pr_curves():
    pred = pd.read_csv(os.path.join(REAL, "real_test_predictions.csv"))
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    styles = {"extracted": "-", "gdelt_coded": "--"}
    for color, ((feats, name), g) in zip(PALETTE * 2, pred.groupby(["features", "model"], sort=False)):
        p, r, _ = precision_recall_curve(g["label"], g["proba"])
        ap = average_precision_score(g["label"], g["proba"])
        label = f"{name}, {'headlines (ours)' if feats == 'extracted' else 'GDELT codes'} (AP {ap:.2f})"
        ax.plot(r, p, styles.get(feats, "-"), color=color, lw=1.8, label=label)
    base = pred.groupby(["features", "model"])["label"].mean().iloc[0]
    ax.axhline(base, color="gray", ls=":", lw=1, label=f"chance ({base:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.05)
    ax.set_title("Real data (GDELT India), Sep-Oct 2026 test period")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    _save(fig, "fig6_real_pr_curves.png")


def fig_real_timeline():
    w = pd.read_csv(os.path.join(REAL, "real_windows_extracted.csv"))
    summary = json.load(open(os.path.join(REAL, "real_summary.json")))
    start = pd.Timestamp(summary["args"]["start"])
    on = w.dropna(subset=["onset_day"]).groupby("region")["onset_day"].nunique()
    region = on.idxmax()
    g = w[w["region"] == region].sort_values("window_end_day")
    dates = start + pd.to_timedelta(g["window_end_day"], unit="D")
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(dates, g["n_posts"] / 3, color="#9aa7b4", lw=1, label="protest headlines / day (3-day mean)")
    ax.plot(dates, g["n_escalation"] / 3, color=PALETTE[1], lw=1.4,
            label="headlines classed 'escalation' by Stage 1")
    for i, o in enumerate(sorted(g["onset_day"].dropna().unique())):
        ax.axvline(start + pd.Timedelta(days=int(o)), color="black", ls="--", lw=0.8,
                   label="escalation onset (GDELT CAMEO 145)" if i == 0 else None)
    ax.axvline(pd.Timestamp(summary["args"]["split"]), color=PALETTE[0], lw=1.2,
               label="train / test split")
    ax.set_ylabel("Articles per day")
    ax.set_title(f"{g['region_name'].iloc[0]}: protest news and escalation onsets, 2026")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    _save(fig, "fig7_real_timeline.png")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_event_stage_timeline()
    fig_extraction_scores()
    fig_feature_importance()
    fig_pr_curves()
    fig_lead_time()
    if os.path.exists(os.path.join(REAL, "real_summary.json")):
        fig_real_pr_curves()
        fig_real_timeline()
    print("Figures written to", FIG_DIR)

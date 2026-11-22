"""
metrics.py
==========
Evaluation for the escalation forecaster.

  * Classification metrics on every forecastable window: accuracy,
    precision, recall, F1 (at a probability threshold), ROC-AUC and average
    precision (AP, the area under the precision-recall curve -- the better
    summary when positives are rare).
  * Early-warning metrics per REGION, the question the project asks
    ("how far in advance can unrest be anticipated?"):
      - detection rate: share of escalating regions that got an alarm on or
        before the onset day
      - lead time: onset day minus the day of the FIRST alarm (days of warning)
      - false-alarm rate: share of quiet regions that ever got an alarm
"""
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)


def prediction_metrics(y_true, y_pred, y_proba=None) -> Dict:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None and len(set(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, y_proba)
        out["avg_precision"] = average_precision_score(y_true, y_proba)
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return out


def lead_time_analysis(df: pd.DataFrame, y_proba: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
    """One row per region. `df` must hold eligible windows only, with columns
    region, window_end_day, onset_day. An alarm on window ending at day e is
    raised at the start of day e."""
    df = df.copy()
    df["proba"] = np.asarray(y_proba)
    rows = []
    for region, g in df.groupby("region"):
        onset = g["onset_day"].iloc[0]
        escalates = onset is not None and not pd.isna(onset)
        alarms = g.loc[g["proba"] >= threshold, "window_end_day"]
        first_alarm = int(alarms.min()) if len(alarms) else None
        rows.append({
            "region": region,
            "escalates": escalates,
            "onset_day": int(onset) if escalates else None,
            "first_alarm_day": first_alarm,
            "lead_time_days": (int(onset) - first_alarm) if (escalates and first_alarm is not None) else None,
            "n_alarm_days": int(len(alarms)),
        })
    return pd.DataFrame(rows)


def summarize_lead_time(lead_df: pd.DataFrame) -> Dict:
    esc = lead_df[lead_df["escalates"]]
    quiet = lead_df[~lead_df["escalates"]]
    detected = esc[esc["lead_time_days"].notna()]
    return {
        "n_escalating_regions": int(len(esc)),
        "n_quiet_regions": int(len(quiet)),
        "n_detected": int(len(detected)),
        "detection_rate": float(len(detected) / len(esc)) if len(esc) else 0.0,
        "mean_lead_time_days": float(detected["lead_time_days"].mean()) if len(detected) else None,
        "median_lead_time_days": float(detected["lead_time_days"].median()) if len(detected) else None,
        "false_alarm_rate": float((quiet["n_alarm_days"] > 0).mean()) if len(quiet) else 0.0,
    }

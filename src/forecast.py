"""
forecast.py
===========
Use the trained pipeline on NEW posts (yours, or real collected data).

    from src.forecast import load_bundle, extract_post, forecast_posts
    bundle = load_bundle()                       # results/models/pipeline.joblib
    extract_post(bundle, "Students plan a protest in Delhi tomorrow")
    forecast_posts(bundle, posts_df)             # posts_df: timestamp, region, text

Run `python -m src.pipeline` once first; it trains and saves the models.
"""
import os

import joblib
import pandas as pd

from src.prediction.aggregate import add_trend_features, build_windows, extracted_post_view

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_PATH = os.path.join(ROOT, "results", "models", "pipeline.joblib")


def load_bundle(path: str = BUNDLE_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained models at {path}.\nRun `python -m src.pipeline` first (takes ~1-2 minutes).")
    return joblib.load(path)


def extract_post(bundle, text: str) -> dict:
    """Event frame + predicted stage for one post."""
    frame = bundle["tagger"].extract_events(text)
    frame["stage"] = str(bundle["stage_clf"].predict([text])[0]) if text.strip() else None
    return frame


def forecast_posts(bundle, posts: pd.DataFrame, window_size: int = 3) -> pd.DataFrame:
    """Escalation risk for every region and day.

    `posts` needs columns: timestamp (or date), region, text.
    Returns one row per (region, alarm day) with the probability that
    escalation starts within the next 3 days, and whether it crosses the
    alarm threshold. The last row per region is the forecast for "now".
    """
    for col in ("region", "text"):
        if col not in posts.columns:
            raise ValueError(f"posts is missing the '{col}' column")
    time_col = "timestamp" if "timestamp" in posts.columns else "date"
    if time_col not in posts.columns:
        raise ValueError("posts needs a 'timestamp' or 'date' column")

    posts = posts.copy().reset_index(drop=True)
    posts["text"] = posts["text"].fillna("").astype(str)
    dates = pd.to_datetime(posts[time_col]).dt.normalize()
    start = dates.min()
    posts["day"] = (dates - start).dt.days
    n_days = int(posts["day"].max()) + 1
    if n_days < window_size:
        raise ValueError(f"Need at least {window_size} days of posts; got {n_days}.")

    frames = bundle["tagger"].extract_many(posts["text"].tolist())
    stages = bundle["stage_clf"].predict(posts["text"])
    view = extracted_post_view(posts, frames, stages)
    regions = {r: None for r in posts["region"].unique()}
    windows = add_trend_features(build_windows(view, regions, n_days, window_size, horizon=0))
    windows["probability"] = bundle["forecaster"].predict_proba(windows)
    windows["alarm"] = windows["probability"] >= bundle["threshold"]
    windows["alarm_date"] = (start + pd.to_timedelta(windows["window_end_day"], unit="D")).dt.date
    return windows[["region", "alarm_date", "n_posts", "n_mobilization", "n_on_ground",
                    "n_escalation", "probability", "alarm"]]

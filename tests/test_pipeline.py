"""Tests for every stage. Run:  .venv\\Scripts\\python -m pytest tests -v   (~30 s)"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_synthetic_data import _bio_tag, build_dataset  # noqa: E402
from src.evaluation.metrics import lead_time_analysis, summarize_lead_time  # noqa: E402
from src.extraction.model import UnrestCRFTagger, span_scores  # noqa: E402
from src.extraction.stage import StageClassifier  # noqa: E402
from src.extraction.tokenize import tokenize  # noqa: E402
from src.prediction.aggregate import (add_trend_features, build_windows,  # noqa: E402
                                      extracted_post_view, gold_post_view)
from src.prediction.model import GRUForecaster, make_models  # noqa: E402


@pytest.fixture(scope="module")
def small():
    events, onset, start, n_days = build_dataset(n_episodes=2, n_regions_escalate=3,
                                                 n_regions_quiet=4, seed=7)
    windows = add_trend_features(build_windows(gold_post_view(events), onset, n_days))
    return events, onset, n_days, windows


# ---------------------------------------------------------------- data + labels
def test_tokenizer_splits_punctuation_keeps_hashtags_and_hyphens():
    assert tokenize("BREAKING: sit-in at #Delhi, can't stop") == \
        ["BREAKING", ":", "sit-in", "at", "#Delhi", ",", "can't", "stop"]


def test_gold_labels_tag_multiword_actor_and_every_trigger():
    text = "Police used tear gas as clashes in Patna turned into a riot. Factory workers fled."
    toks, labs = _bio_tag(text, "clashes", "factory workers", "Patna", "nothing here")
    tagged = dict(zip(toks, labs))
    assert tagged["tear"] == "B-TRIGGER" and tagged["gas"] == "I-TRIGGER"
    assert tagged["riot"] == "B-TRIGGER"                      # not the slot trigger, still tagged
    assert tagged["Factory"] == "B-ACTOR" and tagged["workers"] == "I-ACTOR"


def test_dataset_has_episodes_and_held_out_templates(small):
    events, onset, _, _ = small
    assert events["episode"].nunique() == 2
    assert events["region"].nunique() == 14
    assert events["held_out_template"].any() and (~events["held_out_template"]).any()
    assert set(events["region_kind"]) == {"escalate", "fizzle", "flat"}


def test_no_leakage_forecastable_windows_contain_no_escalation(small):
    _, _, _, w = small
    elig = w[w["eligible"]]
    assert (elig["n_escalation"] == 0).all()                  # only pre-onset data is used
    assert (elig["window_end_day"] <= elig["onset_day"].fillna(99)).all()


def test_label_means_onset_within_next_three_days(small):
    _, _, _, w = small
    pos = w[w["label"] == 1]
    assert len(pos) > 0
    gap = pos["onset_day"] - pos["window_end_day"]
    assert gap.between(0, 2).all()
    quiet = w[w["onset_day"].isna()]
    assert (quiet["label"] == 0).all()


# ---------------------------------------------------------------- extraction
def test_span_scores_exact_match():
    gold = [["B-ACTOR", "I-ACTOR", "O", "B-LOCATION"]]
    pred = [["B-ACTOR", "O", "O", "B-LOCATION"]]               # actor span is cut short
    s = span_scores(gold, pred)
    assert s["per_type"]["LOCATION"]["f1"] == 1.0
    assert s["per_type"]["ACTOR"]["f1"] == 0.0


def test_crf_and_stage_classifier_connect_to_windows(small):
    events, onset, n_days, _ = small
    train = events[~events["held_out_template"]].sample(300, random_state=0)
    tagger = UnrestCRFTagger(max_iterations=40).fit(
        train["tokens"].apply(json.loads).tolist(), train["bio_labels"].apply(json.loads).tolist())
    frame = tagger.extract_events("BREAKING: students planning a strike in Delhi tomorrow.")
    assert frame["location"] == "Delhi" and "BREAKING" not in (frame["actor"] or "")
    stage_clf = StageClassifier().fit(train["text"], train["stage"])
    view = extracted_post_view(events, tagger.extract_many(events["text"].tolist()),
                               stage_clf.predict(events["text"]))
    w = build_windows(view, onset, n_days)
    assert len(w) > 0 and w["n_trigger_posts"].sum() > 0


# ---------------------------------------------------------------- forecasting
def test_gru_gradients_match_numerical_gradients():
    rng = np.random.RandomState(0)
    gru = GRUForecaster(hidden_dim=4)
    gru._init(3)
    X = rng.randn(5, 3, 2)
    y = (rng.rand(5, 1, 2) > 0.5).astype(float)
    m = np.ones((5, 1, 2))
    _, g = gru._loss_and_grads(X, y, m, pos_weight=2.0)
    for name in ("Wz", "Wr", "Wh", "Wo", "bh"):
        idx = (0, 1) if gru.p[name].shape[1] > 1 else (1, 0)
        old = gru.p[name][idx]
        gru.p[name][idx] = old + 1e-5
        lp, _ = gru._loss_and_grads(X, y, m, 2.0)
        gru.p[name][idx] = old - 1e-5
        lm, _ = gru._loss_and_grads(X, y, m, 2.0)
        gru.p[name][idx] = old
        assert abs((lp - lm) / 2e-5 - g[name][idx]) < 1e-5, name


def test_all_models_fit_and_output_probabilities(small):
    _, _, _, w = small
    for name, model in make_models().items():
        if name == "GRU":
            model.epochs = 30
        proba = model.fit(w).predict_proba(w)
        assert proba.shape == (len(w),), name
        assert ((proba >= 0) & (proba <= 1)).all(), name


def test_lead_time_and_false_alarms():
    df = pd.DataFrame({
        "region": ["A"] * 3 + ["B"] * 3,
        "window_end_day": [10, 11, 12] * 2,
        "onset_day": [12, 12, 12, None, None, None],
    })
    proba = np.array([0.1, 0.9, 0.9, 0.1, 0.1, 0.8])
    summary = summarize_lead_time(lead_time_analysis(df, proba, threshold=0.5))
    assert summary["detection_rate"] == 1.0
    assert summary["mean_lead_time_days"] == 1.0             # first alarm day 11, onset day 12
    assert summary["false_alarm_rate"] == 1.0                # quiet region B got an alarm


def test_challenge_set_labels_align_with_text():
    from src.pipeline import load_challenge_set
    cs = load_challenge_set()          # raises if any labelled phrase is not in its sentence
    assert len(cs) == 24
    labels = [l for s in cs["bio_labels"] for l in json.loads(s)]
    assert {"B-ACTOR", "B-LOCATION", "B-TARGET", "B-TRIGGER"} <= set(labels)


# ---------------------------------------------------------------- new roles + real data
def test_challenge_set_has_time_and_scale_roles():
    from src.pipeline import load_challenge_set
    labels = {l for s in load_challenge_set()["bio_labels"] for l in json.loads(s)}
    assert {"B-TIME", "B-SCALE"} <= labels


def test_synthetic_posts_label_time_and_scale():
    events, _, _, _ = build_dataset(n_episodes=1, seed=3)
    labels = {l for s in events["bio_labels"] for l in json.loads(s)}
    assert {"B-TIME", "I-TIME", "B-SCALE"} <= labels


def test_event_frame_has_all_roles_and_event_type():
    from src.extraction.model import bio_to_frame
    toks = ["Hundreds", "of", "students", "clashed", "in", "Patna", "today"]
    labs = ["B-SCALE", "O", "B-ACTOR", "B-TRIGGER", "O", "B-LOCATION", "B-TIME"]
    f = bio_to_frame(toks, labs)
    assert f["scale"] == "Hundreds" and f["time"] == "today" and f["actor"] == "students"
    assert f["event_type"] == "CLASH"


def test_repair_bio_turns_orphan_inside_tags_into_begin_tags():
    from src.extraction.transformer_tagger import repair_bio
    assert repair_bio(["O", "I-ACTOR", "I-ACTOR", "I-TIME"]) == ["O", "B-ACTOR", "I-ACTOR", "B-TIME"]


def test_headline_recovered_from_news_url():
    from src.real_pipeline import headline_from_url
    url = ("https://economictimes.indiatimes.com/news/india/"
           "ankita-bhandaris-parents-congress-leaders-detained-during-india-gate-protest/"
           "articleshow/134955007.cms")
    assert headline_from_url(url) == \
        "ankita bhandaris parents congress leaders detained during india gate protest"
    assert headline_from_url("https://example.com/article/123456") is None


def test_multi_onset_windows_skip_hot_days_and_label_each_onset():
    from src.prediction.aggregate import build_windows_multi
    view = pd.DataFrame({"region": ["A"] * 4, "day": [1, 2, 9, 10], "stage": ["grievance"] * 4,
                         "actors": [""] * 4, "targets": [""] * 4,
                         "has_trigger": [False] * 4, "has_escalation_trigger": [False] * 4})
    w = build_windows_multi(view, {"A": [5, 12]}, {"A": {5, 6, 12}}, n_days=16).set_index(
        "window_end_day")
    assert w.loc[[3, 4, 5], "label"].tolist() == [1, 1, 1]      # onset 5 within 3 days
    assert w.loc[6, "label"] == 0 and not w.loc[6, "eligible"]  # day 5 (hot) is in the window
    assert w.loc[10, "label"] == 1 and w.loc[10, "eligible"]    # second onset, calm again
    assert w.loc[10, "onset_day"] == 12


def test_real_annotations_align_with_text():
    from src.pipeline import load_real_annotated
    real = load_real_annotated()
    if real is None:
        pytest.skip("data/real/annotated_real.json not built yet")
    assert set(real["split"]) == {"train", "test"}
    assert set(real["source"]) <= {"mastodon", "gdelt"}

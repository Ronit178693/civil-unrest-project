"""
generate_synthetic_data.py
===========================
Generates a synthetic, but structurally realistic, dataset of social-media
posts (Twitter/X + Reddit style) describing civil-unrest activity, with
token-level BIO labels for event EXTRACTION and a per-region escalation
onset day for the PREDICTION task.

Why synthetic data?
--------------------
We have no Twitter/X, Reddit, GDELT or ACLED access yet. The generator
encodes the same *structure* real unrest data has:
  - informal, noisy, emotionally charged text
  - civil-unrest trigger vocabulary (protest, march, clash, strike, sit-in...)
  - argument slots: ACTOR, LOCATION, TARGET
  - an escalation process: grievance chatter -> mobilization calls
    -> on-ground reports -> (sometimes) escalation

Design choices that keep the evaluation honest
-----------------------------------------------
  * EPISODES: the 16-city simulation is repeated `n_episodes` times with
    different random seeds. Each (city, episode) pair is an independent
    "region" (e.g. "Jaipur-E2"). This gives enough regions for
    cross-validation instead of a single 16-region run.
  * HELD-OUT TEMPLATES: one sentence template per stage is never used to
    train the extractor, so we can measure how the extractor does on
    sentence structures it has never seen (`HELD_OUT_TEMPLATES`).
  * COMPLETE GOLD LABELS: every trigger word that appears in a post is
    labelled (not only the one the template slot was filled with), and
    multi-word actors are one span (B-ACTOR I-ACTOR).

Swapping in real data: see data/README_DATA_SOURCES.md.
"""

import os
import sys
import random
import json
import datetime as dt
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.extraction.tokenize import tokenize  # noqa: E402
from src.extraction.lexicon import FLAT_TRIGGERS  # noqa: E402

# ---------------------------------------------------------------------------
# Vocabulary banks
# ---------------------------------------------------------------------------

TRIGGERS = {
    "protest": ["protest", "protests", "protesting", "rally", "rallies", "demonstration", "demo"],
    "strike": ["strike", "strikes", "walkout", "walk-out", "shutdown", "bandh"],
    "clash": ["clash", "clashes", "clashed", "confrontation", "scuffle", "standoff"],
    "march": ["march", "marches", "marching", "procession"],
    "riot": ["riot", "riots", "unrest", "mob", "looting"],
    "sit_in": ["sit-in", "sit in", "occupation", "blockade", "road block"],
    "arrest": ["arrested", "detained", "crackdown", "tear gas", "lathi charge"],
}

ACTORS = [
    "students", "farmers", "transport unions", "factory workers", "opposition parties",
    "civil society groups", "teachers", "healthcare workers", "residents", "activists",
    "trade unions", "youth groups", "minority communities", "gig workers",
]

LOCATIONS = [
    "Delhi", "Lucknow", "Kolkata", "Mumbai", "Chennai", "Bengaluru", "Patna",
    "Jaipur", "Chandigarh", "Ahmedabad", "Hyderabad", "Noida", "Gurugram",
    "Ranchi", "Bhopal", "Guwahati",
]

TARGETS = [
    "the new labor bill", "fuel price hikes", "the fee hike", "police action",
    "the farm laws", "unpaid wages", "the curfew order", "election results",
    "water shortage", "land acquisition", "job cuts", "the transport fare hike",
]

SCALES = ["a few dozen", "hundreds of", "thousands of", "over a thousand", "a small group of"]

HASHTAG_POOL = ["#protest", "#justice", "#standwithworkers", "#shutdown", "#enough",
                "#civilrights", "#breakingnews", "#solidarity"]

TEMPLATES = {
    "grievance": [
        "Fed up with {target}. This can't go on anymore. {hashtag}",
        "People in {location} are furious about {target} — something has to give.",
        "{Actor} have had enough of {target}. Anger is building fast.",
        "No one is talking about {target} but we're living it every day in {location}.",
        "{Actor} in {location} say {target} is ruining lives. how long do we stay quiet?? {hashtag}",
        "so {target} again... {location} deserves better tbh",
    ],
    "mobilization": [
        "{Actor} are calling for a {trigger} against {target} tomorrow in {location}. {hashtag}",
        "BREAKING: {scale} {actor} planning to {trigger} in {location} over {target}.",
        "Join us! {trigger} planned outside the collectorate in {location} this weekend. {hashtag}",
        "Organizers confirm a {trigger} will go ahead in {location} despite warnings. {hashtag}",
        "everyone come out!! {trigger} at the main chowk in {location}, {actor} leading it. {hashtag}",
        "{Actor} announce {trigger} in {location} on Monday, demanding rollback of {target}.",
    ],
    "on_ground": [
        "Live: {scale} {actor} have gathered for a {trigger} in {location} right now.",
        "Reports of a {trigger} near {location} — {actor} demanding action on {target}. {hashtag}",
        "Situation tense in {location} as {trigger} continues into the evening.",
        "Visuals from {location}: {scale} {actor} on the streets over {target}.",
        "right now in {location}: {actor} holding a {trigger} against {target}, roads blocked",
        "{scale} {actor} still at the {trigger} site in {location}. police watching closely. {hashtag}",
    ],
    "escalation": [
        "Police used tear gas as {trigger} in {location} turned into a {escalation_trigger}. {hashtag}",
        "{scale} {actor} clashed with police in {location}; several {arrest_verb}.",
        "Situation in {location} spiraling: what started as a {trigger} is now a {escalation_trigger}.",
        "Unrest spreading beyond {location} as {actor} intensify {trigger} activity. {hashtag}",
        "{location} on edge: {escalation_trigger} reported, police use {arrest_verb} on {actor}. {hashtag}",
        "violent {trigger} between {actor} and police in {location} over {target}",
    ],
}

# One template per stage is never used to train the extractor.
HELD_OUT_TEMPLATES = {"grievance_2", "mobilization_3", "on_ground_1", "escalation_4"}

STAGES = ["grievance", "mobilization", "on_ground", "escalation"]
STAGE_TRIGGER_CATEGORIES = {
    "grievance": list(TRIGGERS.keys()),
    "mobilization": ["protest", "strike", "march", "sit_in"],
    "on_ground": ["protest", "march", "sit_in", "strike"],
    "escalation": ["clash", "riot"],
}


# ---------------------------------------------------------------------------
# Gold BIO labelling
# ---------------------------------------------------------------------------

def _bio_tag(text: str, trigger_text: Optional[str], actor: str, location: str, target: str):
    """Deterministic gold BIO labels for the synthetic corpus.

    Spans are tagged in priority order and never overwrite an earlier tag:
    the slot trigger, target, location, actor (as ONE span), then every other
    trigger word from the lexicon that appears in the text (e.g. "tear gas"
    in an escalation template).
    """
    tokens = tokenize(text)
    lower = [t.lower() for t in tokens]
    labels = ["O"] * len(tokens)

    def tag_span(span_text, tag, all_occurrences=False):
        span = [t.lower() for t in tokenize(span_text)]
        n = len(span)
        if n == 0:
            return
        for i in range(len(tokens) - n + 1):
            if lower[i:i + n] == span and all(l == "O" for l in labels[i:i + n]):
                labels[i] = f"B-{tag}"
                for j in range(1, n):
                    labels[i + j] = f"I-{tag}"
                if not all_occurrences:
                    return

    if trigger_text:
        tag_span(trigger_text, "TRIGGER")
    tag_span(target, "TARGET")
    tag_span(location, "LOCATION")
    tag_span(actor, "ACTOR")
    for phrase in sorted(FLAT_TRIGGERS, key=lambda p: -len(p.split())):  # bigrams first
        tag_span(phrase, "TRIGGER", all_occurrences=True)

    return tokens, labels


def _stage_for_day(rng, kind, days_to_peak):
    """Stage mix for one post. kind: 'escalate' | 'fizzle' | 'flat'.
    'fizzle' regions build up like escalating ones (mobilization, some
    on-ground activity) but die down instead of escalating -- the hard
    negatives that make forecasting non-trivial, as in real data."""
    if kind == "escalate":
        if days_to_peak > 6:
            return "grievance" if rng.random() < 0.85 else "mobilization"
        if days_to_peak > 3:
            return rng.choices(["grievance", "mobilization"], weights=[0.4, 0.6])[0]
        if days_to_peak > 0:
            return rng.choices(["mobilization", "on_ground"], weights=[0.5, 0.5])[0]
        return rng.choices(["on_ground", "escalation"], weights=[0.4, 0.6])[0]
    if kind == "fizzle":
        if days_to_peak > 6:
            return "grievance" if rng.random() < 0.85 else "mobilization"
        if days_to_peak > 3:
            return rng.choices(["grievance", "mobilization"], weights=[0.5, 0.5])[0]
        if days_to_peak > 0:
            return rng.choices(["grievance", "mobilization", "on_ground"], weights=[0.2, 0.5, 0.3])[0]
        return rng.choices(["grievance", "mobilization", "on_ground"], weights=[0.5, 0.2, 0.3])[0]
    return rng.choices(["grievance", "mobilization", "on_ground"], weights=[0.7, 0.2, 0.1])[0]


def generate_region_timeline(region_id: str, city: str, episode: int, start_date: dt.date,
                             n_days: int, kind: str, rng: random.Random,
                             np_rng: np.random.RandomState) -> Tuple[List[dict], Optional[int]]:
    """Day-by-day posts for one region of a given kind ('escalate', 'fizzle',
    'flat'). Every region also gets 1-2 random 'viral' days with a burst of
    grievance chatter, so raw post volume alone is not a giveaway."""
    posts = []
    peak_day = rng.randint(n_days - 4, n_days - 1) if kind in ("escalate", "fizzle") else None
    escalation_day = peak_day if kind == "escalate" else None
    viral_days = set(rng.sample(range(n_days), rng.randint(1, 2)))
    slope = {"escalate": 0.8, "fizzle": 0.55, "flat": 0.0}[kind]

    for day in range(n_days):
        date = start_date + dt.timedelta(days=day)
        n_posts = int(np_rng.poisson(4 if kind == "flat" else 3 + day * slope))
        n_extra = int(np_rng.poisson(8)) if day in viral_days else 0
        n_posts = max(1, min(n_posts, 25))

        for k in range(n_posts + n_extra):
            days_to_peak = (peak_day - day) if peak_day is not None else None
            stage = "grievance" if k >= n_posts else _stage_for_day(rng, kind, days_to_peak)
            target = rng.choice(TARGETS)
            actor = rng.choice(ACTORS)
            t_idx = rng.randrange(len(TEMPLATES[stage]))
            template = TEMPLATES[stage][t_idx]
            trig_cat = rng.choice(STAGE_TRIGGER_CATEGORIES[stage])
            trigger_text = rng.choice(TRIGGERS[trig_cat])

            text = template.format(
                target=target, location=city, actor=actor, Actor=actor.capitalize(),
                trigger=trigger_text, scale=rng.choice(SCALES), hashtag=rng.choice(HASHTAG_POOL),
                escalation_trigger=rng.choice(TRIGGERS["riot"]),
                arrest_verb=rng.choice(TRIGGERS["arrest"]),
            )
            # grievance templates have no {trigger} slot: nothing to tag for it
            slot_trigger = trigger_text if "{trigger}" in template else None
            tokens, labels = _bio_tag(text, slot_trigger, actor, city, target)
            ts = dt.datetime.combine(date, dt.time(rng.randint(0, 23), rng.randint(0, 59)))
            template_id = f"{stage}_{t_idx}"

            posts.append({
                "post_id": f"{region_id}-{date.isoformat()}-{rng.randint(1000, 9999)}",
                "text": text,
                "platform": rng.choice(["twitter", "reddit"]),
                "timestamp": ts.isoformat(),
                "date": date,
                "day": day,
                "episode": episode,
                "city": city,
                "region": region_id,
                "region_kind": kind,
                "stage": stage,
                "template_id": template_id,
                "held_out_template": template_id in HELD_OUT_TEMPLATES,
                "trigger_text": slot_trigger,
                "trigger_type": trig_cat if slot_trigger else None,
                "actor": actor,
                "location": city,
                "target": target,
                "tokens": json.dumps(tokens),
                "bio_labels": json.dumps(labels),
            })
    return posts, escalation_day


def build_dataset(n_episodes: int = 5, n_regions_escalate: int = 6, n_regions_quiet: int = 10,
                  n_days: int = 21, start_date: Optional[dt.date] = None, seed: int = 42,
                  fizzle_share: float = 0.5):
    """Returns (events_df, region_escalation_day, start_date, n_days).

    Per episode: n_regions_escalate escalating regions, and n_regions_quiet
    quiet ones of which `fizzle_share` build up and then fizzle out.

    region_escalation_day maps region id -> escalation onset day (0-based)
    or None for regions that never escalate.
    """
    start_date = start_date or dt.date(2026, 1, 1)
    assert n_regions_escalate + n_regions_quiet <= len(LOCATIONS)
    all_posts, region_escalation_day = [], {}

    for ep in range(n_episodes):
        rng = random.Random(seed + 1000 * ep)
        np_rng = np.random.RandomState(seed + 1000 * ep)
        cities = LOCATIONS.copy()
        rng.shuffle(cities)
        chosen = cities[:n_regions_escalate + n_regions_quiet]
        n_fizzle = int(round(n_regions_quiet * fizzle_share))
        for i, city in enumerate(chosen):
            kind = ("escalate" if i < n_regions_escalate
                    else "fizzle" if i < n_regions_escalate + n_fizzle else "flat")
            region_id = f"{city}-E{ep}"
            posts, esc_day = generate_region_timeline(region_id, city, ep, start_date, n_days,
                                                      kind, rng, np_rng)
            all_posts.extend(posts)
            region_escalation_day[region_id] = esc_day

    events_df = pd.DataFrame(all_posts)
    return events_df, region_escalation_day, start_date, n_days


def build_window_labels(events_df, region_escalation_day, start_date=None, n_days=21,
                        window_size=3, horizon=3):
    """Gold-feature windows (kept for backwards compatibility; the pipeline
    uses src.prediction.aggregate.build_windows directly)."""
    from src.prediction.aggregate import build_windows, gold_post_view
    return build_windows(gold_post_view(events_df), region_escalation_day, n_days,
                         window_size=window_size, horizon=horizon)


def write_sample_posts(events_df, region_escalation_day, path):
    """A small CSV for trying the forecaster yourself: one region a few days
    before it escalates, and one region that builds up but never escalates (fizzle), cut at the same day."""
    ep0 = events_df[events_df["episode"] == 0]
    esc = [r for r in ep0["region"].unique() if region_escalation_day.get(r) is not None]
    quiet = [r for r in ep0["region"].unique()
             if ep0.loc[ep0["region"] == r, "region_kind"].iloc[0] == "fizzle"]
    r_esc, r_quiet = esc[0], quiet[0]
    cutoff = region_escalation_day[r_esc]          # posts up to the day before onset
    sample = ep0[ep0["region"].isin([r_esc, r_quiet]) & (ep0["day"] < cutoff)]
    sample = sample.assign(region=sample["city"])[["timestamp", "region", "platform", "text"]]
    sample.to_csv(path, index=False)
    onset_date = (pd.Timestamp(ep0["date"].min()) + pd.Timedelta(days=cutoff)).date().isoformat()
    truth = {r_esc.split("-")[0]: onset_date, r_quiet.split("-")[0]: None}
    with open(path.replace(".csv", "_truth.json"), "w") as f:
        json.dump(truth, f, indent=2)
    return r_esc, r_quiet, cutoff


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    events_df, region_escalation_day, start_date, n_days = build_dataset()
    windows_df = build_window_labels(events_df, region_escalation_day, start_date, n_days)

    events_df.to_csv(os.path.join(out_dir, "synthetic_events.csv"), index=False)
    windows_df.to_csv(os.path.join(out_dir, "synthetic_windows.csv"), index=False)
    r_esc, r_quiet, cutoff = write_sample_posts(
        events_df, region_escalation_day, os.path.join(out_dir, "sample_posts.csv"))

    elig = windows_df[windows_df["eligible"]]
    print(f"Generated {len(events_df)} posts across {events_df['region'].nunique()} regions "
          f"({events_df['episode'].nunique()} episodes x 16 cities).")
    print(f"Generated {len(elig)} forecastable region-windows "
          f"({elig['label'].mean():.1%} positive).")
    print(f"Wrote data/sample_posts.csv: {r_esc.split('-')[0]} (escalates on day {cutoff}, "
          f"posts stop the day before) and {r_quiet.split('-')[0]} (builds up, never escalates).")

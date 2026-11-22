# Connecting Real Data Sources

This repo ships with `generate_synthetic_data.py` so the whole pipeline runs
offline, deterministically, and without any API keys. For a real deployment
or for the thesis/paper's final experiments, replace it with:

## 1. Social media collection

**Twitter/X API v2** (`tweepy`)
```python
import tweepy
client = tweepy.Client(bearer_token=BEARER_TOKEN)
query = '(protest OR strike OR rally OR bandh OR "sit-in") (lang:en OR lang:hi) -is:retweet'
resp = client.search_recent_tweets(query=query, max_results=100,
                                    tweet_fields=["created_at", "geo", "author_id"])
```
Use the **filtered stream** endpoint instead of search for continuous, real-time
collection; search is sufficient for backfilling a historical window (last 7
days on the standard tier).

**Reddit API** (`praw`)
```python
import praw
reddit = praw.Reddit(client_id=..., client_secret=..., user_agent=...)
for submission in reddit.subreddit("india+worldnews").search("protest OR strike", time_filter="week"):
    ...
```

Both sources should be normalized to the same schema the synthetic generator
produces: `post_id, text, platform, timestamp, region (best-effort, see
geotagging note below), raw`.

**Geotagging note**: most posts lack precise geotags. Use a two-stage
approach: (a) regex/gazetteer match of city/state names mentioned in the
text (what this repo's `LOCATION` argument role is designed to capture),
falling back to (b) author-profile location field, treated as weak/noisy
signal only.

## 2. Ground-truth unrest event labels

**GDELT GKG 2.0** -- free, no key required for bulk files; `gdeltdoc` wraps
the DOC 2.0 API for keyword/time/country filtered queries. Useful as **weak
supervision**: cross-reference extracted event (region, date) pairs against
GDELT `EventRootCode` 14 (PROTEST) records to validate / auto-label.

**ACLED** -- requires a free academic API key
(https://acleddata.com/data-export-tool/). Provides curated, human-coded
protest/riot/violence events with date, admin1/2 location, and actor tags --
the strongest available ground truth for the *prediction* task's labels
(replace `region_escalation_day` in `generate_synthetic_data.py` with actual
ACLED event dates per region).

## 3. Swapping the data source without touching the rest of the pipeline

**Just want forecasts on new posts?** No retraining is needed:
`python demo.py forecast your_posts.csv` (columns `timestamp, region, text`),
or `src.forecast.forecast_posts(...)` from Python.

**Retraining on real data** needs a posts dataframe with the same columns
`build_dataset()` returns:
- `text, region, day` (0-based day index), `episode` (use 0 for posts that
  train the extractor and 1+ for forecasting regions)
- `tokens`, `bio_labels`: JSON lists, only for posts used to train or
  evaluate the CRF (see section 4)
- `stage`: grievance / mobilization / on_ground / escalation, only for
  posts used to train the stage classifier
- `held_out_template`: set to False for real data

You also need `region_escalation_day`: region -> day the escalation started
(from ACLED), or None. Then `src/pipeline.py`, both extractors, the window
builder (`src/prediction/aggregate.py`), the forecasters and the evaluation
work unchanged.

## 4. Labeling real text for the extraction model

The CRF needs token-level BIO labels to train on. For a real corpus:
1. Pull ~500-1000 posts matching unrest keywords.
2. Annotate with a lightweight tool (e.g. Doccano, Label Studio) using the
   same label set: `TRIGGER, ACTOR, LOCATION, TARGET`.
3. Export to the same `(tokens, bio_labels)` pair format produced by
   `_bio_tag()` in this file.
4. Consider active learning: run the synthetic-trained CRF on new real
   posts first, then have annotators correct its output rather than
   labeling from scratch -- substantially faster.

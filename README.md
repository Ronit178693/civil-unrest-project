# Civil Unrest Event Extraction and Prediction from Social Media Data

A two-stage pipeline that (1) extracts civil-unrest events (trigger, event
type, actor, location, target, time, scale, and the event's stage) from
informal social-media posts with a fine-tuned transformer,
and (2) aggregates those extracted events per region over time to
**forecast whether escalation will start within the next 3 days**, with
early-warning lead-time analysis.

Built for the project brief *"Civil Unrest Event Extraction and Prediction
from social media data"* (Instructor: Dr. Pratima Singh, Bennett University).

**New here? Read [USER_GUIDE.md](USER_GUIDE.md)**: setup, every command,
what each output means, and how to run the model on your own posts.

> **Data.** Three sources:
> 1. **Real news events (GDELT).** 29,508 GDELT-coded protest events in India,
>    May-October 2026 (`data/real/fetch_gdelt.py`). GDELT's violent-protest
>    codes (CAMEO 145) give the escalation ground truth; article headlines,
>    recovered from the URLs, are the text the pipeline reads.
> 2. **Real social-media posts (Mastodon).** 1,188 public posts from
>    protest/strike hashtags, only from accounts that opted in to search
>    indexing (`data/real/fetch_mastodon.py`). Twitter/X and Reddit refuse
>    unauthenticated access, so Mastodon stands in for them.
> 3. **Synthetic posts** (`data/generate_synthetic_data.py`) with complete gold
>    labels, for controlled experiments.
>
> 160 real items (80 Mastodon posts + 80 GDELT headlines) are hand-annotated
> with every role and stage (`data/real/annotated_real.json`); half trains the
> domain-adapted extractor, half is held out for testing.

## Quickstart (Windows / PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt

.venv\Scripts\python data\generate_synthetic_data.py   # 1. synthetic dataset + sample CSV
.venv\Scripts\python data\real\fetch_gdelt.py         # 2. real GDELT events (~25 min, resumable)
.venv\Scripts\python -m src.pipeline                   # 3. train + evaluate extractors and forecasters
                                                        #    (~1 h on a laptop CPU; --tagger crf: ~2 min)
.venv\Scripts\python -m src.real_pipeline              # 4. forecasting on real GDELT data
.venv\Scripts\python src\make_figures.py               # 5. figures -> paper/figures/
.venv\Scripts\python -m pytest tests -v                # 6. tests
.venv\Scripts\python demo.py                           # 7. guided demo
```

On macOS/Linux use `.venv/bin/python` instead of `.venv\Scripts\python`.

## Architecture

```
posts ──► Stage 1a  fine-tuned transformer tagger (BIO; twitter-roberta-base, CRF baseline)
      │            ─► trigger + event type / actor / location / target / time / scale
      └─► Stage 1b  stage classifier (TF-IDF+LR) ► grievance / mobilization / on-ground / escalation
                          │
                          ▼  aggregate Stage 1 OUTPUT per region, 3-day windows, daily
              Stage 2  forecaster ──► P(escalation starts in the next 3 days)
                          │
                          ▼
              Evaluation: span F1 · P/R/F1, ROC-AUC, average precision ·
                          detection rate, lead time, false-alarm rate
```

## Method

**Synthetic data.** 5 independent simulation *episodes* × 16 Indian cities = 80
regions, 21 days each, 14,402 posts. Per episode: 6 regions escalate, 5
build up and then **fizzle out** (hard negatives), 5 stay flat. Every
region also gets 1–2 random "viral" days, so post volume alone is not a
giveaway. One sentence template per stage is held out from extractor
training.

**Real data.** GDELT 1.0 daily event files, filtered to protest events
(CAMEO 14x) geolocated in India, May–October 2026. Regions are states
(ADM1). A state *escalates* on a day when at least 2 distinct articles report
a violent protest or riot (CAMEO 145) there after 7 calm days. Each event's
article headline is recovered from its URL slug and fed to Stage 1, so the
forecaster sees our extracted events, never GDELT's codes. Train May–August,
test September–October (split by time).

**Stage 1a: event extraction.** Token classification over BIO tags for
TRIGGER, ACTOR, LOCATION, TARGET, TIME and SCALE; the event type (PROTEST,
STRIKE, CLASH, RIOT, SIT_IN, REPRESSION) is read off the trigger. Main model:
`cardiffnlp/twitter-roberta-base` (RoBERTa further pre-trained on tweets),
fine-tuned on the synthetic posts plus the real training half (the
*domain-adapted* model). Baseline: a linear-chain CRF over a trigger lexicon,
place gazetteer, time words, word shape, casing, hashtags, affixes and ±2-token
context. Both are trained with and without the real data, to measure the
effect of adaptation.

**Stage 1b: stage classification.** Logistic regression over word and
character n-grams, plus domain cues (trigger present, escalation-type
trigger, future-tense vs live wording).

**Data split.** Stage 1 is trained only on episode 0. Stage 2 uses episodes
1–4, so the extractor never saw the forecasting regions.

**Stage 2: forecasting.** For every region and day *e*, a window covers days
*e−3…e−1*. The label is 1 if escalation **starts** in days *e…e+2*. Windows
after the onset are excluded, so the model never sees escalation that has
already happened. Features: stage counts, trigger counts, distinct actors
and targets, plus deltas, rolling means and ratios (30 features). Four
models are compared with **region-grouped 5-fold cross-validation**:

- volume-only baseline
- logistic regression
- gradient boosting
- a GRU trained with full backpropagation through time (NumPy, gradient-checked in the tests)

Each model is run on Stage 1's extracted features and on gold features
(upper bound).

## Results (seed 42; reproduce with `python -m src.pipeline` and `python -m src.real_pipeline`)

**Extraction**: exact-match span F1 over all six roles. "Adapted" = also
trained on the 80-item real training half.

| Test set | CRF (synthetic) | CRF (adapted) | Transformer (synthetic) | **Transformer (adapted)** |
|---|---|---|---|---|
| Synthetic, unseen templates (1,000 posts) | 0.99 | 0.99 | 0.93 | 0.94 |
| Challenge set, new vocabulary (24 posts) | 0.77 | 0.78 | 0.85 | **0.86** |
| Real Mastodon posts (40) | 0.38 | **0.61** | 0.32 | 0.58 |
| Real GDELT headlines (40) | 0.49 | 0.55 | 0.43 | **0.67** |
| **All real test items (80)** | 0.43 | 0.58 | 0.37 | **0.63** |

Trained on synthetic text alone, both extractors collapse on real text; a
small amount of real training data (80 items) recovers most of the gap, and
the adapted transformer is the best model overall. On the real test set it
finds triggers well (F1 0.78) and actors and locations moderately (0.63,
0.62); targets (0.42), time (0.50) and scale (0.40) are hard. The stage
classifier's accuracy on real text rises from 0.50 to 0.64 with adaptation.

**Forecasting, synthetic** (64 regions, 1,020 windows, 55 positive, region-grouped
5-fold CV, adapted-transformer features):

| Model | Precision | Recall | F1 | Avg. precision | Lead time | False-alarm rate |
|---|---|---|---|---|---|---|
| Volume baseline | 0.28 | 0.95 | 0.43 | 0.49 | 6.2 d | 40% |
| Logistic regression | 0.79 | 0.96 | 0.87 | 0.98 | 1.9 d | 25% |
| Gradient boosting | 0.94 | 0.91 | **0.93** | **0.99** | 1.8 d | **5%** |
| GRU | 0.79 | 0.98 | 0.88 | 0.94 | 2.0 d | 20% |

**Forecasting, real** (GDELT India, 22 states, 62 escalation onsets; train
May–Aug, test Sep–Oct 2026: 1,202 test windows, 53 positive = 4.4%, 19 onsets):

| Model (features) | Precision | Recall | ROC-AUC | Avg. precision | Onsets warned | Mean lead |
|---|---|---|---|---|---|---|
| Volume baseline (headline count) | 0.05 | 0.32 | 0.62 | 0.06 | 58% | 0.8 d |
| Logistic regression (our headlines) | 0.08 | 0.45 | 0.63 | 0.06 | 68% | 1.0 d |
| GRU (our headlines) | 0.10 | 0.25 | 0.56 | **0.13** | 26% | 2.0 d |
| Logistic regression (GDELT codes) | 0.08 | 0.45 | **0.70** | 0.09 | 74% | 1.0 d |

Real escalation is much harder to forecast than the synthetic version: the
best average precision is about 3x chance (0.13 vs 0.04), and alarms are
mostly false (precision ≤ 0.10). Gradient boosting, the best synthetic model,
raises almost no alarms at the 0.5 threshold on real data. Features from our
extracted headlines are about as informative as GDELT's own event codes.

## Repository layout

```
├── data/
│   ├── generate_synthetic_data.py   # synthetic corpus (episodes, fizzle regions, held-out templates)
│   ├── challenge_set.json           # 24 hand-labelled posts with unseen vocabulary
│   ├── real/
│   │   ├── fetch_gdelt.py            # GDELT 1.0 events -> India protest events (CAMEO 14x)
│   │   ├── fetch_mastodon.py         # public, opted-in Mastodon posts on unrest hashtags
│   │   ├── gdelt_india_protest_events.csv
│   │   └── annotated_real.json       # 160 hand-annotated real items (train/test halves)
│   ├── sample_posts.csv             # small CSV to try the forecaster on (+ _truth.json)
│   └── README_DATA_SOURCES.md       # data sources and how to plug in more
├── src/
│   ├── extraction/                  # tokenizer, lexicon, CRF, transformer_tagger.py, stage classifier
│   ├── prediction/                  # aggregate.py (windows, labels), model.py (4 forecasters)
│   ├── evaluation/metrics.py        # classification + early-warning metrics
│   ├── pipeline.py                  # extractors + synthetic forecasting, saves models
│   ├── real_pipeline.py             # forecasting on real GDELT data (time split)
│   ├── forecast.py                  # use the trained models on new posts
│   └── make_figures.py              # figures from results/
├── demo.py                          # guided tour / type-a-post / forecast-a-CSV
├── tests/test_pipeline.py           # 18 tests, incl. leakage checks and GRU gradient check
├── results/                         # created by the pipelines (metrics, predictions, models/)
├── paper/, slides/                  # paper, figures, decks (paper/slides text predates the real-data work)
├── USER_GUIDE.md
└── requirements.txt
```

## Known limitations

- Twitter/X, Facebook and Reddit need API access we do not have; Mastodon stands in
  for social media, and the real forecasting text is news headlines (from GDELT URLs),
  not posts.
- The real annotated set is small (160 items, one annotator), so extraction scores on
  real text have wide error bars.
- English only; the multilingual text the brief mentions is not handled yet.
- Real forecasting precision is low; the 0.5 threshold is not tuned (`--threshold`).
- Escalation ground truth is GDELT's automatic coding, which contains misclassified
  articles; ACLED's human-coded events would be stronger ground truth.

## License

MIT. See `LICENSE`.

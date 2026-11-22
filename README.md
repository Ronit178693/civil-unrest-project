# Civil Unrest Event Extraction and Prediction from Social Media Data

A two-stage pipeline that (1) extracts civil-unrest events (trigger, actor,
location, target, and the event's stage) from informal social-media posts,
and (2) aggregates those extracted events per region over time to
**forecast whether escalation will start within the next 3 days**, with
early-warning lead-time analysis.

Built for the project brief *"Civil Unrest Event Extraction and Prediction
from social media data"* (Instructor: Dr. Pratima Singh, Bennett University).

**New here? Read [USER_GUIDE.md](USER_GUIDE.md)**: setup, every command,
what each output means, and how to run the model on your own posts.

> **Data note.** We have no Twitter/X, Reddit, GDELT or ACLED access yet, so
> the pipeline runs on a synthetic but structurally realistic dataset
> (`data/generate_synthetic_data.py`). To measure how the extractor does on
> language the generator never produces, we also evaluate on a small
> hand-labelled challenge set (`data/challenge_set.json`).
> `data/README_DATA_SOURCES.md` explains how to plug in real data.

## Quickstart (Windows / PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt

.venv\Scripts\python data\generate_synthetic_data.py   # 1. dataset + sample CSV
.venv\Scripts\python -m src.pipeline                   # 2. train + evaluate (~2 min)
.venv\Scripts\python src\make_figures.py               # 3. figures -> paper/figures/
.venv\Scripts\python -m pytest tests -v                # 4. 11 tests (~15 s)
.venv\Scripts\python demo.py                           # 5. guided demo
```

On macOS/Linux use `.venv/bin/python` instead of `.venv\Scripts\python`.

## Architecture

```
posts ──► Stage 1a  CRF tagger (BIO) ─────────► trigger / actor / location / target
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

**Data.** 5 independent simulation *episodes* × 16 Indian cities = 80
regions, 21 days each, 14,402 posts. Per episode: 6 regions escalate, 5
build up and then **fizzle out** (hard negatives), 5 stay flat. Every
region also gets 1–2 random "viral" days, so post volume alone is not a
giveaway. One sentence template per stage is held out from extractor
training.

**Stage 1a: event extraction.** Linear-chain CRF over BIO tags for TRIGGER,
ACTOR, LOCATION and TARGET. Features: trigger lexicon (unigram + bigram),
place-name gazetteer, word shape, casing, hashtags, affixes, ±2-token
context. Text goes through one shared tokenizer for training and inference.

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

## Results (seed 42; reproduce with `python -m src.pipeline`)

**Extraction**: exact-match span F1.

| Test set | Span F1 | Stage accuracy |
|---|---|---|
| Synthetic, seen templates (3,000 posts) | 1.000 | 1.000 |
| Synthetic, unseen templates (1,926 posts) | 1.000 | 0.877 |
| **Hand-labelled challenge set, new vocabulary (24 posts)** | **0.779** | **0.708** |

The synthetic scores are an upper bound: the generator uses a closed
vocabulary. The challenge set is the realistic number. Locations generalise
(F1 0.98, from the gazetteer), but new actor phrases (recall 0.35) and new
targets (recall 0.43) are often missed.

**Forecasting**: 64 regions, 1,020 forecastable windows, 59 positive (5.8%),
on Stage 1's extracted features.

| Model | Precision | Recall | F1 | Avg. precision | Lead time | False-alarm rate |
|---|---|---|---|---|---|---|
| Volume baseline | 0.33 | 0.92 | 0.49 | 0.55 | 5.5 d | 35% |
| Logistic regression | 0.83 | 0.98 | 0.90 | **0.97** | 2.0 d | 15% |
| Gradient boosting | 0.90 | 0.88 | 0.89 | 0.95 | 1.8 d | **8%** |
| GRU | 0.86 | 0.86 | 0.86 | 0.92 | 1.7 d | 10% |

All escalating regions are flagged before onset. The event features beat
the volume baseline by a wide margin. With gold features the scores are
almost the same (logistic regression F1 0.92), so Stage 1's errors cost
little *on synthetic text*.

The ~2-day lead time is a ceiling set by the generator: on-ground activity
starts 3 days before onset. Real lead time must be measured on ACLED data.

## Repository layout

```
├── data/
│   ├── generate_synthetic_data.py   # synthetic corpus (episodes, fizzle regions, held-out templates)
│   ├── challenge_set.json           # 24 hand-labelled posts with unseen vocabulary
│   ├── sample_posts.csv             # small CSV to try the forecaster on (+ _truth.json)
│   └── README_DATA_SOURCES.md       # how to plug in Twitter/Reddit/GDELT/ACLED data
├── src/
│   ├── extraction/                  # tokenize, lexicon+gazetteer, CRF features/model, stage classifier
│   ├── prediction/                  # aggregate.py (windows, labels), model.py (4 forecasters)
│   ├── evaluation/metrics.py        # classification + early-warning metrics
│   ├── pipeline.py                  # end-to-end train + evaluate + save models
│   ├── forecast.py                  # use the trained models on new posts
│   └── make_figures.py              # figures from results/
├── demo.py                          # guided tour / type-a-post / forecast-a-CSV
├── tests/test_pipeline.py           # 11 tests, incl. leakage checks and GRU gradient check
├── results/                         # created by the pipeline (metrics, predictions, models/)
├── paper/, slides/                  # paper, figures, decks (paper/slides text predates these fixes)
├── USER_GUIDE.md
└── requirements.txt
```

## Known limitations

- Synthetic data with a closed vocabulary; the challenge set has only 24 posts.
- Lead time is capped by the generator design; it needs real (ACLED) validation.
- The CRF does not generalise well to unseen actor and target phrases. A fine-tuned
  transformer (`TransformerTaggerStub` in `src/extraction/model.py`) is the planned upgrade.
- The probability threshold is fixed at 0.5. Lowering it trades more false alarms
  for earlier warnings (`--threshold`).

## License

MIT. See `LICENSE`.

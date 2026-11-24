# User Guide

Everything you need to run, understand and extend the project yourself.
All commands are for **PowerShell on Windows**, run from the project folder:

```powershell
cd "C:\Users\Lenovo\Downloads\Class Project\civil-unrest-project-source-code\civil-unrest-project"
```

---

## 1. One-time setup

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

- This creates a private Python environment in `.venv\` and installs the libraries into it.
- On Python 3.14 the install can take ~10 minutes (one package compiles from source).
  It includes PyTorch (CPU build) and Hugging Face Transformers for the transformer extractor.
- You never need to "activate" the environment. Every command below calls `.venv\Scripts\python` directly.
- The first full pipeline run downloads `cardiffnlp/twitter-roberta-base` (~500 MB). On a slow
  or unstable connection, download it once into `models_cache\twitter-roberta-base\`
  (files `config.json`, `vocab.json`, `merges.txt`, `pytorch_model.bin`); the code uses that
  local copy automatically.

---

## 2. Run everything (in this order)

| # | Command | Time | What it does |
|---|---|---|---|
| 1 | `.venv\Scripts\python data\generate_synthetic_data.py` | 5 s | Builds the synthetic dataset and `data\sample_posts.csv` |
| 2 | `.venv\Scripts\python data\real\fetch_gdelt.py` | ~25 min | Downloads 6 months of GDELT events, keeps India protest events (resumable) |
| 3 | `.venv\Scripts\python -m src.pipeline` | ~1 h (CPU) | Trains and compares 4 extractors, trains Stage 2, evaluates, saves the models |
| 4 | `.venv\Scripts\python -m src.real_pipeline` | ~11 min | Forecasting on real GDELT data (train May–Aug, test Sep–Oct) |
| 5 | `.venv\Scripts\python src\make_figures.py` | 10 s | Draws 7 figures into `paper\figures\` |
| 6 | `.venv\Scripts\python -m pytest tests -v` | 15 s | Runs the 18 automated tests |
| 7 | `.venv\Scripts\python demo.py` | 30 s | Guided demo using the saved models |

- Step 2 is only needed once; `data\real\gdelt_india_protest_events.csv` is already in the repo.
  `data\real\fetch_mastodon.py` re-collects the Mastodon posts (optional; the annotated sample is in the repo).
- Step 3 must run before steps 4, 5 and 7, because they read what it saves.
- Re-running gives the same numbers, because everything is seeded.
- For a fast check that nothing is broken, run `.venv\Scripts\python -m src.pipeline --quick`
  (~1 min, CRF only, smaller data). `--tagger crf` runs the full data with the CRF only (~2 min).
- Laptop going to sleep kills a long run: keep it plugged in and awake during step 3.

**Do not** run `python src\pipeline.py`; use `-m src.pipeline`. The `-m`
form lets Python find the `src` package.

---

## 3. Reading the pipeline output

### Dataset line
`14402 posts | 80 regions (5 episodes x 16 cities) | 30 regions escalate`

- An **episode** is one 21-day simulation of 16 cities.
- A **region** is a city in one episode, e.g. `Jaipur-E2`.
- Per episode, 6 regions escalate, 5 build up and then fizzle out, and 5 stay flat.

### Stage 1: extraction
Four extractors are trained and scored; each prints one line of span F1 per test set:
```
training Transformer (adapted) on 1600 posts ...
  seen_templates 1.000 | unseen_templates 0.940 | challenge_set 0.860 | real_test 0.627 | ...
```
- **CRF / Transformer:** the feature-based baseline and the fine-tuned `twitter-roberta-base`.
- **synthetic / adapted:** trained on synthetic posts only, or also on the 80 real training items
  from `data\real\annotated_real.json` (the domain-adapted model, used downstream).
- **Span F1:** an extracted phrase counts only if its start, end and role are all exactly right.
- **Seen and unseen templates:** synthetic posts from other episodes. Unseen means sentence patterns the model never trained on.
- **Challenge set:** 24 hand-written posts with new words (`data\challenge_set.json`).
- **real_test:** the 80 held-out real items (40 Mastodon posts, 40 GDELT headlines). **This is the realistic number.**
- **Stage classifier** lines: accuracy of the post-stage classifier (grievance, mobilization, on-ground, escalation).

### Stage 2: forecasting table
One row per model, run on two feature sources:
- `extracted`: features built from Stage 1's output. This is the real, connected pipeline.
- `gold`: features built from the true labels. This is an upper bound.

| Column | Meaning |
|---|---|
| precision | Of the windows flagged, the share that really preceded escalation |
| recall | Of the windows that preceded escalation, the share that were flagged |
| f1 `+/-` | Balance of precision and recall; `+/-` is the spread across the 5 folds |
| roc_auc | Ranking quality from 0.5 (random) to 1 (perfect) |
| avg_precision | Like ROC-AUC, but honest when positives are rare (here 5.4%). This is the main score. |
| detection_rate | Share of escalating regions flagged on or before the onset day |
| mean_lead_days | Average days between the first alarm and the onset |
| false_alarm_rate | Share of quiet regions that got any alarm at all |

### Files in `results\`
| File | Contents |
|---|---|
| `extraction_metrics.json` | Extraction scores per test set and per role |
| `extraction_report_*.txt` | Detailed per-label reports for every extractor |
| `model_comparison.csv` | The Stage 2 table above |
| `oof_predictions.csv` | Every window's predicted probability (from the fold that held it out) |
| `lead_time_by_region.csv` | First alarm day and lead time per region and model |
| `feature_importances.json` | Which features gradient boosting relies on |
| `windows_extracted.csv`, `windows_gold.csv` | The forecasting dataset itself |
| `models\pipeline.joblib` | Stage classifier, best forecaster and a pointer to the extractor (used by `demo.py`) |
| `models\transformer_tagger\` | The fine-tuned transformer weights |
| `run_summary.json` | Settings and headline numbers of the last run |
| `real\real_model_comparison.csv` | Real-data forecasting table (`src.real_pipeline`) |
| `real\real_summary.json` | Real-data counts: events, states, onsets, windows |
| `real\real_test_predictions.csv`, `real\real_windows_*.csv` | Real test predictions and windows |

---

## 4. Using the demo

```powershell
.venv\Scripts\python demo.py                                    # guided tour
.venv\Scripts\python demo.py extract                            # type your own posts
.venv\Scripts\python demo.py forecast data\sample_posts.csv     # forecast from a CSV
.venv\Scripts\python demo.py forecast data\sample_posts.csv --all   # every day, not just the latest
```

- **extract:** type a post and press Enter to see the trigger, event type, actor, location, target, time, scale and stage. An empty line quits.
- **forecast:** for each region, prints the probability that escalation **starts within the next 3 days**, plus an ALARM if it is at or above the threshold (0.5).

Things to know before presenting:
- On real text the extractor is right about 63% of the time at the exact-span level. Triggers are reliable (F1 0.78); targets, times and crowd sizes are the weak spots.
- In the sample CSV, Gurugram gets a false alarm on the last day. It built up but never escalated, which is the hardest case. The tour says this explicitly.

---

## 5. Running it on your own posts

Make a CSV with these three columns (more columns are ignored):

```csv
timestamp,region,text
2026-02-01T09:15:00,Pune,Auto drivers are angry about the new parking fines
2026-02-01T18:40:00,Pune,Auto drivers call for a strike on Friday #justice
2026-02-02T11:05:00,Indore,Water shortage again in our area...
```

- You need at least 3 days of posts.
- `region` can be any name; posts are grouped by it.
- Then run `.venv\Scripts\python demo.py forecast my_posts.csv`.

From Python:
```python
from src.forecast import load_bundle, extract_post, forecast_posts
import pandas as pd
bundle = load_bundle()
print(extract_post(bundle, "Students plan a protest in Delhi tomorrow"))
print(forecast_posts(bundle, pd.read_csv("my_posts.csv")))
```

The extractor was adapted with only 80 real items. Adding more labelled
real posts to `data\real\annotated_real.json` (same format, `"split": "train"`)
and re-running step 3 is the most effective way to improve it.

---

## 6. Changing settings

```powershell
.venv\Scripts\python -m src.pipeline --threshold 0.3   # earlier alarms, more false alarms
.venv\Scripts\python -m src.pipeline --episodes 8      # more data (slower)
.venv\Scripts\python -m src.pipeline --folds 10        # more cross-validation folds
.venv\Scripts\python -m src.pipeline --seed 7          # a different random dataset
.venv\Scripts\python -m src.pipeline --epochs 4        # more transformer fine-tuning
.venv\Scripts\python -m src.real_pipeline --min-violent 3 --out results\real_strict   # stricter escalation
.venv\Scripts\python -m src.real_pipeline --split 2026-08-15                         # different test period
```

Running with a different `--seed` is a good robustness check. If the
conclusions hold, they don't depend on one lucky dataset.

Where to change things in the code:

| To change… | Edit |
|---|---|
| Trigger words, place names | `src\extraction\lexicon.py` |
| CRF features | `src\extraction\features.py` |
| Transformer model, epochs, learning rate | `src\extraction\transformer_tagger.py` |
| Real annotated data | `data\real\annotated_real.json` |
| Real escalation definition, test period | `src\real_pipeline.py` (command-line options) |
| Stage classifier, cue words | `src\extraction\stage.py` |
| Window size, label definition, features | `src\prediction\aggregate.py` |
| Models and their settings | `src\prediction\model.py` (`make_models`) |
| Metrics | `src\evaluation\metrics.py` |
| Synthetic data (templates, region types) | `data\generate_synthetic_data.py` |
| Challenge set | `data\challenge_set.json` (add posts with labelled phrases) |

After changing anything, re-run steps 3–6 of section 2.

---

## 7. What the tests check

| Test | Guards against |
|---|---|
| tokenizer | "BREAKING:" kept as one token (the old cause of "BREAKING:" tagged as an actor) |
| gold labels | Missing trigger labels ("riot", "tear gas") and split multi-word actors |
| dataset | Episodes, held-out templates and fizzle regions all present |
| no leakage | Any forecastable window containing escalation posts |
| label meaning | A positive label that doesn't mean "onset within the next 3 days" |
| span scores | The strict exact-match metric being computed wrongly |
| Stage 1 → Stage 2 | The extractor's output not feeding the forecaster |
| GRU gradients | Backpropagation maths, checked against numerical gradients |
| all models | Any model failing to train or giving invalid probabilities |
| lead time | Wrong lead-time or false-alarm arithmetic |
| challenge set | A hand label that doesn't match its sentence |
| time and scale roles | TIME and SCALE missing from the gold labels or the event frame |
| event frame | The event type not read off the trigger |
| BIO repair | The transformer producing an I- tag with no B- tag before it |
| headline recovery | A news URL slug not turned into a readable headline |
| multi-onset windows | Real-data labels or eligibility wrong when a state escalates more than once |
| real annotations | A real annotated phrase that doesn't match its text |

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'numpy'` (or similar) | You used plain `python`. Use `.venv\Scripts\python`, or redo section 1. |
| `ModuleNotFoundError: No module named 'src'` | Run from the project folder, with `-m src.pipeline`. |
| `No trained models at ...pipeline.joblib` | Run `.venv\Scripts\python -m src.pipeline` first. |
| `'python' is not recognized` | Install Python from python.org and tick "Add to PATH". |
| Numbers differ from the README | Check you didn't pass `--quick`, `--seed` or `--episodes`. Defaults reproduce the README. |
| Odd characters (`�`) in the output | Cosmetic only: the Windows console can't show some characters in the sample posts. |
| `Can't load the model for 'cardiffnlp/twitter-roberta-base'` | The model download failed. Retry on a stable connection, or put the four files in `models_cache\twitter-roberta-base\` (section 1). |
| Pipeline log stops with no error | The process was killed, usually by sleep. Keep the laptop awake and re-run. |
| `real_pipeline` says the GDELT file is missing | Run `.venv\Scripts\python data\real\fetch_gdelt.py`. |

---

## 9. Questions evaluators are likely to ask

- **Is the data real?** Partly. Extraction is tested on 80 real held-out items (Mastodon posts and GDELT headlines), and forecasting is tested on six months of real GDELT events in India. Synthetic data is used for training and controlled experiments, where full gold labels exist.
- **Why Mastodon, not Twitter/X or Reddit?** Twitter/X's API is paid and Reddit refuses unauthenticated requests (HTTP 403). Mastodon's public API serves the same kind of short, informal text. Only posts from accounts that opted in to search indexing are used, with no usernames stored.
- **Why news headlines for real forecasting?** GDELT links each event to a news article, not a post. Headlines are recovered from the article URLs and go through the same Stage 1 as posts.
- **Why a transformer and a CRF?** The brief asks for a fine-tuned transformer; the CRF is the baseline it has to beat. The adapted transformer is best on real text (0.63 vs 0.58).
- **What helps extraction most?** Real training data. Trained on synthetic posts alone, both models fall to ~0.4 F1 on real text; 80 real items lift them to 0.58–0.63.
- **How do you avoid leakage?**
  - Stage 1 is trained on different regions (episode 0) from the ones Stage 2 is evaluated on.
  - Synthetic cross-validation is grouped by region; real data is split by time.
  - Windows that already contain escalation are removed, and tests check this.
- **Why is real forecasting so weak?** Real escalations often come with no build-up in the news, headlines report events after they happen, and GDELT's labels are noisy. Using GDELT's own event codes as features does only slightly better, so the limit is predictability, not our extraction.
- **Which model is best?** Synthetic: gradient boosting (F1 0.93, 5% false alarms). Real: the GRU has the best average precision (0.13 vs 0.04 chance); logistic regression warns before the most onsets (13 of 19) but with many false alarms.
- **Why is the synthetic lead time only ~2 days?** The generator starts on-ground activity 3 days before onset, so ~2 days is the most the data allows.

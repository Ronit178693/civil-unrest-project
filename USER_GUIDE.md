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
- You never need to "activate" the environment. Every command below calls `.venv\Scripts\python` directly.

---

## 2. Run everything (in this order)

| # | Command | Time | What it does |
|---|---|---|---|
| 1 | `.venv\Scripts\python data\generate_synthetic_data.py` | 5 s | Builds the dataset and `data\sample_posts.csv` |
| 2 | `.venv\Scripts\python -m src.pipeline` | ~2 min | Trains Stage 1 and Stage 2, evaluates both, saves the models |
| 3 | `.venv\Scripts\python src\make_figures.py` | 5 s | Draws 5 figures into `paper\figures\` |
| 4 | `.venv\Scripts\python -m pytest tests -v` | 15 s | Runs the 11 automated tests |
| 5 | `.venv\Scripts\python demo.py` | 5 s | Guided demo using the saved models |

- Step 2 must run before steps 3 and 5, because they read what it saves.
- Re-running gives identical numbers, because everything is seeded.
- For a fast check that nothing is broken, run `.venv\Scripts\python -m src.pipeline --quick` (~1 min, smaller data).

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
```
seen_templates    span F1 1.000 | stage accuracy 1.000
unseen_templates  span F1 1.000 | stage accuracy 0.877
challenge_set     span F1 0.779 | stage accuracy 0.708
```
- **Span F1:** an extracted phrase counts only if its start, end and type are all exactly right.
- **Seen and unseen templates:** synthetic posts from other episodes. Unseen means sentence patterns the model never trained on.
- **Challenge set:** 24 posts we wrote and labelled by hand, with new words (`data\challenge_set.json`). **This is the realistic number.**
- **Stage accuracy:** how often the post-stage classifier picks the right stage (grievance, mobilization, on-ground or escalation).

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
| avg_precision | Like ROC-AUC, but honest when positives are rare (here 5.8%). This is the main score. |
| detection_rate | Share of escalating regions flagged on or before the onset day |
| mean_lead_days | Average days between the first alarm and the onset |
| false_alarm_rate | Share of quiet regions that got any alarm at all |

### Files in `results\`
| File | Contents |
|---|---|
| `extraction_metrics.json` | Extraction scores per test set and per role |
| `extraction_report_*.txt` | Detailed per-label reports for the CRF and the stage classifier |
| `model_comparison.csv` | The Stage 2 table above |
| `oof_predictions.csv` | Every window's predicted probability (from the fold that held it out) |
| `lead_time_by_region.csv` | First alarm day and lead time per region and model |
| `feature_importances.json` | Which features gradient boosting relies on |
| `windows_extracted.csv`, `windows_gold.csv` | The forecasting dataset itself |
| `models\pipeline.joblib` | Trained CRF, stage classifier and best forecaster (used by `demo.py`) |
| `run_summary.json` | Settings and headline numbers of the last run |

---

## 4. Using the demo

```powershell
.venv\Scripts\python demo.py                                    # guided tour
.venv\Scripts\python demo.py extract                            # type your own posts
.venv\Scripts\python demo.py forecast data\sample_posts.csv     # forecast from a CSV
.venv\Scripts\python demo.py forecast data\sample_posts.csv --all   # every day, not just the latest
```

- **extract:** type a post and press Enter to see the trigger, actor, location, target and stage. An empty line quits.
- **forecast:** for each region, prints the probability that escalation **starts within the next 3 days**, plus an ALARM if it is at or above the threshold (0.5).

Things to know before presenting:
- The extractor knows place names from its gazetteer (`src\extraction\lexicon.py`) and trigger words from its lexicon. Actor or target phrases it has never seen ("jasmine farmers", "the import policy") are often missed. That's a known limitation, shown by the challenge set.
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

The models were trained on synthetic text. Expect weaker results on real
posts until they are retrained on real labelled data (see
`data\README_DATA_SOURCES.md`).

---

## 6. Changing settings

```powershell
.venv\Scripts\python -m src.pipeline --threshold 0.3   # earlier alarms, more false alarms
.venv\Scripts\python -m src.pipeline --episodes 8      # more data (slower)
.venv\Scripts\python -m src.pipeline --folds 10        # more cross-validation folds
.venv\Scripts\python -m src.pipeline --seed 7          # a different random dataset
```

Running with a different `--seed` is a good robustness check. If the
conclusions hold, they don't depend on one lucky dataset.

Where to change things in the code:

| To change… | Edit |
|---|---|
| Trigger words, place names | `src\extraction\lexicon.py` |
| CRF features | `src\extraction\features.py` |
| Stage classifier, cue words | `src\extraction\stage.py` |
| Window size, label definition, features | `src\prediction\aggregate.py` |
| Models and their settings | `src\prediction\model.py` (`make_models`) |
| Metrics | `src\evaluation\metrics.py` |
| Synthetic data (templates, region types) | `data\generate_synthetic_data.py` |
| Challenge set | `data\challenge_set.json` (add posts with labelled phrases) |

After changing anything, re-run steps 2–4 of section 2.

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

---

## 9. Questions evaluators are likely to ask

- **Why synthetic data?** We have no API access yet. The generator copies the real grievance → mobilization → on-ground → escalation process. The challenge set and the planned ACLED evaluation check how far that carries to real text.
- **Why a CRF, not a transformer?** It trains on a CPU in seconds, and its features are easy to explain. The challenge set shows its limit (unseen actors and targets), which is exactly why a fine-tuned transformer is the planned next step.
- **How do you avoid leakage?** There are three safeguards:
  - Stage 1 is trained on different regions (episode 0) from the ones Stage 2 is evaluated on.
  - Cross-validation is grouped by region, so no region appears in both train and test.
  - Windows after onset are removed, and a test checks that no forecastable window contains escalation posts.
- **Why is the lead time only ~2 days?** The generator starts on-ground activity 3 days before onset, so ~2 days is the most the data allows. Real lead time needs real data.
- **Which model is best?** Logistic regression has the highest average precision (0.97). Gradient boosting raises half as many false alarms (8% vs 15%). Which is better depends on whether missed warnings or false alarms cost more.
- **Does extraction actually help?** Yes. Forecasting from event features reaches AP 0.97, against 0.55 from post volume alone.

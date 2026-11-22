r"""
demo.py -- try the trained system yourself.

First train once:   .venv\Scripts\python -m src.pipeline

Then:
  .venv\Scripts\python demo.py                         # guided tour (good for presenting)
  .venv\Scripts\python demo.py extract                 # type posts, see what is extracted
  .venv\Scripts\python demo.py forecast data\sample_posts.csv   # escalation risk per region
  .venv\Scripts\python demo.py forecast my_posts.csv --all      # every day, not just the latest

A forecast CSV needs the columns: timestamp, region, text (one row per post,
at least 3 days of posts).
"""
import argparse
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.forecast import extract_post, forecast_posts, load_bundle  # noqa: E402

EXAMPLES = [
    "Students are calling for a protest against the fee hike tomorrow in Delhi. #justice",
    "Live: thousands of farmers have gathered for a march in Jaipur right now.",
    "Police used tear gas as the protest in Lucknow turned into a riot.",
    "Fed up with water shortage in Patna. Nobody listens anymore.",
]


def show_frame(text, frame):
    print(f"\n  POST:     {text}")
    for key in ("trigger", "actor", "location", "target", "stage"):
        print(f"  {key:9s} {frame.get(key) or '-'}")


def cmd_extract(bundle):
    print("Type a social-media post and press Enter (empty line to quit).")
    while True:
        try:
            text = input("\n> ").strip()
        except EOFError:
            break
        if not text:
            break
        show_frame(text, extract_post(bundle, text))


def cmd_forecast(bundle, path, show_all=False):
    posts = pd.read_csv(path)
    result = forecast_posts(bundle, posts)
    print(f"\nModel: {bundle['forecaster_name']} | alarm threshold: {bundle['threshold']}")
    print("probability = chance that escalation STARTS within the next 3 days\n")
    if show_all:
        print(result.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    else:
        latest = result.sort_values("alarm_date").groupby("region").tail(1)
        print(latest.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        for _, r in latest.iterrows():
            verdict = "ALARM: escalation likely soon" if r["alarm"] else "no alarm"
            print(f"  {r['region']}: {r['probability']:.0%} -> {verdict}")
    return result


def cmd_tour(bundle):
    print("=" * 70)
    print("1) EVENT EXTRACTION: what the system pulls out of single posts")
    print("=" * 70)
    for text in EXAMPLES:
        show_frame(text, extract_post(bundle, text))
    sample = os.path.join(ROOT, "data", "sample_posts.csv")
    print("\n" + "=" * 70)
    print("2) FORECASTING: escalation risk per region from a stream of posts")
    print("=" * 70)
    if not os.path.exists(sample):
        print("  data/sample_posts.csv not found; run: .venv\\Scripts\\python data\\generate_synthetic_data.py")
        return
    posts = pd.read_csv(sample)
    print(f"  data/sample_posts.csv: {len(posts)} posts from {', '.join(posts['region'].unique())}.")
    truth_path = sample.replace(".csv", "_truth.json")
    truth = json.load(open(truth_path)) if os.path.exists(truth_path) else {}
    result = cmd_forecast(bundle, sample)
    print("\n  Risk over the last 6 days:")
    tail = result.sort_values("alarm_date").groupby("region").tail(6)
    pivot = tail.pivot(index="alarm_date", columns="region", values="probability")
    print(pivot.to_string(float_format=lambda v: f"{v:.2f}"))
    if truth:
        print("\n  What really happened (the model never saw this):")
        for region, onset in truth.items():
            print(f"  {region}: " + (f"escalation started on {onset}" if onset
                                      else "built up, then fizzled out -- never escalated"))
        print("  An alarm on a region that never escalates is a FALSE ALARM -- the hardest case,\n"
              "  measured by the false-alarm rate in results/model_comparison.csv.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("extract", help="type posts and see the extracted event")
    f = sub.add_parser("forecast", help="escalation risk per region from a CSV of posts")
    f.add_argument("csv")
    f.add_argument("--all", action="store_true", help="show every day, not only the latest")
    sub.add_parser("tour", help="guided demo (default)")
    args = ap.parse_args()

    try:
        bundle = load_bundle()
    except FileNotFoundError as e:
        sys.exit(str(e))
    if args.cmd == "extract":
        cmd_extract(bundle)
    elif args.cmd == "forecast":
        cmd_forecast(bundle, args.csv, args.all)
    else:
        cmd_tour(bundle)


if __name__ == "__main__":
    main()

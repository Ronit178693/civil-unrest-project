"""
fetch_gdelt.py
==============
Downloads GDELT 1.0 daily event files and keeps the civil-unrest events
(CAMEO root code 14, "PROTEST") that GDELT geolocates in India.

GDELT is free and needs no API key. Each daily file is ~8 MB zipped; we
stream it, filter it in memory and keep only a few hundred rows a day.

Output
------
data/real/gdelt_india_protest_events.csv   one row per protest event
    event_id, date, event_code, root_code, actor1, actor1_type, actor2,
    num_mentions, num_sources, num_articles, tone, place, adm1, url
data/real/gdelt_india_daily_volume.csv     all GDELT events per state per day
    date, adm1, n_events        (a news-volume control for the baseline)

CAMEO codes used later (src/real_pipeline.py):
    140-144  non-violent protest: political dissent, demonstrate/rally, hunger
             strike, strike/boycott, obstruct/blockade
    145x     protest violently / riot  -> escalation ground truth

Run:  python data/real/fetch_gdelt.py --start 2026-05-01 --end 2026-10-31
Re-running resumes: finished days are cached in data/real/gdelt_cache/.
"""
import argparse
import csv
import datetime as dt
import io
import os
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "gdelt_cache")
URL = "https://data.gdeltproject.org/events/{d}.export.CSV.zip"

# GDELT 1.0 event file columns (0-based) that we keep
COLS = {"event_id": 0, "date": 1, "actor1": 6, "actor1_type": 12, "actor2": 16,
        "event_code": 26, "root_code": 28, "num_mentions": 31, "num_sources": 32,
        "num_articles": 33, "tone": 34, "place": 50, "country": 51, "adm1": 52, "url": 57}
EVENT_FIELDS = ["event_id", "date", "event_code", "root_code", "actor1", "actor1_type", "actor2",
                "num_mentions", "num_sources", "num_articles", "tone", "place", "adm1", "url"]
MAX_EVENT_AGE_DAYS = 2   # drop retrospective mentions of old events


def _fetch_day(day: dt.date, retries: int = 4):
    tag = day.strftime("%Y%m%d")
    ev_path = os.path.join(CACHE, f"{tag}.events.csv")
    vol_path = os.path.join(CACHE, f"{tag}.volume.csv")
    if os.path.exists(ev_path) and os.path.exists(vol_path):
        return tag, "cached"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(URL.format(d=tag),
                                         headers={"User-Agent": "civil-unrest-research/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                blob = r.read()
            break
        except Exception as e:  # network blip: back off and retry
            if attempt == retries - 1:
                return tag, f"failed: {e}"
            time.sleep(2 ** attempt)

    events, volume = [], {}
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        with z.open(z.namelist()[0]) as f:
            for line in io.TextIOWrapper(f, encoding="utf-8", errors="replace"):
                c = line.rstrip("\n").split("\t")
                if len(c) < 58 or c[COLS["country"]] != "IN":
                    continue
                try:
                    ev_date = dt.datetime.strptime(c[COLS["date"]], "%Y%m%d").date()
                except ValueError:
                    continue
                if (day - ev_date).days > MAX_EVENT_AGE_DAYS:
                    continue
                key = (ev_date.isoformat(), c[COLS["adm1"]])
                volume[key] = volume.get(key, 0) + 1
                if c[COLS["root_code"]] == "14":
                    row = {k: c[i] for k, i in COLS.items() if k != "country"}
                    row["date"] = ev_date.isoformat()
                    events.append(row)

    os.makedirs(CACHE, exist_ok=True)
    with open(ev_path + ".tmp", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EVENT_FIELDS)
        w.writeheader()
        w.writerows(events)
    with open(vol_path + ".tmp", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "adm1", "n_events"])
        for (d, a), n in sorted(volume.items()):
            w.writerow([d, a, n])
    os.replace(ev_path + ".tmp", ev_path)
    os.replace(vol_path + ".tmp", vol_path)
    return tag, f"{len(events)} protest events"


def merge(start: dt.date, end: dt.date):
    import pandas as pd
    days = [start + dt.timedelta(d) for d in range((end - start).days + 1)]
    ev, vol = [], []
    for d in days:
        tag = d.strftime("%Y%m%d")
        p = os.path.join(CACHE, f"{tag}.events.csv")
        if os.path.exists(p):
            ev.append(pd.read_csv(p, dtype=str))
            vol.append(pd.read_csv(os.path.join(CACHE, f"{tag}.volume.csv"), dtype=str))
    events = pd.concat(ev).drop_duplicates("event_id")
    events = events[(events["date"] >= start.isoformat()) & (events["date"] <= end.isoformat())]
    volume = pd.concat(vol)
    volume["n_events"] = volume["n_events"].astype(int)
    volume = volume.groupby(["date", "adm1"], as_index=False)["n_events"].sum()
    volume = volume[(volume["date"] >= start.isoformat()) & (volume["date"] <= end.isoformat())]
    events.sort_values(["date", "event_id"]).to_csv(
        os.path.join(HERE, "gdelt_india_protest_events.csv"), index=False)
    volume.sort_values(["date", "adm1"]).to_csv(
        os.path.join(HERE, "gdelt_india_daily_volume.csv"), index=False)
    return events, volume


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-05-01")
    ap.add_argument("--end", default="2026-10-31")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    start, end = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
    # events are filed on the day GDELT adds them, so read one extra day past the end
    days = [start + dt.timedelta(d) for d in range((end - start).days + 2)]
    t0, failed = time.time(), []
    with ThreadPoolExecutor(a.workers) as pool:
        futs = {pool.submit(_fetch_day, d): d for d in days}
        for i, fut in enumerate(as_completed(futs), 1):
            tag, status = fut.result()
            if status.startswith("failed"):
                failed.append(tag)
            print(f"[{i}/{len(days)}] {tag}: {status}  ({time.time() - t0:.0f}s)", flush=True)
    if failed:
        print(f"{len(failed)} days failed: {failed}. Re-run to retry.")
        sys.exit(1)
    events, volume = merge(start, end)
    print(f"Wrote {len(events)} protest events and {len(volume)} state-day volume rows.")


if __name__ == "__main__":
    main()

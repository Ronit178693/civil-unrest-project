"""
fetch_mastodon.py
=================
Collects real, informal social-media posts about civil unrest from the
public Mastodon API (no key needed) for evaluating event extraction on
real text.

Why Mastodon: Twitter/X and Reddit no longer allow unauthenticated access
(Reddit answers 403, the X API is paid), and the project has no keys yet.
Mastodon posts are the same kind of short, informal, hashtag-heavy text.

Ethics / privacy
----------------
  * Only posts from accounts that opted in to search indexing
    (account.indexable == true) and are not bots.
  * We keep the post text, its id, instance and timestamp. No usernames,
    display names or profile data are stored; @mentions and links are
    removed from the text.
  * English only; content-warned (spoiler) and sensitive posts are skipped.

Output: data/real/mastodon_posts.csv  (post_id, instance, created_at, tag, text)

Run:  python data/real/fetch_mastodon.py --per-tag 200
"""
import argparse
import csv
import html
import json
import os
import re
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
INSTANCE = "mastodon.social"
TAGS = ["protest", "protests", "strike", "generalstrike", "demonstration", "riot",
        "farmersprotest", "picketline", "walkout", "unrest", "rally", "india"]
UNREST_WORDS = re.compile(
    r"\b(protest\w*|strike\w*|march\w*|rall(y|ies)|riot\w*|demonstrat\w*|clash\w*|"
    r"walk-?out|picket\w*|blockade\w*|sit-in|bandh|unrest|tear gas|arrest\w*|police)\b", re.I)

TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+(@[\w.]+)?")


def clean(content_html: str) -> str:
    text = content_html.replace("<br>", " ").replace("<br />", " ").replace("</p><p>", " ")
    text = html.unescape(TAG_RE.sub("", text))
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _get_json(url: str, retries: int = 5):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "civil-unrest-research/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(3 * 2 ** attempt)


def fetch_tag(tag: str, want: int, max_pages: int = 40):
    out, max_id = [], None
    for _ in range(max_pages):
        if len(out) >= want:
            break
        q = {"limit": 40}
        if max_id:
            q["max_id"] = max_id
        batch = _get_json(f"https://{INSTANCE}/api/v1/timelines/tag/{tag}?{urllib.parse.urlencode(q)}")
        if not batch:
            break
        max_id = batch[-1]["id"]
        for s in batch:
            acc = s["account"]
            if s.get("reblog") or s["language"] != "en" or not acc.get("indexable") \
                    or acc.get("bot") or s.get("sensitive") or s.get("spoiler_text"):
                continue
            text = clean(s["content"])
            if len(text.split()) < 8 or len(text) > 500 or not UNREST_WORDS.search(text):
                continue
            if len(out) >= want:
                break
            out.append({"post_id": s["id"], "instance": INSTANCE, "created_at": s["created_at"],
                        "tag": tag, "text": text})
        time.sleep(1.1)   # stay well under the 300 requests / 5 min public limit
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tag", type=int, default=200)
    a = ap.parse_args(argv)
    rows, seen = [], set()
    cache_dir = os.path.join(HERE, "mastodon_cache")
    os.makedirs(cache_dir, exist_ok=True)
    for tag in TAGS:
        cache = os.path.join(cache_dir, f"{tag}.json")
        if os.path.exists(cache):                    # resume after a network failure
            got = json.load(open(cache, encoding="utf-8"))
        else:
            got = fetch_tag(tag, a.per_tag)
            json.dump(got, open(cache, "w", encoding="utf-8"))
        new = [r for r in got if r["text"].lower() not in seen]
        seen.update(r["text"].lower() for r in new)
        rows.extend(new)
        print(f"#{tag}: {len(new)} posts", flush=True)
    path = os.path.join(HERE, "mastodon_posts.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["post_id", "instance", "created_at", "tag", "text"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} posts to {os.path.relpath(path)}")


if __name__ == "__main__":
    main()

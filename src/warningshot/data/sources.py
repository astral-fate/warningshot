"""Cached clients for the two confirmed-public data sources.

Both APIs are keyless. Responses are cached to disk on first fetch and read
from disk thereafter, for three reasons:

  1. Reproducibility. A reviewer re-running this months later gets the same
     numbers as the paper, not whatever the APIs say that day. Hacker News
     scores in particular keep accruing — the survey's own replication of
     CeSIA's "1,522 points" landed at 1,632 for exactly this reason.
  2. Politeness. GDELT already returned HTTP 429 during exploration; the
     same courtesy applies here.
  3. Offline test runs.

Delete the cache directory to force a refresh.
"""

import json
import os
import time

import requests

from warningshot import paths

CACHE_DIR = paths.CACHE
UA = "WarningShotAttentionStudy/1.0 (AI Incident Response Sprint; research use)"
WIKI = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
HN = "https://hn.algolia.com/api/v1/search"


class FetchError(RuntimeError):
    """Raised when a source cannot be fetched and no cache entry exists."""


def _cache_path(key):
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
    return str(CACHE_DIR / (safe + ".json"))


def _cached(key, fetch, use_cache=True):
    path = _cache_path(key)
    if use_cache and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    payload = fetch()
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return payload


def pageviews(article, start, end, project="en.wikipedia",
              access="all-access", agent="user", use_cache=True, retries=6):
    """Daily pageviews as {YYYYMMDD: views}.

    agent="user" excludes traffic Wikimedia has flagged as automated, which is
    the correct choice for a study about human attention.
    """
    key = f"pv_{project}_{access}_{agent}_{article}_{start}_{end}"

    def fetch():
        url = f"{WIKI}/{project}/{access}/{agent}/{article}/daily/{start}/{end}"
        last = None
        for attempt in range(retries):
            r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
            if r.status_code == 200:
                return r.json().get("items", [])
            if r.status_code == 404:
                # A page with no data in the range is a real answer, not a failure.
                return []
            last = r.status_code
            # Wikimedia returns 429 under burst load. Exponential backoff, since
            # the placebo null needs many series and will otherwise trip it.
            time.sleep(min(30, 2 ** attempt))
        raise FetchError(f"pageviews {article}: HTTP {last} after {retries} tries")

    items = _cached(key, fetch, use_cache)
    return {i["timestamp"][:8]: i["views"] for i in items}


def hn_stories(query, start_epoch, end_epoch, hits=100, use_cache=True):
    """Hacker News stories in a time window, sorted by points descending."""
    key = f"hn_{query}_{start_epoch}_{end_epoch}"

    def fetch():
        r = requests.get(
            HN,
            params={
                "query": query,
                "tags": "story",
                "hitsPerPage": hits,
                "numericFilters": f"created_at_i>{start_epoch},created_at_i<{end_epoch}",
            },
            headers={"User-Agent": UA},
            timeout=30,
        )
        if r.status_code != 200:
            raise FetchError(f"hn '{query}': HTTP {r.status_code}")
        return r.json().get("hits", [])

    items = _cached(key, fetch, use_cache)
    rows = [
        {
            "title": h.get("title"),
            "points": h.get("points") or 0,
            "comments": h.get("num_comments") or 0,
            "date": (h.get("created_at") or "")[:10],
            "story_id": h.get("objectID"),
            "url": f"https://news.ycombinator.com/item?id={h.get('objectID')}",
        }
        for h in items
        if h.get("title")
    ]
    return sorted(rows, key=lambda r: -r["points"])


def hn_comment_times(story_id, use_cache=True, max_pages=40):
    """Unix timestamps of every comment on one story, oldest first.

    Algolia caps a single query at 1,000 hits, so this walks backwards in time
    with a created_at_i ceiling rather than paging. Without that the newest
    1,000 comments are returned and the early hours — the part that carries the
    decay signal — are silently missing.
    """
    key = f"hncomments_{story_id}"

    def fetch():
        seen, ceiling = {}, None
        for _ in range(max_pages):
            nf = f"created_at_i<{ceiling}" if ceiling else None
            params = {"tags": f"comment,story_{story_id}", "hitsPerPage": 1000}
            if nf:
                params["numericFilters"] = nf
            r = requests.get(HN + "_by_date", params=params,
                             headers={"User-Agent": UA}, timeout=30)
            if r.status_code != 200:
                raise FetchError(f"hn comments {story_id}: HTTP {r.status_code}")
            hits = r.json().get("hits", [])
            fresh = {h["objectID"]: h["created_at_i"] for h in hits
                     if h.get("created_at_i") and h["objectID"] not in seen}
            if not fresh:
                break
            seen.update(fresh)
            ceiling = min(fresh.values())
            if len(hits) < 1000:
                break
            time.sleep(0.2)
        return sorted(seen.values())

    return _cached(key, fetch, use_cache)


def gdelt_timeline(query, start, end, mode="timelinevol", use_cache=True, retries=5):
    """GDELT DOC 2.0 daily coverage volume as {YYYYMMDD: value}.

    `timelinevol` is a *percentage of all monitored coverage*, not an article
    count, so it is comparable across time but never a volume in absolute
    terms. GDELT rate-limits hard (one request every five seconds by its own
    error text) and serves a rolling ~3-month window, so history older than
    that is simply unavailable.
    """
    key = f"gdelt_{mode}_{query}_{start}_{end}"

    def fetch():
        last = None
        for attempt in range(retries):
            r = requests.get(
                "https://api.gdeltproject.org/api/v2/doc/doc",
                params={"query": query, "mode": mode, "format": "json",
                        "startdatetime": start + "000000",
                        "enddatetime": end + "000000"},
                headers={"User-Agent": "Mozilla/5.0 (compatible; " + UA + ")"},
                timeout=60,
            )
            if r.status_code == 200 and r.text.strip().startswith("{"):
                tl = r.json().get("timeline", [])
                return tl[0].get("data", []) if tl else []
            last = r.status_code
            time.sleep(6 * (attempt + 1))
        raise FetchError(f"gdelt '{query}': HTTP {last} after {retries} tries")

    pts = _cached(key, fetch, use_cache)
    return {p["date"][:8]: p["value"] for p in pts if p.get("date")}

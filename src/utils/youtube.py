import os
import re
import requests

_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
_MIN_DURATION_SECS = 60  # filter out Shorts


def _parse_duration(duration: str) -> int:
    """Convert ISO 8601 duration (PT4M13S) to total seconds."""
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration)
    if not m:
        return 0
    h = int(m.group(1) or 0)
    mins = int(m.group(2) or 0)
    s = int(m.group(3) or 0)
    return h * 3600 + mins * 60 + s


def _format_duration(seconds: int) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def fetch_youtube_videos(query: str) -> list[dict]:
    """
    Search YouTube for the query, filter out Shorts, return top 3 as dicts.
    Returns [] silently on any error (API failure, missing key, quota, etc.)

    Each returned dict:
        video_id, title, channel, thumbnail, duration_str
    """
    api_key = os.getenv("YOUTUBE_API_KEY", "")
    if not api_key:
        return []

    try:
        # Step 1: search.list — top 5 results
        search_resp = requests.get(_SEARCH_URL, params={
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": 5,
            "key": api_key,
        }, timeout=5)
        search_resp.raise_for_status()
        items = search_resp.json().get("items", [])
        if not items:
            return []

        # Extract IDs + metadata
        candidates = []
        for item in items:
            vid_id = item["id"]["videoId"]
            snippet = item["snippet"]
            candidates.append({
                "video_id": vid_id,
                "title": snippet.get("title", ""),
                "channel": snippet.get("channelTitle", ""),
                "thumbnail": snippet.get("thumbnails", {}).get("medium", {}).get("url", ""),
            })

        # Step 2: videos.list — get durations to filter Shorts
        ids_param = ",".join(c["video_id"] for c in candidates)
        dur_resp = requests.get(_VIDEOS_URL, params={
            "part": "contentDetails",
            "id": ids_param,
            "key": api_key,
        }, timeout=5)
        dur_resp.raise_for_status()
        dur_items = {
            v["id"]: v["contentDetails"]["duration"]
            for v in dur_resp.json().get("items", [])
        }

        # Filter and annotate
        results = []
        for c in candidates:
            raw_dur = dur_items.get(c["video_id"], "PT0S")
            secs = _parse_duration(raw_dur)
            if secs < _MIN_DURATION_SECS:
                continue
            c["duration_str"] = _format_duration(secs)
            results.append(c)
            if len(results) == 3:
                break

        return results

    except Exception:
        return []

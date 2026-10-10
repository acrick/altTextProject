#!/usr/bin/env python3
"""band_research.py — Look up a band's current lineup and per-member reference
photos (via Claude + web search), cache the result, and let the user manually
add or correct members.

Used by generate_alt_gallery.py when --band is passed. Not meant to be run
directly, but everything here is written to be testable in isolation.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlparse

import anthropic
import requests

MEDIA_TYPE_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

MIN_PHOTO_BYTES = 2000  # anything smaller is almost certainly an icon/placeholder, not a real photo

BAND_RESEARCH_MODEL = "claude-sonnet-5"
VALIDATION_MODEL = "claude-sonnet-5"

BAND_RESEARCH_PROMPT_TEMPLATE = """Search the web to research the musical act "{band_name}". \
It may be a band or a solo artist.

Find:
1. The act's Wikipedia page URL, if one exists.
2. A short one-line description (genre, hometown, year formed or active) if available.
3. Whether this is a solo artist (one named performer, e.g. Sting) or a band.
4. The CURRENT lineup: each member's name and their role or instrument. For a \
solo artist, list the artist themself FIRST (role = what they sing/play), \
then their current touring/backing musicians if you can find them.
5. For each person, if you can find one, a direct URL to a photo of \
just that person — an official press photo, a photo from the act's own \
site, or a social media profile photo. Only include a URL if you're \
reasonably confident it shows that specific person individually, not a \
full-band group shot, a logo, or an unrelated image. It's fine to leave \
photo_url as null for members you can't find an individual photo for.

Respond with ONLY valid JSON, no markdown formatting, no explanation, in \
exactly this shape:
{{
  "band_name": "...",
  "wikipedia_url": "... or null",
  "description": "...",
  "is_solo_artist": true or false,
  "members": [
    {{"name": "...", "role": "...", "photo_url": "... or null"}}
  ]
}}

If you cannot find a band or artist by this name at all, respond with exactly:
{{"error": "not_found"}}

If there are multiple different, similarly-known bands or artists with this \
name and you can't tell which one is meant, respond with exactly:
{{"error": "ambiguous", "candidates": ["short description of each band", "..."]}}
"""

FACE_CHECK_PROMPT = (
    "Does this image clearly show one identifiable person's face, suitable "
    "as a reference photo for recognizing them in other photos? It should "
    "not be a logo, a group photo, an illustration, or a heavily obscured "
    "or very low quality image. Answer with only the single word YES or NO."
)


def slugify(name: str) -> str:
    """'The Lashes' -> 'the-lashes'. Used for cache folder and file names."""
    slug = name.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug or "band"


def _band_dir(cache_dir: Path, slug: str) -> Path:
    return cache_dir / slug


def _cache_path(cache_dir: Path, slug: str) -> Path:
    return _band_dir(cache_dir, slug) / "band-info.json"


def load_band_cache(cache_dir: Path, slug: str) -> dict | None:
    path = _cache_path(cache_dir, slug)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def is_cache_fresh(band_info: dict, max_age_days: int = 90) -> bool:
    fetched = band_info.get("fetched_date")
    if not fetched:
        return False
    try:
        fetched_date = datetime.strptime(fetched, "%Y-%m-%d").date()
    except ValueError:
        return False
    return (date.today() - fetched_date).days < max_age_days


def _save_band_cache(cache_dir: Path, slug: str, band_info: dict) -> None:
    _band_dir(cache_dir, slug).mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, slug)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(band_info, f, indent=2, ensure_ascii=False)


def _guess_extension(url: str) -> str:
    path = urlparse(url).path
    suffix = Path(path).suffix.lower()
    return suffix if suffix in MEDIA_TYPE_MAP else ".jpg"


def download_reference_photo(url: str, dest_path: Path, timeout: int = 15) -> bool:
    """Download a candidate reference photo. Returns False (and leaves no
    file behind) if it's not a real, reasonably-sized image."""
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "alt-text-gallery-bot/1.0"},
        )
        resp.raise_for_status()
    except requests.RequestException:
        return False

    content_type = resp.headers.get("Content-Type", "")
    if not content_type.startswith("image/"):
        return False
    if len(resp.content) < MIN_PHOTO_BYTES:
        return False

    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(resp.content)
    except OSError:
        return False
    return True


def validate_reference_photo(client: anthropic.Anthropic, photo_path: Path) -> bool:
    """Ask Claude vision to sanity-check that a downloaded image is actually
    a usable individual reference photo, not a logo/group shot/junk."""
    suffix = photo_path.suffix.lower()
    media_type = MEDIA_TYPE_MAP.get(suffix, "image/jpeg")
    try:
        with open(photo_path, "rb") as f:
            import base64

            image_data = base64.standard_b64encode(f.read()).decode("utf-8")
    except OSError:
        return False

    try:
        response = client.messages.create(
            model=VALIDATION_MODEL,
            max_tokens=10,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": FACE_CHECK_PROMPT},
                    ],
                }
            ],
        )
    except anthropic.APIError:
        return False

    for block in response.content:
        if block.type == "text":
            return block.text.strip().upper().startswith("YES")
    return False


def _extract_json_from_text_blocks(content_blocks) -> dict | None:
    """Claude's response mixes text, server_tool_use, and web_search_tool_result
    blocks when web search runs. We want the JSON that appears in a text
    block — usually the last one, but we search from the end to be safe."""
    text_blocks = [b.text for b in content_blocks if getattr(b, "type", None) == "text"]
    for raw in reversed(text_blocks):
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            continue
    return None


def _call_band_research_api(client: anthropic.Anthropic, band_name: str) -> dict:
    prompt = BAND_RESEARCH_PROMPT_TEMPLATE.format(band_name=band_name)
    messages = [{"role": "user", "content": prompt}]
    tools = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 6}]

    # The API can pause a long-running search turn (stop_reason "pause_turn").
    # Continuing means sending the paused assistant message back unchanged.
    for _ in range(3):
        response = client.messages.create(
            model=BAND_RESEARCH_MODEL,
            max_tokens=2000,
            tools=tools,
            messages=messages,
        )
        if response.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": response.content})
            continue
        break
    else:
        return {"error": "search_timed_out"}

    data = _extract_json_from_text_blocks(response.content)
    if data is None:
        return {"error": "unparseable_response"}
    return data


def _print_band_summary(band_info: dict) -> None:
    name = band_info.get("band_name", "?")
    wiki = band_info.get("wikipedia_url") or "(no Wikipedia page found)"
    members = band_info.get("members", [])
    with_photos = sum(1 for m in members if m.get("reference_photo"))
    label = "Artist" if band_info.get("is_solo_artist") else "Band"
    print(f"\n{label}: {name}")
    print(f"  Wikipedia: {wiki}")
    print(f"  Members found: {len(members)} ({with_photos} with a usable reference photo)")
    for m in members:
        photo_flag = "photo ok" if m.get("reference_photo") else "no photo"
        print(f"    - {m.get('name', '?')} ({m.get('role', '?')}) [{photo_flag}]")
    print()


def research_band(
    client: anthropic.Anthropic,
    band_name: str,
    cache_dir: Path,
    force_refresh: bool = False,
    max_age_days: int = 90,
) -> dict:
    """Main entry point. Returns a BandInfo dict. Exits with a clear error
    for not-found/ambiguous bands rather than silently continuing."""
    slug = slugify(band_name)

    if not force_refresh:
        cached = load_band_cache(cache_dir, slug)
        if cached and is_cache_fresh(cached, max_age_days):
            print(f"Using cached band info for '{band_name}' (fetched {cached['fetched_date']}).")
            _print_band_summary(cached)
            return cached

    print(f"Researching band/artist: {band_name} ...")
    raw = _call_band_research_api(client, band_name)

    if "error" in raw:
        err = raw["error"]
        if err == "not_found":
            raise SystemExit(
                f"Error: could not find a band or artist called '{band_name}'. "
                "Check the spelling, or try --add-member to add people manually."
            )
        if err == "ambiguous":
            candidates = raw.get("candidates", [])
            lines = "\n".join(f"  - {c}" for c in candidates)
            raise SystemExit(
                f"Error: multiple bands/artists found matching '{band_name}':\n{lines}\n"
                "Try a more specific name (e.g. add a city or genre)."
            )
        raise SystemExit(f"Error: band/artist research failed ({err}). Try again, or use --add-member.")

    band_dir = _band_dir(cache_dir, slug)
    ref_dir = band_dir / "reference-photos"
    ref_dir.mkdir(parents=True, exist_ok=True)

    members = []
    for m in raw.get("members", []):
        name = (m.get("name") or "").strip()
        if not name:
            continue
        role = (m.get("role") or "").strip()
        photo_url = m.get("photo_url")
        entry = {
            "name": name,
            "role": role,
            "reference_photo": None,
            "source": "web_search",
            "source_url": photo_url,
        }
        if photo_url:
            ext = _guess_extension(photo_url)
            dest = ref_dir / f"{slugify(name)}{ext}"
            print(f"  Downloading reference photo for {name} ...", end=" ", flush=True)
            if download_reference_photo(photo_url, dest):
                if validate_reference_photo(client, dest):
                    entry["reference_photo"] = str(dest.relative_to(cache_dir))
                    print("ok")
                else:
                    dest.unlink(missing_ok=True)
                    print("rejected (doesn't look like a clear individual photo)")
            else:
                print("failed (couldn't download)")
        members.append(entry)

    band_info = {
        "band_name": raw.get("band_name") or band_name,
        "wikipedia_url": raw.get("wikipedia_url"),
        "description": raw.get("description", ""),
        "is_solo_artist": bool(raw.get("is_solo_artist")),
        "fetched_date": date.today().isoformat(),
        "members": members,
    }

    _save_band_cache(cache_dir, slug, band_info)
    _print_band_summary(band_info)
    return band_info


def add_manual_member(
    cache_dir: Path,
    band_name: str,
    name: str,
    role: str,
    photo_path: str,
) -> dict:
    """Implements --add-member. Adds a new member or overwrites an existing
    one (matched by name, case-insensitive) in that band's cache."""
    slug = slugify(band_name)
    src = Path(photo_path).expanduser()
    if not src.exists():
        raise SystemExit(f"Error: --add-member photo not found: {photo_path}")

    band_info = load_band_cache(cache_dir, slug) or {
        "band_name": band_name,
        "wikipedia_url": None,
        "description": "",
        "fetched_date": date.today().isoformat(),
        "members": [],
    }

    ref_dir = _band_dir(cache_dir, slug) / "reference-photos"
    ref_dir.mkdir(parents=True, exist_ok=True)
    suffix = src.suffix.lower() if src.suffix.lower() in MEDIA_TYPE_MAP else ".jpg"
    dest = ref_dir / f"{slugify(name)}{suffix}"
    shutil.copy2(src, dest)

    new_entry = {
        "name": name.strip(),
        "role": role.strip(),
        "reference_photo": str(dest.relative_to(cache_dir)),
        "source": "manual",
        "source_url": None,
    }

    members = band_info.get("members", [])
    replaced = False
    for i, existing in enumerate(members):
        if existing.get("name", "").strip().lower() == name.strip().lower():
            members[i] = new_entry
            replaced = True
            break
    if not replaced:
        members.append(new_entry)
    band_info["members"] = members

    _save_band_cache(cache_dir, slug, band_info)
    action = "Updated" if replaced else "Added"
    print(f"{action} member '{name}' ({role}) for band '{band_info['band_name']}'.")
    return band_info

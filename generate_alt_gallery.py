#!/usr/bin/env python3
"""generate_alt_gallery.py — Scan a folder of images, generate two alt text descriptions
per image via Claude vision, and output a static HTML gallery with a JSON cache.

Usage:
    python generate_alt_gallery.py
    python generate_alt_gallery.py --input ./images --output ./site
    python generate_alt_gallery.py --input ./images --output ./site --force

Optional band member identification (see README / PROJECT_VISION.md for details):
    python generate_alt_gallery.py --band "The Lashes"
    python generate_alt_gallery.py --band "The Lashes" --refresh-band
    python generate_alt_gallery.py --artist "Sting"
    python generate_alt_gallery.py --band "The Lashes" --add-member "Jane Doe:guitarist:./jane.jpg"
"""

import argparse
import base64
import html
import json
import shutil
from collections import Counter
from pathlib import Path

import anthropic

import band_research

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

MEDIA_TYPE_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

ALT_TEXT_RULES = (
    "Generate two distinct, accessible alt text descriptions for this image for use with a screen reader.\n\n"
    "Rules:\n"
    "- Describe the meaningful visual content\n"
    "- Avoid starting with 'image of' or 'photo of' unless genuinely needed for clarity\n"
    "- Keep each under 150 characters unless the image is genuinely complex\n"
    "- Do not invent details that are not visually evident\n"
    "- Be specific about what matters most to a person who cannot see the image\n"
    "- The two descriptions must differ in focus or phrasing — not be near-duplicates\n"
)

ALT_TEXT_PROMPT = ALT_TEXT_RULES + (
    "\nRespond with valid JSON only, no markdown, no explanation:\n"
    "{\"alt1\": \"first description\", \"alt2\": \"second description\"}"
)

# Used instead of ALT_TEXT_PROMPT when --band is given and the band has at
# least one known lineup member (with or without a reference photo).
MEMBER_ALT_TEXT_PROMPT_TEMPLATE = ALT_TEXT_RULES + (
    "\nThis photo is from a performance by {act_description}. Their "
    "full current lineup is:\n{lineup_text}\n\n"
    "Use two ways to identify people, in order of reliability:\n"
    "1. Face match — a person in the actual photo matches one of the "
    "reference photos shown above. This is the strongest signal.\n"
    "2. Role match — a person is clearly playing an instrument or filling a "
    "role marked \"unique role in this lineup\" above. Only do this for "
    "roles marked unique — never guess between two lineup members who share "
    "a role (e.g. two guitarists) based on role alone.\n\n"
    "{naming_rule} Anyone you don't recognize "
    "or can't confidently place stays generic (e.g. \"a drummer\"), in the "
    "same sentence if needed.\n\n"
    "For every person you name, rate your confidence honestly:\n"
    "- \"high\": a clear face match to a reference photo\n"
    "- \"medium\": identified by a unique role only, or a weak/partial face "
    "match\n"
    "- \"low\": you're naming them but you're genuinely unsure\n"
    "Also record how you identified them: \"face\", \"role\", or \"both\". "
    "These ratings are only for the photographer's own review and are never "
    "shown as-is in the alt text itself.\n\n"
    "Respond with valid JSON only, no markdown, no explanation:\n"
    "{{\"alt1\": \"first description\", \"alt2\": \"second description\", "
    "\"members_detected\": [{{\"name\": \"...\", \"role\": \"...\", "
    "\"confidence\": \"high|medium|low\", \"matched_by\": \"face|role|both\"}}]}}"
)


def scan_images(input_dir: Path) -> list[Path]:
    found = set()
    for ext in SUPPORTED_EXTENSIONS:
        found.update(input_dir.glob(f"*{ext}"))
        found.update(input_dir.glob(f"*{ext.upper()}"))
    return sorted(found, key=lambda p: p.name.lower())


def load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        try:
            with open(cache_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_cache(cache_path: Path, cache: dict) -> None:
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _image_block(image_path: Path) -> dict:
    suffix = image_path.suffix.lower()
    media_type = MEDIA_TYPE_MAP[suffix]
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": image_data},
    }


def _parse_alt_response(response) -> tuple[str, str, list[dict]]:
    """Shared JSON-from-response parsing for both generation functions below.
    Returns (alt1, alt2, members_detected) — members_detected is [] for the
    plain (no-band) case, since that prompt never asks for it."""
    for block in response.content:
        if block.type == "text":
            raw = block.text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            data = json.loads(raw)
            alt1 = data.get("alt1", "").strip()
            alt2 = data.get("alt2", "").strip()
            members_detected = []
            for m in data.get("members_detected", []) or []:
                if isinstance(m, dict) and m.get("name"):
                    matched_by = str(m.get("matched_by", "")).strip().lower()
                    if matched_by not in ("face", "role", "both"):
                        matched_by = "unknown"
                    members_detected.append({
                        "name": str(m.get("name", "")).strip(),
                        "role": str(m.get("role", "")).strip(),
                        "confidence": str(m.get("confidence", "low")).strip().lower(),
                        "matched_by": matched_by,
                    })
            return alt1, alt2, members_detected

    return "", "", []


def generate_alt_text(client: anthropic.Anthropic, image_path: Path) -> tuple[str, str]:
    """Original, unchanged behavior — used whenever --band is not passed."""
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=400,
        messages=[
            {
                "role": "user",
                "content": [_image_block(image_path), {"type": "text", "text": ALT_TEXT_PROMPT}],
            }
        ],
    )
    alt1, alt2, _ = _parse_alt_response(response)
    return alt1, alt2


def _role_uniqueness(members: list[dict]) -> dict[str, bool]:
    """Maps each member's name -> True if their role string is unique across
    the lineup (case-insensitive exact match), False if 2+ members share it.
    Computed in plain Python rather than left to the model, so 'unique role'
    claims in the prompt are guaranteed correct, not just plausible."""
    roles = [(m.get("name", "").strip(), (m.get("role", "").strip().lower() or "member")) for m in members]
    role_counts = Counter(role for _, role in roles)
    return {name: role_counts[role] == 1 for name, role in roles if name}


def generate_alt_text_with_members(
    client: anthropic.Anthropic,
    image_path: Path,
    band_info: dict,
    band_cache_dir: Path,
) -> tuple[str, str, list[dict]]:
    """Same as generate_alt_text, but gives Claude the band's full lineup and
    (where available) reference photos, so it can name people either by face
    match or — for roles no one else in the lineup shares — by role alone.
    One API call per photo, same as the no-band path, just richer."""
    all_members = [m for m in band_info.get("members", []) if m.get("name", "").strip()]
    if not all_members:
        # No lineup info at all — nothing to reason about, behave like the plain path.
        alt1, alt2 = generate_alt_text(client, image_path)
        return alt1, alt2, []

    uniqueness = _role_uniqueness(all_members)

    content: list[dict] = []
    lineup_lines: list[str] = []

    for member in all_members:
        name = member["name"].strip()
        role = member.get("role", "").strip() or "member"
        ref_rel = member.get("reference_photo")
        ref_path = band_cache_dir / ref_rel if ref_rel else None
        has_photo = bool(ref_path and ref_path.exists())

        if has_photo:
            content.append({"type": "text", "text": f"Reference photo of {name}, {role}:"})
            content.append(_image_block(ref_path))
            photo_note = "reference photo shown above"
        else:
            photo_note = "no reference photo available"

        role_note = (
            "unique role in this lineup"
            if uniqueness.get(name)
            else "role shared by multiple lineup members — do not identify by role alone"
        )
        lineup_lines.append(f"- {name} ({role}) — {photo_note}; {role_note}")

    content.append({"type": "text", "text": "Actual photo to describe:"})
    content.append(_image_block(image_path))

    act_name = band_info.get("band_name", "the band")
    if band_info.get("is_solo_artist"):
        act_description = f"the solo artist \"{act_name}\""
        naming_rule = (
            f"Name {act_name} plainly, e.g. \"{act_name} sings...\" — not "
            f"\"... of {act_name}\". Name backing musicians as their role, then "
            f"their name, then the artist, e.g. \"Bass guitarist Jane Doe, "
            f"performing with {act_name}, ...\". If more than one named person "
            f"appears in the same description, mention {act_name} once."
        )
    else:
        act_description = f"the band \"{act_name}\""
        naming_rule = (
            "When you name someone, phrase it as their role, then their name, then "
            f"the band, e.g. \"Bass guitarist Jane Doe of {act_name} performs...\". "
            "If more than one named person appears in the same description, state "
            "the band name once, not for every person."
        )
    prompt = MEMBER_ALT_TEXT_PROMPT_TEMPLATE.format(
        act_description=act_description,
        naming_rule=naming_rule,
        lineup_text="\n".join(lineup_lines),
    )
    content.append({"type": "text", "text": prompt})

    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=600,
        messages=[{"role": "user", "content": content}],
    )
    return _parse_alt_response(response)


def build_html(entries: list[dict]) -> str:
    rows_html = ""
    for entry in entries:
        img_src = f"images/{entry['filename']}"
        alt_escaped = html.escape(entry["alt1"], quote=True)
        filename_display = html.escape(entry["filename"])
        alt1_display = html.escape(entry["alt1"])
        alt2_display = html.escape(entry["alt2"])
        rows_html += f"""
    <div class="row">
      <div class="thumb-cell">
        <img src="{img_src}" alt="{alt_escaped}">
      </div>
      <div class="filename-cell">
        <p>{filename_display}</p>
      </div>
      <div class="alt-cell">
        <p>{alt1_display}</p>
        <p>{alt2_display}</p>
      </div>
    </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Alt Text Gallery</title>
  <style>
    *, *::before, *::after {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      background-color: #000;
      color: #fff;
      font-family: "SF Pro Display", "SF Pro Text", -apple-system, BlinkMacSystemFont,
                   "Segoe UI", sans-serif;
      padding: 2.5rem 2rem;
      line-height: 1.6;
    }}

    h1 {{
      font-size: 1.25rem;
      font-weight: 500;
      letter-spacing: 0.02em;
      margin-bottom: 2rem;
      color: #fff;
    }}

    .gallery {{
      border-top: 2px solid #333;
    }}

    .row {{
      display: flex;
      align-items: stretch;
      border-bottom: 2px solid #333;
    }}

    .thumb-cell {{
      flex: 0 0 220px;
      width: 220px;
      padding: 1.25rem;
      border-right: 2px solid #333;
      display: flex;
      align-items: center;
      justify-content: center;
    }}

    .thumb-cell img {{
      display: block;
      width: 100%;
      height: auto;
      object-fit: contain;
    }}

    .filename-cell {{
      flex: 0 0 180px;
      width: 180px;
      padding: 1.25rem 1rem;
      border-right: 2px solid #333;
      display: flex;
      align-items: flex-start;
      padding-top: 1.5rem;
    }}

    .filename-cell p {{
      font-size: 0.7rem;
      color: #888;
      line-height: 1.55;
      word-break: break-all;
    }}

    .alt-cell {{
      flex: 1;
      padding: 1.25rem 1.75rem;
      display: flex;
      flex-direction: column;
      justify-content: center;
      gap: 0.875rem;
    }}

    .alt-cell p {{
      font-size: 0.9375rem;
      color: #fff;
      line-height: 1.75;
    }}

    @media (max-width: 760px) {{
      body {{
        padding: 1.25rem 1rem;
      }}

      .row {{
        flex-direction: column;
      }}

      .thumb-cell {{
        flex: none;
        width: 100%;
        border-right: none;
        border-bottom: 2px solid #333;
      }}

      .filename-cell {{
        flex: none;
        width: 100%;
        border-right: none;
        border-bottom: 2px solid #333;
        padding: 0.75rem 1.25rem;
      }}

      .alt-cell {{
        padding: 1rem 1.25rem;
      }}
    }}
  </style>
</head>
<body>
  <h1>Alt Text Gallery</h1>
  <div class="gallery">{rows_html}
  </div>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate alt text for images using Claude vision and output an HTML gallery."
    )
    parser.add_argument(
        "--input", default="./images", metavar="DIR",
        help="Folder containing source images (JPG, JPEG, PNG, WEBP) [default: ./images]"
    )
    parser.add_argument(
        "--output", default="./site", metavar="DIR",
        help="Folder for the generated site (index.html, alt-text.json, images/) [default: ./site]"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate alt text even if a cached result already exists"
    )
    parser.add_argument(
        "--band", "--artist", dest="band", default=None, metavar="NAME",
        help="Band or solo artist name (e.g. --band \"Devo\" or --artist \"Sting\"). Enables looking up members and naming them in alt text when recognized."
    )
    parser.add_argument(
        "--refresh-band", "--refresh-artist", dest="refresh_band", action="store_true",
        help="Force a fresh lookup of band member info, ignoring the 90-day cache. Requires --band."
    )
    parser.add_argument(
        "--add-member", action="append", default=[], metavar="NAME:ROLE:PHOTO_PATH",
        help="Manually add or override a band member, e.g. 'Jane Doe:guitarist:./jane.jpg'. "
             "Applied after the automatic lookup. Requires --band. Repeatable."
    )
    args = parser.parse_args()

    if args.refresh_band and not args.band:
        raise SystemExit("Error: --refresh-band requires --band")
    if args.add_member and not args.band:
        raise SystemExit("Error: --add-member requires --band")

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_dir.exists():
        raise SystemExit(f"Error: input folder not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    cache_path = output_dir / "alt-text.json"
    cache = load_cache(cache_path)

    image_paths = scan_images(input_dir)
    if not image_paths:
        raise SystemExit(f"No supported images found in: {input_dir}")

    print(f"Found {len(image_paths)} image(s) in {input_dir}")

    client = anthropic.Anthropic()

    band_info = None
    band_cache_dir = None
    if args.band:
        band_cache_dir = Path("band-cache").resolve()
        band_cache_dir.mkdir(parents=True, exist_ok=True)
        band_info = band_research.research_band(
            client, args.band, band_cache_dir, force_refresh=args.refresh_band
        )
        for spec in args.add_member:
            parts = spec.split(":", 2)
            if len(parts) != 3:
                raise SystemExit(
                    f"Error: --add-member must look like 'Name:role:photo-path', got: {spec}"
                )
            member_name, member_role, member_photo = (p.strip() for p in parts)
            band_info = band_research.add_manual_member(
                band_cache_dir, args.band, member_name, member_role, member_photo
            )

    entries: list[dict] = []

    for img_path in image_paths:
        filename = img_path.name
        cached = cache.get(filename)

        if not args.force and isinstance(cached, dict) and "alt1" in cached and "alt2" in cached:
            alt1, alt2 = cached["alt1"], cached["alt2"]
            members_detected = cached.get("members_detected", [])
            status = "cached"
        else:
            print(f"  Generating: {filename} ...", end=" ", flush=True)
            if band_info:
                alt1, alt2, members_detected = generate_alt_text_with_members(
                    client, img_path, band_info, band_cache_dir
                )
            else:
                alt1, alt2 = generate_alt_text(client, img_path)
                members_detected = []

            cache_entry = {"alt1": alt1, "alt2": alt2}
            if members_detected:
                cache_entry["members_detected"] = members_detected
            cache[filename] = cache_entry
            save_cache(cache_path, cache)
            status = "done"

        shutil.copy2(img_path, images_dir / filename)
        entries.append({"filename": filename, "alt1": alt1, "alt2": alt2})

        detail = ""
        if members_detected:
            names = ", ".join(
                f"{m['name']} ({m['confidence']}/{m.get('matched_by', '?')})" for m in members_detected
            )
            detail = f" [members: {names}]"
        print(f"  [{status}] {filename}{detail}")

    index_path = output_dir / "index.html"
    index_path.write_text(build_html(entries), encoding="utf-8")

    print(f"\nOutput written to: {output_dir}")
    print(f"  index.html     — gallery page")
    print(f"  alt-text.json  — cached alt text ({len(entries)} entries)")
    print(f"  images/        — {len(entries)} copied images")


if __name__ == "__main__":
    main()

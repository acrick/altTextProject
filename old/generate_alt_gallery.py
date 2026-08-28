#!/usr/bin/env python3
"""generate_alt_gallery.py — Scan a folder of images, generate two alt text descriptions
per image via Claude vision, and output a static HTML gallery with a JSON cache.

Usage:
    python generate_alt_gallery.py
    python generate_alt_gallery.py --input ./images --output ./site
    python generate_alt_gallery.py --input ./images --output ./site --force
"""

import argparse
import base64
import html
import json
import shutil
from pathlib import Path

import anthropic

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

MEDIA_TYPE_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

ALT_TEXT_PROMPT = (
    "Generate two distinct, accessible alt text descriptions for this image for use with a screen reader.\n\n"
    "Rules:\n"
    "- Describe the meaningful visual content\n"
    "- Avoid starting with 'image of' or 'photo of' unless genuinely needed for clarity\n"
    "- Keep each under 150 characters unless the image is genuinely complex\n"
    "- Do not invent details that are not visually evident\n"
    "- Be specific about what matters most to a person who cannot see the image\n"
    "- The two descriptions must differ in focus or phrasing — not be near-duplicates\n\n"
    "Respond with valid JSON only, no markdown, no explanation:\n"
    "{\"alt1\": \"first description\", \"alt2\": \"second description\"}"
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


def generate_alt_text(client: anthropic.Anthropic, image_path: Path) -> tuple[str, str]:
    suffix = image_path.suffix.lower()
    media_type = MEDIA_TYPE_MAP[suffix]

    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=400,
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
                    {"type": "text", "text": ALT_TEXT_PROMPT},
                ],
            }
        ],
    )

    for block in response.content:
        if block.type == "text":
            raw = block.text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            data = json.loads(raw)
            return data["alt1"].strip(), data["alt2"].strip()

    return "", ""


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
    args = parser.parse_args()

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

    entries: list[dict] = []

    for img_path in image_paths:
        filename = img_path.name
        cached = cache.get(filename)

        if not args.force and isinstance(cached, dict) and "alt1" in cached and "alt2" in cached:
            alt1, alt2 = cached["alt1"], cached["alt2"]
            status = "cached"
        else:
            print(f"  Generating: {filename} ...", end=" ", flush=True)
            alt1, alt2 = generate_alt_text(client, img_path)
            cache[filename] = {"alt1": alt1, "alt2": alt2}
            save_cache(cache_path, cache)
            status = "done"

        shutil.copy2(img_path, images_dir / filename)
        entries.append({"filename": filename, "alt1": alt1, "alt2": alt2})
        print(f"  [{status}] {filename}")

    index_path = output_dir / "index.html"
    index_path.write_text(build_html(entries), encoding="utf-8")

    print(f"\nOutput written to: {output_dir}")
    print(f"  index.html     — gallery page")
    print(f"  alt-text.json  — cached alt text ({len(entries)} entries)")
    print(f"  images/        — {len(entries)} copied images")


if __name__ == "__main__":
    main()

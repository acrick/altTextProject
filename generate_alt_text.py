#!/usr/bin/env python3
"""Generate alt text for JPG/PNG images using the Claude API."""

import argparse
import base64
import sys
from pathlib import Path

import anthropic

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

SYSTEM_PROMPT = (
    "You are an accessibility expert. Generate concise, descriptive alt text for images. "
    "Alt text should describe the meaningful content and context of the image in one to two sentences. "
    "Be specific and objective. Do not start with 'Image of' or 'Photo of'."
)


def generate_alt_text(client: anthropic.Anthropic, image_path: Path) -> str:
    ext = image_path.suffix.lower()
    media_type = MEDIA_TYPES[ext]

    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=256,
        system=SYSTEM_PROMPT,
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
                    {"type": "text", "text": "Generate alt text for this image."},
                ],
            }
        ],
    )

    for block in response.content:
        if block.type == "text":
            return block.text.strip()

    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate alt text for images using Claude")
    parser.add_argument("images", nargs="+", help="Image file paths")
    args = parser.parse_args()

    client = anthropic.Anthropic()
    exit_code = 0

    for image_arg in args.images:
        image_path = Path(image_arg)

        if not image_path.exists():
            print(f"Error: {image_path} not found", file=sys.stderr)
            exit_code = 1
            continue

        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(
                f"Error: {image_path} — unsupported format (supported: {', '.join(SUPPORTED_EXTENSIONS)})",
                file=sys.stderr,
            )
            exit_code = 1
            continue

        alt_text = generate_alt_text(client, image_path)
        print(f"{image_path}\n{alt_text}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

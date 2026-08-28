#!/usr/bin/env python3
"""Not a unit test — runs the real main() flow end to end with the Anthropic
client and requests.get mocked out, so we can eyeball the actual files it
produces (band-info.json, alt-text.json, index.html) without hitting the
real network. Prints the resulting files at the end."""

import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))
import generate_alt_gallery as gag  # noqa: E402  (this also imports band_research, needed for the patch() targets below)


class FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


def fake_msg(obj):
    return SimpleNamespace(content=[FakeTextBlock(json.dumps(obj))], stop_reason="end_turn")


def main():
    workdir = Path(tempfile.mkdtemp())
    images_dir = workdir / "images"
    images_dir.mkdir()
    output_dir = workdir / "site"

    # two fake source photos
    for name in ["show1_01.jpg", "show1_02.jpg"]:
        (images_dir / name).write_bytes(b"\xff\xd8\xff" + b"0" * 200)

    band_research_reply = fake_msg({
        "band_name": "The Lashes",
        "wikipedia_url": "https://en.wikipedia.org/wiki/The_Lashes",
        "description": "Rock band",
        "members": [
            {"name": "Jane Doe", "role": "bassist", "photo_url": "https://example.com/jane.jpg"},
            {"name": "Sam Roe", "role": "drummer", "photo_url": None},
        ],
    })
    alt_text_reply_1 = fake_msg({
        "alt1": "Bass guitarist Jane Doe of The Lashes performs on a smoke-filled stage under red light.",
        "alt2": "Jane Doe plays bass guitar while the crowd watches from behind a barrier.",
        "members_detected": [
            {"name": "Jane Doe", "role": "bassist", "confidence": "high", "matched_by": "face"}
        ],
    })
    alt_text_reply_2 = fake_msg({
        "alt1": "Drummer Sam Roe of The Lashes plays behind a kit lit in blue stage lighting.",
        "alt2": "Sam Roe keeps time on drums during a live set.",
        "members_detected": [
            {"name": "Sam Roe", "role": "drummer", "confidence": "medium", "matched_by": "role"}
        ],
    })
    face_check_reply = SimpleNamespace(content=[FakeTextBlock("YES")])

    alt_text_calls = []

    def fake_create(**kwargs):
        if "tools" in kwargs:
            return band_research_reply
        if kwargs.get("max_tokens") == 10:
            return face_check_reply
        # Anything else is a per-photo alt-text call — dispatch by call order,
        # tracked separately so band-research/face-check calls don't throw off the count.
        alt_text_calls.append(kwargs)
        return alt_text_reply_1 if len(alt_text_calls) == 1 else alt_text_reply_2

    fake_client = MagicMock()
    fake_client.messages.create.side_effect = fake_create

    def fake_get(url, timeout=None, headers=None):
        resp = MagicMock()
        resp.headers = {"Content-Type": "image/jpeg"}
        resp.content = b"\xff\xd8\xff" + b"1" * 5000
        resp.raise_for_status = MagicMock()
        return resp

    argv = [
        "generate_alt_gallery.py",
        "--input", str(images_dir),
        "--output", str(output_dir),
        "--band", "The Lashes",
    ]

    with patch("sys.argv", argv), \
         patch("anthropic.Anthropic", return_value=fake_client), \
         patch("band_research.requests.get", side_effect=fake_get), \
         patch("pathlib.Path.cwd", return_value=workdir):
        # band-cache/ is written relative to cwd inside main() via Path("band-cache").resolve()
        import os
        old_cwd = os.getcwd()
        os.chdir(workdir)
        try:
            gag.main()
        finally:
            os.chdir(old_cwd)

    print("=== band-cache/the-lashes/band-info.json ===")
    print((workdir / "band-cache" / "the-lashes" / "band-info.json").read_text())

    print("\n=== site/alt-text.json ===")
    print((output_dir / "alt-text.json").read_text())

    print("\n=== files on disk ===")
    for p in sorted(workdir.rglob("*")):
        if p.is_file():
            print(" ", p.relative_to(workdir))

    shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()

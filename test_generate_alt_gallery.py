#!/usr/bin/env python3
"""Unit tests for generate_alt_gallery.py's new band-aware logic. Mocks the
Anthropic client entirely — no real network needed."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import generate_alt_gallery as gag


class FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


def fake_response(json_obj):
    return SimpleNamespace(content=[FakeTextBlock(json.dumps(json_obj))])


class TestParseAltResponse(unittest.TestCase):
    def test_plain_response_no_members(self):
        resp = fake_response({"alt1": "A", "alt2": "B"})
        alt1, alt2, members = gag._parse_alt_response(resp)
        self.assertEqual((alt1, alt2, members), ("A", "B", []))

    def test_response_with_members(self):
        resp = fake_response({
            "alt1": "Jane plays guitar",
            "alt2": "A guitarist on stage",
            "members_detected": [
                {"name": "Jane Doe", "role": "guitarist", "confidence": "HIGH", "matched_by": "FACE"}
            ],
        })
        alt1, alt2, members = gag._parse_alt_response(resp)
        self.assertEqual(
            members,
            [{"name": "Jane Doe", "role": "guitarist", "confidence": "high", "matched_by": "face"}],
        )

    def test_invalid_matched_by_becomes_unknown(self):
        resp = fake_response({
            "alt1": "A", "alt2": "B",
            "members_detected": [{"name": "Jane Doe", "role": "guitarist", "confidence": "high"}],
        })
        _, _, members = gag._parse_alt_response(resp)
        self.assertEqual(members[0]["matched_by"], "unknown")

    def test_ignores_malformed_member_entries(self):
        resp = fake_response({"alt1": "A", "alt2": "B", "members_detected": [{"role": "no name here"}, "junk"]})
        _, _, members = gag._parse_alt_response(resp)
        self.assertEqual(members, [])

    def test_markdown_fenced(self):
        block = FakeTextBlock('```json\n{"alt1": "A", "alt2": "B"}\n```')
        resp = SimpleNamespace(content=[block])
        alt1, alt2, members = gag._parse_alt_response(resp)
        self.assertEqual((alt1, alt2), ("A", "B"))


class TestRoleUniqueness(unittest.TestCase):
    def test_unique_roles(self):
        members = [{"name": "Jane", "role": "bassist"}, {"name": "Sam", "role": "drummer"}]
        result = gag._role_uniqueness(members)
        self.assertTrue(result["Jane"])
        self.assertTrue(result["Sam"])

    def test_shared_role_marked_not_unique(self):
        members = [
            {"name": "Alex", "role": "guitarist"},
            {"name": "Chris", "role": "guitarist"},
            {"name": "Sam", "role": "drummer"},
        ]
        result = gag._role_uniqueness(members)
        self.assertFalse(result["Alex"])
        self.assertFalse(result["Chris"])
        self.assertTrue(result["Sam"])

    def test_role_matching_is_case_insensitive(self):
        members = [{"name": "Alex", "role": "Guitarist"}, {"name": "Chris", "role": "guitarist"}]
        result = gag._role_uniqueness(members)
        self.assertFalse(result["Alex"])
        self.assertFalse(result["Chris"])

    def test_missing_role_treated_as_shared_generic_member(self):
        members = [{"name": "Alex", "role": ""}, {"name": "Chris", "role": ""}]
        result = gag._role_uniqueness(members)
        self.assertFalse(result["Alex"])
        self.assertFalse(result["Chris"])


class TestGenerateAltTextWithMembers(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.band_cache_dir = self.tmp / "band-cache"
        self.image_path = self.tmp / "photo.jpg"
        self.image_path.write_bytes(b"\xff\xd8\xff" + b"0" * 100)  # fake jpeg-ish bytes

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_falls_back_to_plain_only_when_no_members_at_all(self):
        band_info = {"members": []}
        client = MagicMock()
        client.messages.create.return_value = fake_response({"alt1": "A", "alt2": "B"})

        alt1, alt2, members = gag.generate_alt_text_with_members(
            client, self.image_path, band_info, self.band_cache_dir
        )
        self.assertEqual((alt1, alt2, members), ("A", "B", []))
        self.assertEqual(client.messages.create.call_count, 1)

    def test_member_without_photo_still_included_via_lineup_text(self):
        # No reference_photo at all — should NOT fall back to plain anymore,
        # since role-based identification doesn't need a photo.
        band_info = {"members": [{"name": "Jane Doe", "role": "guitarist", "reference_photo": None}]}
        client = MagicMock()
        client.messages.create.return_value = fake_response({
            "alt1": "Guitarist Jane Doe of The Lashes performs",
            "alt2": "Jane Doe plays guitar",
            "members_detected": [
                {"name": "Jane Doe", "role": "guitarist", "confidence": "medium", "matched_by": "role"}
            ],
        })
        alt1, alt2, members = gag.generate_alt_text_with_members(
            client, self.image_path, band_info, self.band_cache_dir
        )
        self.assertIn("Jane Doe", alt1)
        self.assertEqual(members[0]["matched_by"], "role")

        # the lineup text (not an image) should mention Jane Doe even with no photo
        sent_content = client.messages.create.call_args.kwargs["messages"][0]["content"]
        image_blocks = [b for b in sent_content if b.get("type") == "image"]
        text_blocks = [b["text"] for b in sent_content if b.get("type") == "text"]
        self.assertEqual(len(image_blocks), 1)  # only the target photo, no reference photo
        self.assertTrue(any("Jane Doe" in t and "no reference photo available" in t for t in text_blocks))

    def test_attaches_reference_photo_when_present(self):
        ref_dir = self.band_cache_dir / "the-lashes" / "reference-photos"
        ref_dir.mkdir(parents=True)
        ref_photo = ref_dir / "jane-doe.jpg"
        ref_photo.write_bytes(b"\xff\xd8\xff" + b"1" * 100)

        band_info = {
            "band_name": "The Lashes",
            "members": [{
                "name": "Jane Doe",
                "role": "guitarist",
                "reference_photo": "the-lashes/reference-photos/jane-doe.jpg",
            }]
        }
        client = MagicMock()
        client.messages.create.return_value = fake_response({
            "alt1": "Guitarist Jane Doe of The Lashes performs",
            "alt2": "Guitarist performs",
            "members_detected": [
                {"name": "Jane Doe", "role": "guitarist", "confidence": "high", "matched_by": "face"}
            ],
        })

        alt1, alt2, members = gag.generate_alt_text_with_members(
            client, self.image_path, band_info, self.band_cache_dir
        )
        self.assertIn("Jane Doe", alt1)
        self.assertEqual(members[0]["name"], "Jane Doe")

        # verify the request actually included the reference photo + the target photo (2 images total)
        sent_content = client.messages.create.call_args.kwargs["messages"][0]["content"]
        image_blocks = [b for b in sent_content if b.get("type") == "image"]
        self.assertEqual(len(image_blocks), 2)

    def test_missing_reference_photo_file_on_disk_falls_back_to_role_only(self):
        # band-info.json references a photo path that doesn't actually exist on disk —
        # should still be treated as "no photo available" (role-only), not crash.
        band_info = {
            "members": [{
                "name": "Jane Doe",
                "role": "guitarist",
                "reference_photo": "the-lashes/reference-photos/does-not-exist.jpg",
            }]
        }
        client = MagicMock()
        client.messages.create.return_value = fake_response({"alt1": "A", "alt2": "B"})
        alt1, alt2, members = gag.generate_alt_text_with_members(
            client, self.image_path, band_info, self.band_cache_dir
        )
        self.assertEqual((alt1, alt2, members), ("A", "B", []))
        sent_content = client.messages.create.call_args.kwargs["messages"][0]["content"]
        image_blocks = [b for b in sent_content if b.get("type") == "image"]
        self.assertEqual(len(image_blocks), 1)  # only target photo, no broken reference image sent

    def test_shared_role_members_marked_as_not_unique_in_prompt(self):
        band_info = {
            "band_name": "The Lashes",
            "members": [
                {"name": "Alex", "role": "guitarist", "reference_photo": None},
                {"name": "Chris", "role": "guitarist", "reference_photo": None},
            ],
        }
        client = MagicMock()
        client.messages.create.return_value = fake_response({"alt1": "A", "alt2": "B"})
        gag.generate_alt_text_with_members(client, self.image_path, band_info, self.band_cache_dir)

        sent_content = client.messages.create.call_args.kwargs["messages"][0]["content"]
        prompt_text = next(b["text"] for b in sent_content if "Respond with valid JSON" in b.get("text", ""))
        lineup_lines = [line for line in prompt_text.splitlines() if line.startswith("- Alex") or line.startswith("- Chris")]
        self.assertEqual(len(lineup_lines), 2)
        for line in lineup_lines:
            self.assertIn("role shared by multiple lineup members", line)
            self.assertNotIn("unique role in this lineup", line)


class TestCliValidation(unittest.TestCase):
    """Runs the actual script with --help / bad flag combos as a subprocess,
    since argparse errors happen before any of our code runs."""

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "generate_alt_gallery.py", *args],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
        )

    def test_refresh_band_without_band_errors(self):
        result = self.run_cli("--refresh-band")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--refresh-band requires --band", result.stdout + result.stderr)

    def test_add_member_without_band_errors(self):
        result = self.run_cli("--add-member", "Jane:guitarist:./x.jpg")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--add-member requires --band", result.stdout + result.stderr)

    def test_help_still_works(self):
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("--band", result.stdout)


if __name__ == "__main__":
    unittest.main()

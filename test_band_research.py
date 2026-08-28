#!/usr/bin/env python3
"""Unit tests for band_research.py. No real network or API calls — everything
that talks to the internet or the Anthropic API is mocked, since this needs
to run in environments (like this one) that can't reach arbitrary hosts."""

import json
import shutil
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import band_research as br


class FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class FakeOtherBlock:
    def __init__(self):
        self.type = "server_tool_use"


def fake_response(text_blocks, stop_reason="end_turn"):
    return SimpleNamespace(content=text_blocks, stop_reason=stop_reason)


class TestSlugify(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(br.slugify("The Lashes"), "the-lashes")

    def test_punctuation_and_case(self):
        self.assertEqual(br.slugify("Sigur Rós!!"), "sigur-r-s")

    def test_empty_falls_back(self):
        self.assertEqual(br.slugify("   "), "band")


class TestCacheFreshness(unittest.TestCase):
    def test_fresh(self):
        info = {"fetched_date": date.today().isoformat()}
        self.assertTrue(br.is_cache_fresh(info, max_age_days=90))

    def test_expired(self):
        old = (date.today() - timedelta(days=200)).isoformat()
        info = {"fetched_date": old}
        self.assertFalse(br.is_cache_fresh(info, max_age_days=90))

    def test_missing_date(self):
        self.assertFalse(br.is_cache_fresh({}, max_age_days=90))

    def test_malformed_date(self):
        self.assertFalse(br.is_cache_fresh({"fetched_date": "not-a-date"}, max_age_days=90))


class TestCacheRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_load_missing_returns_none(self):
        self.assertIsNone(br.load_band_cache(self.tmp, "no-such-band"))

    def test_save_then_load(self):
        info = {"band_name": "The Lashes", "fetched_date": date.today().isoformat(), "members": []}
        br._save_band_cache(self.tmp, "the-lashes", info)
        loaded = br.load_band_cache(self.tmp, "the-lashes")
        self.assertEqual(loaded["band_name"], "The Lashes")

    def test_corrupt_json_returns_none(self):
        band_dir = self.tmp / "the-lashes"
        band_dir.mkdir(parents=True)
        (band_dir / "band-info.json").write_text("{not valid json")
        self.assertIsNone(br.load_band_cache(self.tmp, "the-lashes"))


class TestExtractJson(unittest.TestCase):
    def test_plain_json(self):
        blocks = [FakeTextBlock('{"band_name": "X"}')]
        self.assertEqual(br._extract_json_from_text_blocks(blocks), {"band_name": "X"})

    def test_markdown_fenced_json(self):
        blocks = [FakeTextBlock('```json\n{"band_name": "X"}\n```')]
        self.assertEqual(br._extract_json_from_text_blocks(blocks), {"band_name": "X"})

    def test_prefers_last_valid_json_block(self):
        # First text block is Claude "thinking out loud" before the tool call,
        # not parseable as JSON — should be skipped in favor of the real answer.
        blocks = [
            FakeTextBlock("I'll search for this band now."),
            FakeOtherBlock(),
            FakeTextBlock('{"band_name": "Real Answer"}'),
        ]
        self.assertEqual(br._extract_json_from_text_blocks(blocks)["band_name"], "Real Answer")

    def test_no_valid_json_returns_none(self):
        blocks = [FakeTextBlock("no json here")]
        self.assertIsNone(br._extract_json_from_text_blocks(blocks))


class TestDownloadReferencePhoto(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @patch("band_research.requests.get")
    def test_successful_download(self, mock_get):
        mock_get.return_value = MagicMock(
            headers={"Content-Type": "image/jpeg"},
            content=b"x" * 5000,
        )
        mock_get.return_value.raise_for_status = MagicMock()
        dest = self.tmp / "photo.jpg"
        self.assertTrue(br.download_reference_photo("https://example.com/a.jpg", dest))
        self.assertTrue(dest.exists())

    @patch("band_research.requests.get")
    def test_rejects_non_image_content_type(self, mock_get):
        mock_get.return_value = MagicMock(
            headers={"Content-Type": "text/html"},
            content=b"x" * 5000,
        )
        mock_get.return_value.raise_for_status = MagicMock()
        dest = self.tmp / "photo.jpg"
        self.assertFalse(br.download_reference_photo("https://example.com/a.html", dest))
        self.assertFalse(dest.exists())

    @patch("band_research.requests.get")
    def test_rejects_too_small_file(self, mock_get):
        mock_get.return_value = MagicMock(
            headers={"Content-Type": "image/jpeg"},
            content=b"x" * 50,
        )
        mock_get.return_value.raise_for_status = MagicMock()
        dest = self.tmp / "photo.jpg"
        self.assertFalse(br.download_reference_photo("https://example.com/a.jpg", dest))

    @patch("band_research.requests.get")
    def test_handles_request_exception(self, mock_get):
        import requests

        mock_get.side_effect = requests.RequestException("boom")
        dest = self.tmp / "photo.jpg"
        self.assertFalse(br.download_reference_photo("https://example.com/a.jpg", dest))


class TestAddManualMember(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.photo = self.tmp / "source.jpg"
        self.photo.write_bytes(b"fake-image-bytes")
        self.cache_dir = self.tmp / "band-cache"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_add_new_member(self):
        info = br.add_manual_member(self.cache_dir, "The Lashes", "Jane Doe", "guitarist", str(self.photo))
        self.assertEqual(len(info["members"]), 1)
        self.assertEqual(info["members"][0]["source"], "manual")
        ref = self.cache_dir / info["members"][0]["reference_photo"]
        self.assertTrue(ref.exists())

    def test_overwrite_existing_member_by_name(self):
        br.add_manual_member(self.cache_dir, "The Lashes", "Jane Doe", "guitarist", str(self.photo))
        info = br.add_manual_member(self.cache_dir, "The Lashes", "jane doe", "lead guitarist", str(self.photo))
        self.assertEqual(len(info["members"]), 1)  # replaced, not duplicated
        self.assertEqual(info["members"][0]["role"], "lead guitarist")

    def test_missing_source_photo_raises(self):
        with self.assertRaises(SystemExit):
            br.add_manual_member(self.cache_dir, "The Lashes", "Jane Doe", "guitarist", "/no/such/file.jpg")


class TestResearchBandOrchestration(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_uses_fresh_cache_without_calling_api(self):
        info = {"band_name": "The Lashes", "fetched_date": date.today().isoformat(), "members": []}
        br._save_band_cache(self.tmp, "the-lashes", info)
        client = MagicMock()
        result = br.research_band(client, "The Lashes", self.tmp)
        client.messages.create.assert_not_called()
        self.assertEqual(result["band_name"], "The Lashes")

    def test_not_found_raises_systemexit(self):
        client = MagicMock()
        client.messages.create.return_value = fake_response(
            [FakeTextBlock('{"error": "not_found"}')]
        )
        with self.assertRaises(SystemExit):
            br.research_band(client, "Some Nonexistent Band", self.tmp)

    def test_ambiguous_raises_systemexit_with_candidates(self):
        client = MagicMock()
        client.messages.create.return_value = fake_response(
            [FakeTextBlock('{"error": "ambiguous", "candidates": ["Band A (UK)", "Band B (US)"]}')]
        )
        with self.assertRaises(SystemExit) as ctx:
            br.research_band(client, "Common Name", self.tmp)
        self.assertIn("Band A", str(ctx.exception))

    @patch("band_research.download_reference_photo", return_value=False)
    def test_member_without_downloadable_photo_still_included(self, mock_download):
        client = MagicMock()
        client.messages.create.return_value = fake_response(
            [FakeTextBlock(json.dumps({
                "band_name": "The Lashes",
                "wikipedia_url": "https://en.wikipedia.org/wiki/The_Lashes",
                "description": "Rock band",
                "members": [
                    {"name": "Jane Doe", "role": "guitarist", "photo_url": "https://example.com/jane.jpg"}
                ],
            }))]
        )
        result = br.research_band(client, "The Lashes", self.tmp)
        self.assertEqual(len(result["members"]), 1)
        self.assertIsNone(result["members"][0]["reference_photo"])

    def test_force_refresh_ignores_fresh_cache(self):
        info = {"band_name": "The Lashes", "fetched_date": date.today().isoformat(), "members": []}
        br._save_band_cache(self.tmp, "the-lashes", info)
        client = MagicMock()
        client.messages.create.return_value = fake_response(
            [FakeTextBlock(json.dumps({
                "band_name": "The Lashes", "wikipedia_url": None, "description": "", "members": []
            }))]
        )
        br.research_band(client, "The Lashes", self.tmp, force_refresh=True)
        client.messages.create.assert_called_once()


if __name__ == "__main__":
    unittest.main()

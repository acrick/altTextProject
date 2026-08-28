# Band member alt text — usage

## Setup

New dependency — reinstall requirements once:

```
.venv/bin/pip install -r requirements.txt
```

## Basic use

```
.venv/bin/python generate_alt_gallery.py --band "The Lashes"
```

First run for a band: prints what it found (Wikipedia page, member list, how
many got a usable photo) *before* processing any of your photos — check that
it found the right band. Ctrl+C if not.

Later runs for the same band reuse the cached lookup automatically (valid 90
days), so you won't re-search every show.

## Flags

| Flag | Purpose |
|---|---|
| `--band "Name"` | Enables the feature |
| `--refresh-band` | Force a fresh lookup even if the cache is still valid |
| `--add-member "Name:role:photo-path"` | Manually add or fix a member (repeatable) |

Example fixing a bad auto-match or adding a former member search missed:

```
.venv/bin/python generate_alt_gallery.py --band "The Lashes" \
  --add-member "Alex Old:former bassist:./photos/alex.jpg"
```

## How names get assigned

Two signals, in order of trust:

1. **Face match** — the photo is compared against that member's reference
   photo. Strongest signal, rated `high` confidence.
2. **Role match** — if a member has no findable reference photo (common for
   smaller bands), but they play the *only* instrument of its kind in the
   lineup (e.g. the one bassist), a clear shot of that instrument can name
   them on its own. Capped at `medium` confidence, and only used when the
   role is genuinely unique — two guitarists in a lineup never get guessed
   apart this way. Uniqueness is computed in plain Python before the prompt
   is built, not left for the model to judge.

Alt text reads like: *"Bass guitarist Jane Doe of The Lashes performs..."*
— role, name, band. If two named people appear in one description, the band
name is only stated once, not repeated per person.

Each `alt-text.json` entry also gets a `matched_by` field (`face`, `role`,
or `both`) next to `confidence`, so you can tell at a glance *why* a name
was assigned, not just how sure the tool was.

## What to check afterward

Open `band-cache/<band-slug>/band-info.json` — every named member in your
alt text has a `"confidence"` rating in `site/alt-text.json` next to their
name (`"high"`, `"medium"`, or `"low"`). The alt text itself never shows
this rating — it reads clean either way — but if you see `"low"`, look at
that photo. Reference photos used for matching are saved under
`band-cache/<band-slug>/reference-photos/` so you can see exactly who was
compared against what.

Nothing changes if you skip `--band` — same output as before.

## Testing this build

```
python3 -m unittest test_band_research test_generate_alt_gallery -v
python3 smoke_test.py     # full pipeline, mocked, prints the actual output files
```

**Important:** all of this was tested with the Claude API and web requests
mocked out — this sandbox can't reach Wikipedia or arbitrary photo hosts. Run
one real `--band` command yourself on an actual show before trusting it on a
full shoot, in case the live Wikipedia/web-search response shape differs from
what these tests assume.

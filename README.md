# Fast MTG Card Identifier

A multi-stage pipeline that identifies Magic: The Gathering cards from
photos. Designed to match or beat Claude's vision accuracy at a fraction of
the latency by combining cheap classical signals (perceptual hashing,
local features, OCR) with small specialist neural networks (~3M params each).

It is built for **real phone photos**, not just clean scans:

- **Any background.** Wood tables, dark play mats, busy play-mat art,
  notebook paper, cards in hands, cards in slabs or toploaders.
- **Glare.** Glare on sleeves and foils is detected and inpainted before
  matching, and the final check uses local features that ignore glare.
- **Perspective and rotation.** Cards shot at an angle, lying sideways or
  upside down.
- **Several cards per photo** (`--all`).

## Pipeline architecture

```
Photo
  ↓
[Card detection]  fine-tuned YOLO proposals  +  classical quad detector
  │                (both snapped onto the real card edges; perspective-correct)
  ↓                → one outline per card, plus alternative outlines
[Rectify]  → 488×680 card (both 0° and 180° are scored)
  ↓
[Glare removal]  clipped highlights inpainted, glare mask kept for verification
  ↓
[Frame classifier]  → modern / fullart / special   (unsure → try both routes)
  ↓
[Retrieval over the WHOLE database]
  │   modern:   art crop   → art pHash/dHash  + art re-ranker embedding
  │   fallback: whole card → whole pHash/dHash + whole re-ranker embedding
  ↓
[Verification]  ORB + RANSAC restricted to near-aligned matches (weighted to the
  │             illustration) + glare-masked, illumination-normalised correlation
  ↓
[Tie-breakers]  set-symbol classifier, OCR (close calls on modern frames)
  ↓
Card + match quality (strong / good / weak / unknown)
```

Five trained models, each tiny:

| Component | Size | Purpose |
|-----------|------|---------|
| YOLO detector (fine-tuned) | 2.7M params | Propose card locations (refined onto the card edges afterwards) |
| Frame classifier | 98K params | Choose extraction strategy from rectified card |
| Art re-ranker | 3M params | Embedding for modern-frame art crops |
| Whole-card re-ranker | 3M params | Universal fallback embedding |
| Set symbol classifier | 500K params | Disambiguate reprints by set |

Detection, glare handling and verification are classical computer vision,
so they need no training and work with the artifacts you already have.

## File layout

```
project/
├── mtg_layout.py              # shared constants & helpers (DO NOT run)
├── card_detection.py          # card finding / perspective rectification (library)
├── card_matching.py           # glare handling + candidate verification (library)
├── photo_synthesis.py         # realistic synthetic photos for training (library)
├── train_detector.py          # train the YOLO card detector
├── train_identifier.py        # train everything else
├── Identify_card.py           # the identifier (run this to use it)
├── evaluate_real_photos.py    # accuracy on YOUR real photos vs ground truth
├── benchmark_run.py           # benchmark vs Claude on synthetic photos
├── tests/                     # pytest suite (no data needed)
└── mtg_data/
    ├── cards_metadata.json        # Scryfall metadata (input)
    ├── card_images/{id}.jpg       # card images (input)
    ├── yolo_card_best.pt          # trained detector (output)
    ├── frame_classifier.pth       # trained frame CNN (output)
    ├── art_reranker.pth           # trained art embedding model (output)
    ├── whole_reranker.pth         # trained whole-card embedding model (output)
    ├── set_classifier.pth         # trained set symbol CNN (output)
    ├── phash_db.npz               # hash database (output)
    ├── art_embeddings.npy         # precomputed art embeddings (output)
    ├── whole_embeddings.npy       # precomputed whole-card embeddings (output)
    ├── set_codes.json             # set classifier label mapping (output)
    └── frame_classes_cache.json   # per-card frame class cache (output)
```

## Setup

### Requirements

```
torch >= 2.0
torchvision
opencv-python
numpy >= 2.0  (for fast bit_count; older numpy works but slower)
Pillow
matplotlib
tqdm
ultralytics
easyocr        (optional, for OCR confirmation)
symspellpy     (optional, for fuzzy name matching)
anthropic      (only for benchmarking against Claude)
```

### Metadata

Your `mtg_data/cards_metadata.json` must contain Scryfall records with at
least these fields per card:

- `id` (string, the Scryfall UUID)
- `name` (string)
- `set` (string, set code)
- `layout` (string, e.g. "normal", "split", "saga")
- `full_art` (bool)
- `border_color` (string)

If your metadata is missing the layout-related fields, see "Re-downloading
metadata" at the bottom of this document.

### Card images

`mtg_data/card_images/` should contain one JPG per Scryfall ID. The
trainers will resize to 488×680 if the source isn't already that size, but
clean Scryfall `large` or `normal` images at native resolution work best.

## Training

Two phases. Run them in order.

### Phase 1: train the detector (~2-3 hours)

```bash
python train_detector.py
```

What this does:
1. Generates 8,000 synthetic training photos + 500 validation photos with
   `photo_synthesis.py`. Each photo has one to eight cards (scattered, in a
   grid, or fanned in a hand) seen through a tilted camera, so they appear
   in true perspective. Cards may be sleeved and overlapping. Scenes add
   glare (hot spots, light streaks, sheen, rainbow), shadows, fingers and
   dice, and phones, ID cards or paper as non-card negatives. Backgrounds
   are play mats (built from card art), wood, cloth, paper, or your own
   photos. White balance, blur, noise and JPEG artifacts are applied last.
   Scenes are saved to `mtg_data/yolo_card/` with one YOLO-OBB label per
   visible card.
2. Fine-tunes `yolo11n-obb.pt` on those scenes for 50 epochs. The
   pretrained checkpoint was trained on aerial imagery — fine-tuning
   teaches it what an MTG card looks like.
3. Copies the best weights to `mtg_data/yolo_card_best.pt`.

The biggest realism boost is to photograph the surfaces you actually use
(your table, your play mats, your desk) **without cards** and pass the folder
with `--backgrounds`. Ten to thirty photos are plenty.

Useful flags:

```bash
python train_detector.py --gen-only --show        # just inspect the dataset
python train_detector.py --train-only             # data already generated
python train_detector.py --n-train 15000          # bigger dataset
python train_detector.py --epochs 30              # shorter training
python train_detector.py --imgsz 800              # higher input resolution
python train_detector.py --backgrounds my_tables/ # your own backgrounds
python train_detector.py --base-weights yolo11s-obb.pt   # bigger model
```

What to watch:
- `mAP50(OBB)` should reach 0.85+ within 30 epochs. This is the metric
  that matters — bbox-only mAP can be high while rotation is wrong.
- `mtg_data/yolo_card_runs/cards/val_batch0_pred.jpg` is the visual
  sanity check. Open it after training and confirm boxes hug the cards
  at correct angles.

The detector is optional at inference: without `yolo_card_best.pt` the
classical detector runs on its own. With it, YOLO's proposals and the
classical candidates are combined, and each YOLO box is snapped onto the
real card edges. That also fixes perspective, which a rotated rectangle
can't represent.

### Phase 2: train the identifier components (~4-6 hours)

```bash
# First run the diagnosis to confirm metadata is healthy
python train_identifier.py --diagnose
```

Look for: 100% present for `layout`, `full_art`, `border_color`, `frame`.
Final classification should produce all three classes (e.g.
`{modern: 168, fullart: 28, special: 4}`). If everything classifies as
one class, see "Re-downloading metadata" below before proceeding.

```bash
python train_identifier.py
```

What this does:
1. Precomputes frame classes for every card (cached to disk).
2. Trains the frame classifier (~10 min).
3. Trains the art re-ranker on modern-frame cards (~1 hour).
4. Trains the whole-card re-ranker on all cards (~1.5 hours).
5. Trains the set symbol classifier on modern cards (~30 min).
6. Builds the hash database with both art and whole-card hashes.
7. Builds embedding databases for both re-rankers.
8. Runs an end-to-end sanity check (Top-1 should be ≥99%).

Synthetic training crops come from the same photo synthesiser as the
detector. They are rectified back with detector-like corner error, and glare
is usually inpainted. So the networks train on the same kind of input the
identifier produces at inference time. `--backgrounds DIR` works here too.

Useful flags:

```bash
python train_identifier.py --skip set frame   # skip specific components
python train_identifier.py --skip-train       # rebuild DB and embeddings only
python train_identifier.py --epochs 8         # shorter training
python train_identifier.py --rebuild-frame-cache   # discard cached labels
python train_identifier.py --backgrounds my_tables/ # your own backgrounds
```

If you re-download metadata or change classification logic, always pass
`--rebuild-frame-cache`. Otherwise the trainer reuses old labels.

## Using the identifier

### Interactive mode

```bash
python Identify_card.py --no-gui
```

Type a path, see the prediction, type another. Without `--no-gui` a file
picker opens. Running the script with no arguments at all processes the
`DEFAULT_BATCH_FOLDER` configured at the top of `Identify_card.py` (if it
exists).

### One-off mode

```bash
python Identify_card.py path/to/card.jpg
python Identify_card.py path/to/table.jpg --all    # every card in the photo
```

Prints the prediction and shows a figure: the photo with the detected
outline, the rectified card, the region used for matching, and the top-3
matches from the database.

### Batch mode

```bash
python Identify_card.py --batch my_photos/                # one card per photo
python Identify_card.py --batch my_photos/ --all          # all cards per photo
```

Saves a result figure per image to `my_photos_results/`, plus
`results.json` with the name, match quality and confidence for each image.

### Other flags

```bash
python Identify_card.py --once           # quit after one image
python Identify_card.py --no-ocr         # skip OCR even on close calls
python Identify_card.py --fast           # skip local-feature verification
python Identify_card.py --no-yolo        # classical detection only
python Identify_card.py --no-parallel    # trust the frame classifier blindly
```

From Python:

```python
from Identify_card import load_everything
cards, pipeline = load_everything()
best = pipeline.identify('photo.jpg')            # most confident card
every = pipeline.identify_all('table.jpg')       # list, one result per card
print(cards[best['top_k'][0][0]]['name'], best['quality'])
```

### Reading the output

```
PREDICTION: Lightning Bolt
  Set:           Magic 2010 (M10)
  Collector:     #146
  Type:          Instant
  Mana cost:     {R}
  Match:         strong
  Confidence:    0.781  (margin 0.293)
  Frame class:   modern (0.97) -> route via art crop
  Detection:     contour / outline primary
  Pipeline:      verified
  Glare:         6% of card inpainted
  Verification:  88 art / 241 total keypoint matches, correlation 0.31
```

- `Match` is the verdict to trust:
  - `strong` / `good`: verified by the local-feature match.
  - `weak`: plausible, but check it.
  - `unknown`: the card probably isn't in your database (or isn't a card).
    `--all` drops these.
- `Confidence` is the fused retrieval + verification score. `margin` is its
  lead over the best card with a *different name* (reprints of the same card
  don't count as competition).
- `Detection` shows the source: `yolo`, `contour` (classical), or
  `full_image` for pre-cropped scans. `outline` shows which variant won:
  - `primary`: the detected outline.
  - `alt*`: another outline for the same card, e.g. the card inside a slab.
  - `expand*`: the outline grown by a border width, for white-bordered or
    black-bordered cards whose outer edge blends into the background.
- `Pipeline`:
  - `fast_hash`: pHash was decisive, so verification was skipped.
  - `verified`: candidates were checked with local features.
  - `rerank`: `--fast`, no verification.

## Real photos: what makes them hard and what the pipeline does

| Problem | What happens now |
|---|---|
| Busy / dark / non-white background | Two detectors vote. The classical one uses several complementary segmentations: colour edges, adaptive and Otsu thresholds, and distance from the background colour. YOLO boxes are snapped onto the real edges. |
| Card at an angle | A true 4-corner outline is fitted with robust per-side line fits. Card shape is checked after undoing perspective, so a card shot at 50° still passes. |
| Card sideways / upside down | Corners are ordered so the warp is always portrait, and both 0° and 180° are scored. |
| Glare on sleeves / foils | Clipped highlights are inpainted before hashing and embedding. Glare pixels are ignored by the verification step. |
| Glare pushing the true card out of the hash top-500 | Embedding retrieval now scans the whole database, not just hash hits. |
| Fingers, dice, overlapping cards | Line fits and RANSAC verification ignore partial occlusion. |
| White border on a white table, black border on a black mat, card in a toploader / slab | Alternative and border-expanded outlines are tried, and the best match wins. |

Tips for your own photos: fill a good part of the frame with the card, and
tilt the card or phone slightly if a light is reflecting straight back.
Sleeves are fine.

## Evaluating on your own real photos

Synthetic benchmarks can't tell you how the pipeline does on your camera,
lighting and surfaces. Put some photos in a folder and either name each
file after the card it shows (`lightning_bolt.jpg`, `Counterspell 2.jpg`),
or add a `ground_truth.json`:

```json
{
  "IMG_2031.jpg": ["Lightning Bolt"],
  "table.jpg":    ["Island", "Island", "Counterspell"]
}
```

Then:

```bash
python evaluate_real_photos.py my_photos/ --pipeline full --save-vis my_photos_eval/
```

It prints recall (ground-truth cards found and correctly named), precision,
single-card accuracy and timing, and writes an overlay per photo.
`--reference-dir DIR` runs a training-free identifier built from a folder of
reference scans instead. That's useful for trying detection and matching
without trained models.

On the public example photos from
[tmikonen/magic_card_detector](https://github.com/tmikonen/magic_card_detector)
(9 photos, 61 cards: dark mats, patterned play mats, wood, steep
perspective, dice on cards, graded slabs, white-bordered cards on white),
matched against its 295 Alpha reference scans, the pipeline finds and names
98.4% of the cards at 98.4% precision. The one miss is a card not in the
reference set.

### Tests

```bash
python -m pytest tests
```

The tests use procedurally drawn cards, so no data or downloads are needed.

## Benchmarking

### Against Claude on synthetic photos

```bash
python benchmark_run.py
```

Generates synthetic photos stratified by difficulty (easy / medium / hard)
and frame class (by default 550 modern + 100 full-art + 50 special per
difficulty), runs both the local pipeline and Claude on them, and prints
accuracy and latency per bucket. Results go to
`mtg_data/benchmark_results_v2.json` and a plot to
`mtg_data/benchmark_v2_comparison.png`.

Claude API cost scales with the number of images (the script prints an
estimate before running); use `--no-claude` for a free run.

```bash
python benchmark_run.py --no-claude          # hash only
python benchmark_run.py --n-modern 100       # smaller run
python benchmark_run.py --reuse-images       # don't regenerate scenes
```

You'll need either `ANTHROPIC_API_KEY` in your environment or in
`mtg_data/.env`, or it'll prompt you.

### Diagnosing failures

Run the real-photo evaluation with overlays:

```bash
python evaluate_real_photos.py my_photos/ --pipeline full --save-vis my_photos_eval/
```

Each overlay shows which outline was used for every card and what it was
identified as. Look at it this way:
- **No outline, or the outline misses the card.** Detection is the
  problem. Try `--no-yolo`, or retrain the detector with `--backgrounds`
  photos of that surface.
- **Outline fits, name wrong, match `weak`/`unknown`.** Check the card is
  in `cards_metadata.json` and `card_images/`. Check that the embeddings
  are in sync: re-run `python train_identifier.py --skip-train`.
- **Outline fits, name wrong, match `strong`.** It's usually a reprint with
  identical art. The top-5 list will show the right card nearby.

## Re-downloading metadata

If `train_identifier.py --diagnose` shows missing layout fields, you need
to re-fetch metadata from Scryfall. Save this as `download_metadata.py`:

```python
import requests, json
from pathlib import Path

# Get the bulk-data manifest to find the current default-cards URL
manifest = requests.get('https://api.scryfall.com/bulk-data').json()
default_cards = next(b for b in manifest['data'] if b['type'] == 'default_cards')
print(f"Downloading {default_cards['size'] / 1e6:.0f} MB...")

resp = requests.get(default_cards['download_uri'])
all_cards = resp.json()
print(f"Got {len(all_cards):,} cards")

# Merge layout fields into existing metadata
with open('mtg_data/cards_metadata.json') as f:
    existing = json.load(f)

by_id = {c['id']: c for c in all_cards}
fields_to_add = ['layout', 'full_art', 'border_color', 'frame_effects',
                 'frame', 'type_line', 'mana_cost', 'set_name',
                 'collector_number']

for card in existing:
    if card['id'] in by_id:
        for field in fields_to_add:
            if field in by_id[card['id']]:
                card[field] = by_id[card['id']][field]

Path('mtg_data/cards_metadata.json').rename('mtg_data/cards_metadata_old.json')
with open('mtg_data/cards_metadata.json', 'w') as f:
    json.dump(existing, f)
print("Done.")
```

Run it once, then re-run `train_identifier.py --diagnose` to confirm
fields are populated, then proceed with training (passing
`--rebuild-frame-cache` to discard any stale frame labels).

## Updating the database when new cards release

When a new MTG set drops or you add more printings:

1. Update `mtg_data/cards_metadata.json` with the new cards
2. Add the new card images to `mtg_data/card_images/`
3. Run:
   ```bash
   python train_identifier.py --skip-train
   ```

This rebuilds the hash database and embeddings to cover the new cards
without retraining any models. Runs in 15-30 minutes depending on how
many cards were added.

You only need a full retrain (without `--skip-train`) when:
- You add 10%+ new cards relative to the existing dataset
- Cards are added in a stylistically novel category (Universes Beyond
  crossovers, experimental Secret Lair drops, new frame revisions)
- The sanity check Top-1 drops below 95% on a held-out sample of new cards

For typical incremental additions, the embedding space generalizes well
enough to skip training entirely.

## Troubleshooting

### `ModuleNotFoundError: No module named 'mtg_layout'`

`mtg_layout.py` must be in the same directory as the script you're running.
The trainers and identifier import from it.

### "0% accuracy" on the benchmark

Almost always the hash DB and embeddings are out of sync with
`cards_metadata.json` (the identifier refuses to start when their sizes
differ). Re-run `python train_identifier.py --skip-train`. Otherwise follow
"Diagnosing failures" above.

### Frame classifier achieves 100% accuracy on one class

This means metadata classification is degenerate (all cards labeled as
one class). Run `python train_identifier.py --diagnose` to see which
metadata fields are missing, then re-download metadata as described above.

### Detector finds objects inside the card art instead of the card itself

This happens with the pretrained `yolo11n-obb.pt`, which was trained on
aerial imagery. The identifier never uses it: without fine-tuned weights
it falls back to the classical detector. Run `train_detector.py` to get
a fine-tuned YOLO.

### A card isn't found at all

For single-card photos the identifier falls back to the whole image, so
cropping the photo roughly around the card always works. For hard scenes
(black-bordered cards on a black mat, heavy glare across the card edge),
retrain the detector with `--backgrounds` photos of that surface.

### EasyOCR or symspellpy import errors

Both are optional. The pipeline runs without them, just without OCR-based
disambiguation on close calls. Install with `pip install easyocr symspellpy`
if you want the extra signal.

## Performance notes

Typical numbers on a 5070 Ti with the full trained pipeline (before the
real-photo additions):

- Detection: ~20ms (YOLO inference)
- Hash retrieval: <1ms (vectorized over ~92k entries)
- Re-ranker: ~5ms on GPU, ~30ms on CPU
- OCR (when triggered): ~15ms
- Total: 30-50ms typical, 80ms with OCR

The real-photo additions cost extra time:
- Classical detection: ~0.3–0.8 s per photo on CPU. Use `--no-yolo` or
  YOLO-only to trade robustness for speed.
- Scanning the whole embedding DB for both orientations: a few ms.
- Local-feature verification of the top 10 candidates: ~0.2–0.5 s on CPU
  the first time a reference is seen (reference features are cached).

Expect roughly 1 s per card on CPU. `--fast` skips verification.

Compare to ~1.5-2.0 seconds per Claude API call. The accuracy gap
narrows as image difficulty increases (Claude wins by more on hard
photos), but for clean scans and decent phone photos the hash pipeline
matches Claude at roughly 30-50× the speed.

The hash database is ~3MB total for 92k cards. Embeddings add ~95MB
(50MB art + 45MB whole-card). The trained models together are ~25MB.
The whole runtime footprint fits comfortably on a phone.
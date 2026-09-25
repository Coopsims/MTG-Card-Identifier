"""
Benchmark the identification pipeline vs Claude on simulated photos.

  - Runs the real pipeline from Identify_card.py (YOLO + classical
    detection, glare removal, whole-database retrieval, local-feature
    verification), so the numbers reflect what you actually use.
  - Test photos come from photo_synthesis.py: camera-tilt perspective,
    sleeves, specular glare, shadows, occluders, play-mat / wood / cloth /
    paper backgrounds and camera degradation, at easy / medium / hard.
  - Stratifies test cards by frame class so we can measure modern,
    fullart, and special accuracy separately.
  - Reports per-frame-class accuracy in addition to the easy/medium/hard
    difficulty breakdown.

For accuracy on REAL photos use evaluate_real_photos.py instead.

Outputs:
  mtg_data/benchmark_samples_v2/        - synthetic test images
  mtg_data/benchmark_results_v2.json    - raw results
  mtg_data/benchmark_v2_comparison.png  - summary plots
"""

import argparse
import base64
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from mtg_layout import CARD_W, CARD_H, classify_frame_from_metadata
from photo_synthesis import BackgroundBank, synthesize_scene

DATA_DIR = Path('mtg_data')
IMAGE_DIR = DATA_DIR / 'card_images'
METADATA_PATH = DATA_DIR / 'cards_metadata.json'
FRAME_CACHE_PATH = DATA_DIR / 'frame_classes_cache.json'

BENCH_IMG_DIR = DATA_DIR / 'benchmark_samples_v2'
BENCH_ITEMS = DATA_DIR / 'benchmark_items_v2.json'
BENCH_RESULTS = DATA_DIR / 'benchmark_results_v2.json'
BENCH_PLOT = DATA_DIR / 'benchmark_v2_comparison.png'


# ---------------------------------------------------------------------------
# Synthetic photo generation (photo_synthesis.py, same difficulty buckets)

_BANK: Optional[BackgroundBank] = None


def _background_bank() -> BackgroundBank:
    global _BANK
    if _BANK is None:
        paths = list(IMAGE_DIR.glob('*.jpg'))

        def art_sampler():
            img = cv2.imread(str(random.choice(paths))) if paths else None
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None \
                else np.zeros((CARD_H, CARD_W, 3), np.uint8)
        _BANK = BackgroundBank(art_sampler=art_sampler)
    return _BANK


def synthesize_photo(card_img: np.ndarray, difficulty: str,
                     output_size: int = 1024) -> np.ndarray:
    """A realistic photo containing this one card."""
    rng = np.random.default_rng(random.getrandbits(32))
    scene, _ = synthesize_scene([card_img], (output_size, output_size), difficulty,
                                _background_bank(), rng, min_visible=0.0)
    return scene


# ---------------------------------------------------------------------------
# Claude

def identify_with_claude(image_path: str, api_key: str) -> Tuple[str, float]:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    with open(image_path, 'rb') as f:
        image_data = base64.standard_b64encode(f.read()).decode('utf-8')
    ext = Path(image_path).suffix.lower()
    media_type = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                  '.png': 'image/png', '.webp': 'image/webp'}.get(ext, 'image/jpeg')
    start = time.time()
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64",
                                              "media_type": media_type,
                                              "data": image_data}},
                {"type": "text",
                 "text": "What is the name of this Magic: The Gathering card? "
                         "Reply with ONLY the card name, nothing else."}
            ]
        }]
    )
    return msg.content[0].text.strip(), time.time() - start


# ---------------------------------------------------------------------------
# Test set construction
#
# Stratify by frame class so we get meaningful per-class numbers. Without
# this, ~85% of test cards would be modern and the fullart/special numbers
# would be too noisy to interpret.

def build_test_set(cards, available, frame_classes_by_id,
                   n_per_difficulty: int, n_per_frame_class: dict,
                   seed: int = 42) -> List[Dict]:
    """
    Generate test images stratified across difficulty AND frame class.
    n_per_frame_class is a dict like {'modern': 20, 'fullart': 10, 'special': 5}
    that controls the sampling within each difficulty.
    """
    BENCH_IMG_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # Index cards by frame class
    by_frame = defaultdict(list)
    for i in available:
        cls = frame_classes_by_id.get(cards[i]['id'], 'modern')
        by_frame[cls].append(i)
    print(f"Frame distribution in available cards: "
          f"{ {k: len(v) for k, v in by_frame.items()} }")

    items = []
    difficulties = ['easy', 'medium', 'hard']

    for difficulty in difficulties:
        for frame_cls, n_class in n_per_frame_class.items():
            pool = by_frame.get(frame_cls, [])
            if not pool:
                print(f"  Skipping {difficulty}/{frame_cls} - no cards")
                continue
            sampled = rng.sample(pool, min(n_class, len(pool)))
            for i, card_idx in enumerate(tqdm(sampled,
                                               desc=f"  {difficulty}/{frame_cls}")):
                card = cards[card_idx]
                src = IMAGE_DIR / f"{card['id']}.jpg"
                card_img = np.array(ImageOps.exif_transpose(Image.open(src)).convert('RGB'))
                if card_img.shape[0] != CARD_H or card_img.shape[1] != CARD_W:
                    card_img = cv2.resize(card_img, (CARD_W, CARD_H))
                scene = synthesize_photo(card_img, difficulty=difficulty)
                out_path = (BENCH_IMG_DIR /
                            f"{difficulty}_{frame_cls}_{i:04d}_{card['id']}.jpg")
                Image.fromarray(scene).save(out_path, quality=88)
                items.append({
                    'difficulty': difficulty,
                    'frame_class': frame_cls,
                    'card_idx': card_idx,
                    'card_id': card['id'],
                    'true_name': card['name'],
                    'true_set': card.get('set', '?'),
                    'image_path': str(out_path),
                })
    return items


# ---------------------------------------------------------------------------
# Loading

def load_pipeline():
    import Identify_card
    print("Loading pipeline components...")
    cards, pipeline = Identify_card.load_everything(verbose=True, use_ocr=False)
    return cards, pipeline


def load_frame_classes_by_id(cards, available):
    """Load the frame class cache. Falls back to metadata-only classification."""
    if FRAME_CACHE_PATH.exists():
        with open(FRAME_CACHE_PATH) as f:
            cache = json.load(f)
        avail_ids = {cards[i]['id'] for i in available}
        return {k: v for k, v in cache.items() if k in avail_ids}
    print("  No frame cache found - computing from metadata...")
    return {cards[i]['id']: classify_frame_from_metadata(cards[i])
            for i in available}


def get_api_key() -> Optional[str]:
    key = os.getenv('ANTHROPIC_API_KEY')
    if key:
        return key
    env_file = DATA_DIR / '.env'
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith('ANTHROPIC_API_KEY='):
                return line.split('=', 1)[1].strip()
    try:
        key = input("Anthropic API key (or Enter to skip): ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    return key or None


# ---------------------------------------------------------------------------
# Benchmark loop + reporting

def run_benchmark(cards, pipeline, items, api_key: Optional[str]):
    """Run both pipelines on every item. Returns nested dict of results."""
    # bucket by difficulty -> frame_class
    buckets = defaultdict(lambda: defaultdict(lambda: {
        'hash_top1': 0, 'hash_top5': 0, 'hash_times': [],
        'claude_top1': 0, 'claude_times': [],
        'route_counts': defaultdict(int),
        'frame_pred_counts': defaultdict(int),
        'frame_correct': 0,
        'samples': [], 'n': 0,
    }))

    for item in tqdm(items, desc="Benchmarking"):
        path = item['image_path']
        true_name = item['true_name']
        diff = item['difficulty']
        frame_cls = item['frame_class']
        bucket = buckets[diff][frame_cls]
        bucket['n'] += 1

        # Hash pipeline
        try:
            result = pipeline.identify(path, top_k=5, use_ocr=False)
            result['route'] = result.get('detection_route', '?')
            top = result['top_k']
            top_names = [cards[t[0]]['name'] for t in top]
            hash_top1 = top_names[0] == true_name
            hash_top5 = true_name in top_names
            bucket['hash_top1'] += int(hash_top1)
            bucket['hash_top5'] += int(hash_top5)
            bucket['hash_times'].append(result['timings']['total_ms'])
            bucket['route_counts'][result['route']] += 1
            bucket['frame_pred_counts'][result['predicted_frame']] += 1
            if result['predicted_frame'] == frame_cls:
                bucket['frame_correct'] += 1
            hash_pred = top_names[0]
            hash_score = top[0][1]
        except Exception as e:
            print(f"  hash error on {path}: {e}")
            hash_pred = f"ERROR: {e}"; hash_score = 0.0
            hash_top1 = hash_top5 = False
            result = {'predicted_frame': '?', 'route': 'error', 'timings': {}}

        # Claude
        if api_key:
            try:
                claude_pred, claude_t = identify_with_claude(path, api_key)
                claude_top1 = claude_pred.lower().strip() == true_name.lower().strip()
                bucket['claude_top1'] += int(claude_top1)
                bucket['claude_times'].append(claude_t * 1000)
            except Exception as e:
                print(f"  claude error: {e}")
                claude_pred = f"ERROR: {e}"; claude_top1 = False
        else:
            claude_pred = "skipped"; claude_top1 = False

        if len(bucket['samples']) < 2:
            bucket['samples'].append({
                'image_path': path,
                'true_name': true_name,
                'true_frame_class': frame_cls,
                'predicted_frame': result.get('predicted_frame', '?'),
                'route': result.get('route', '?'),
                'hash_pred': hash_pred,
                'hash_correct': bool(hash_top1),
                'claude_pred': claude_pred,
                'claude_correct': bool(claude_top1),
            })

    return buckets


def summarize(buckets, has_claude: bool):
    print()
    print("=" * 96)
    print("RESULTS")
    print("=" * 96)

    # Per-difficulty x frame class
    diff_order = ['easy', 'medium', 'hard']
    frame_order = ['modern', 'fullart', 'special']

    print(f"\n{'Difficulty':<10} {'Frame':<10} {'N':>4} "
          f"{'Hash T1':>8} {'Hash T5':>8} {'Claude T1':>10} "
          f"{'Frame OK':>9} {'Hash ms':>9} {'Claude ms':>10}")
    print("-" * 96)

    overall = {'n': 0, 'hash_top1': 0, 'hash_top5': 0,
               'claude_top1': 0, 'frame_correct': 0,
               'hash_times': [], 'claude_times': []}
    by_frame_overall = defaultdict(lambda: {
        'n': 0, 'hash_top1': 0, 'hash_top5': 0, 'claude_top1': 0,
    })

    for diff in diff_order:
        if diff not in buckets:
            continue
        for fc in frame_order:
            if fc not in buckets[diff]:
                continue
            r = buckets[diff][fc]
            n = r['n']
            if n == 0:
                continue
            h1 = r['hash_top1'] / n * 100
            h5 = r['hash_top5'] / n * 100
            c1 = r['claude_top1'] / n * 100 if has_claude else float('nan')
            fok = r['frame_correct'] / n * 100
            ht = np.mean(r['hash_times']) if r['hash_times'] else 0
            ct = np.mean(r['claude_times']) if has_claude and r['claude_times'] else 0

            c1_str = f"{c1:>8.1f}%" if has_claude else f"{'N/A':>9}"
            ct_str = f"{ct:>7.0f}ms" if has_claude else f"{'N/A':>9}"

            print(f"{diff:<10} {fc:<10} {n:>4} "
                  f"{h1:>6.1f}% {h5:>6.1f}% {c1_str} "
                  f"{fok:>7.1f}% {ht:>6.0f}ms {ct_str}")

            overall['n'] += n
            overall['hash_top1'] += r['hash_top1']
            overall['hash_top5'] += r['hash_top5']
            overall['claude_top1'] += r['claude_top1']
            overall['frame_correct'] += r['frame_correct']
            overall['hash_times'].extend(r['hash_times'])
            overall['claude_times'].extend(r['claude_times'])

            by_frame_overall[fc]['n'] += n
            by_frame_overall[fc]['hash_top1'] += r['hash_top1']
            by_frame_overall[fc]['hash_top5'] += r['hash_top5']
            by_frame_overall[fc]['claude_top1'] += r['claude_top1']

    print("-" * 96)
    n = overall['n']
    if n == 0:
        print("No results.")
        return {}
    h1 = overall['hash_top1'] / n * 100
    h5 = overall['hash_top5'] / n * 100
    c1 = overall['claude_top1'] / n * 100 if has_claude else float('nan')
    fok = overall['frame_correct'] / n * 100
    ht = np.mean(overall['hash_times']) if overall['hash_times'] else 0
    ct = np.mean(overall['claude_times']) if has_claude and overall['claude_times'] else 0
    c1_str = f"{c1:>8.1f}%" if has_claude else f"{'N/A':>9}"
    ct_str = f"{ct:>7.0f}ms" if has_claude else f"{'N/A':>9}"
    print(f"{'OVERALL':<10} {'all':<10} {n:>4} "
          f"{h1:>6.1f}% {h5:>6.1f}% {c1_str} "
          f"{fok:>7.1f}% {ht:>6.0f}ms {ct_str}")

    # Per-frame-class summary across difficulties
    print(f"\n{'By frame class:':<20}")
    print(f"  {'frame':<10} {'N':>5} {'Hash T1':>9} {'Hash T5':>9} "
          f"{'Claude T1':>11}")
    for fc in frame_order:
        r = by_frame_overall[fc]
        if r['n'] == 0:
            continue
        h1 = r['hash_top1'] / r['n'] * 100
        h5 = r['hash_top5'] / r['n'] * 100
        c1 = r['claude_top1'] / r['n'] * 100 if has_claude else float('nan')
        c1_str = f"{c1:>9.1f}%" if has_claude else f"{'N/A':>10}"
        print(f"  {fc:<10} {r['n']:>5} {h1:>7.1f}% {h5:>7.1f}% {c1_str}")

    # Speedup / accuracy gap
    if has_claude and ct > 0:
        speedup = ct / ht
        gap = h1 - c1
        print(f"\nSpeed:    hash pipeline is {speedup:.1f}x faster on average")
        if gap > 1:
            print(f"Accuracy: hash pipeline beats Claude by {gap:+.1f} points overall")
        elif gap < -1:
            print(f"Accuracy: Claude beats hash pipeline by {-gap:+.1f} points overall")
        else:
            print(f"Accuracy: within {abs(gap):.1f} points overall (effectively tied)")

    return {
        'overall': {'n': n, 'hash_top1': h1, 'hash_top5': h5,
                    'claude_top1': c1, 'frame_correct': fok,
                    'hash_ms_mean': ht, 'claude_ms_mean': ct},
        'by_frame_class': {
            fc: {
                'n': r['n'],
                'hash_top1': r['hash_top1'] / r['n'] * 100 if r['n'] else 0,
                'hash_top5': r['hash_top5'] / r['n'] * 100 if r['n'] else 0,
                'claude_top1': (r['claude_top1'] / r['n'] * 100
                                if r['n'] and has_claude else None),
            } for fc, r in by_frame_overall.items() if r['n'] > 0
        },
    }


def plot_results(buckets, has_claude: bool, out_path: Path):
    diff_order = [d for d in ['easy', 'medium', 'hard'] if d in buckets]
    frame_order = ['modern', 'fullart', 'special']

    if not diff_order:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: accuracy by difficulty (collapsed across frame class)
    ax = axes[0, 0]
    h1 = []; h5 = []; c1 = []
    for d in diff_order:
        n = sum(buckets[d][fc]['n'] for fc in buckets[d])
        h1.append(sum(buckets[d][fc]['hash_top1'] for fc in buckets[d]) / max(n, 1) * 100)
        h5.append(sum(buckets[d][fc]['hash_top5'] for fc in buckets[d]) / max(n, 1) * 100)
        if has_claude:
            c1.append(sum(buckets[d][fc]['claude_top1'] for fc in buckets[d]) / max(n, 1) * 100)
    x = np.arange(len(diff_order))
    w = 0.27 if has_claude else 0.4
    ax.bar(x - w, h1, w, label='Hash T1', color='#2ecc71', edgecolor='black')
    ax.bar(x,     h5, w, label='Hash T5', color='#27ae60', edgecolor='black')
    if has_claude:
        ax.bar(x + w, c1, w, label='Claude T1', color='#3498db', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels([d.capitalize() for d in diff_order])
    ax.set_ylabel('Accuracy (%)'); ax.set_title('Accuracy by difficulty')
    ax.set_ylim(0, 105); ax.legend(); ax.grid(axis='y', alpha=0.3)

    # Plot 2: accuracy by frame class
    ax = axes[0, 1]
    h1f = []; c1f = []; labels = []
    for fc in frame_order:
        n_total = sum(buckets[d][fc]['n'] for d in diff_order if fc in buckets[d])
        if n_total == 0:
            continue
        labels.append(fc)
        h1_sum = sum(buckets[d][fc]['hash_top1'] for d in diff_order if fc in buckets[d])
        h1f.append(h1_sum / n_total * 100)
        if has_claude:
            c1_sum = sum(buckets[d][fc]['claude_top1'] for d in diff_order if fc in buckets[d])
            c1f.append(c1_sum / n_total * 100)
    x = np.arange(len(labels))
    w = 0.4 if has_claude else 0.6
    ax.bar(x - w/2, h1f, w, label='Hash T1', color='#2ecc71', edgecolor='black')
    if has_claude:
        ax.bar(x + w/2, c1f, w, label='Claude T1', color='#3498db', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Accuracy (%)'); ax.set_title('Accuracy by frame class')
    ax.set_ylim(0, 105); ax.legend(); ax.grid(axis='y', alpha=0.3)

    # Plot 3: latency
    ax = axes[1, 0]
    ht_means = []; ct_means = []
    for d in diff_order:
        all_ht = [t for fc in buckets[d] for t in buckets[d][fc]['hash_times']]
        all_ct = [t for fc in buckets[d] for t in buckets[d][fc]['claude_times']]
        ht_means.append(np.mean(all_ht) if all_ht else 0)
        ct_means.append(np.mean(all_ct) if all_ct else 0)
    x = np.arange(len(diff_order))
    w = 0.4 if has_claude else 0.6
    ax.bar(x - w/2, ht_means, w, label='Hash', color='#2ecc71', edgecolor='black')
    if has_claude:
        ax.bar(x + w/2, ct_means, w, label='Claude', color='#3498db', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels([d.capitalize() for d in diff_order])
    ax.set_ylabel('Mean latency (ms)'); ax.set_title('Latency by difficulty')
    ax.set_yscale('log' if has_claude else 'linear')
    ax.legend(); ax.grid(axis='y', alpha=0.3)

    # Plot 4: frame classifier accuracy by difficulty
    ax = axes[1, 1]
    fok = []
    for d in diff_order:
        n_total = sum(buckets[d][fc]['n'] for fc in buckets[d])
        f_total = sum(buckets[d][fc]['frame_correct'] for fc in buckets[d])
        fok.append(f_total / max(n_total, 1) * 100)
    ax.bar(np.arange(len(diff_order)), fok, color='#9b59b6', edgecolor='black')
    ax.set_xticks(np.arange(len(diff_order)))
    ax.set_xticklabels([d.capitalize() for d in diff_order])
    ax.set_ylabel('Frame class accuracy (%)')
    ax.set_title('Frame classifier accuracy at inference')
    ax.set_ylim(0, 105); ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(fok):
        ax.text(i, v + 1, f"{v:.0f}", ha='center')

    plt.tight_layout()
    plt.savefig(out_path, dpi=80, bbox_inches='tight')
    plt.show()
    print(f"\nSaved plot to {out_path}")


def show_samples(buckets):
    print("\nSample predictions (per difficulty / frame class):")
    print("-" * 96)
    for d in ['easy', 'medium', 'hard']:
        if d not in buckets:
            continue
        for fc in ['modern', 'fullart', 'special']:
            if fc not in buckets[d] or not buckets[d][fc]['samples']:
                continue
            print(f"\n{d.upper()} / {fc.upper()}:")
            for s in buckets[d][fc]['samples'][:1]:
                print(f"  true:           {s['true_name']}  (frame={s['true_frame_class']})")
                print(f"  predicted frame: {s['predicted_frame']}, route: {s['route']}")
                mh = '+' if s['hash_correct'] else ' '
                mc = '+' if s['claude_correct'] else ' '
                print(f"  [{mh}] hash:       {s['hash_pred']}")
                print(f"  [{mc}] claude:     {s['claude_pred']}")


# ---------------------------------------------------------------------------
# Main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-modern', type=int, default=550,
                        help="Modern-frame cards per difficulty (default 550)")
    parser.add_argument('--n-fullart', type=int, default=100,
                        help="Full-art cards per difficulty (default 100)")
    parser.add_argument('--n-special', type=int, default=50,
                        help="Special-layout cards per difficulty (default 50)")
    parser.add_argument('--difficulties', nargs='+',
                        default=['easy', 'medium', 'hard'],
                        choices=['easy', 'medium', 'hard'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-claude', action='store_true')
    parser.add_argument('--reuse-images', action='store_true',
                        help="Reuse existing test images")
    args = parser.parse_args()

    print("=" * 96)
    print("BENCHMARK v2: Multi-region Pipeline vs Claude")
    print("=" * 96)

    cards, pipeline = load_pipeline()
    available = [i for i, c in enumerate(cards)
                 if (IMAGE_DIR / f"{c['id']}.jpg").exists()]
    frame_classes_by_id = load_frame_classes_by_id(cards, available)

    api_key = None if args.no_claude else get_api_key()
    has_claude = api_key is not None

    n_per_frame_class = {
        'modern': args.n_modern,
        'fullart': args.n_fullart,
        'special': args.n_special,
    }
    total_per_diff = sum(n_per_frame_class.values())
    total = total_per_diff * len(args.difficulties)

    if args.reuse_images and BENCH_ITEMS.exists():
        with open(BENCH_ITEMS) as f:
            items = json.load(f)
        items = [it for it in items if it['difficulty'] in args.difficulties]
        print(f"\nReusing {len(items)} existing benchmark images")
    else:
        print(f"\nGenerating {total} test images "
              f"({total_per_diff}/diff: "
              f"{args.n_modern} modern + {args.n_fullart} fullart + "
              f"{args.n_special} special)...")
        items = build_test_set(cards, available, frame_classes_by_id,
                                n_per_difficulty=total_per_diff,
                                n_per_frame_class=n_per_frame_class,
                                seed=args.seed)
        with open(BENCH_ITEMS, 'w') as f:
            json.dump(items, f, indent=2)

    if has_claude:
        cost = (1568 * 3e-6 + 10 * 1.5e-5) * len(items)
        print(f"Claude will be called {len(items)} times (~${cost:.3f})")
    else:
        print(f"\nRunning hash pipeline only (no API key).")

    buckets = run_benchmark(cards, pipeline, items, api_key)
    summary = summarize(buckets, has_claude)
    show_samples(buckets)

    # Persist
    serializable = {}
    for d, by_fc in buckets.items():
        serializable[d] = {}
        for fc, r in by_fc.items():
            serializable[d][fc] = {
                'n': r['n'],
                'hash_top1': r['hash_top1'],
                'hash_top5': r['hash_top5'],
                'claude_top1': r['claude_top1'],
                'frame_correct': r['frame_correct'],
                'hash_times': r['hash_times'],
                'claude_times': r['claude_times'],
                'route_counts': dict(r['route_counts']),
                'frame_pred_counts': dict(r['frame_pred_counts']),
                'samples': r['samples'],
            }
    with open(BENCH_RESULTS, 'w') as f:
        json.dump({
            'summary': summary,
            'per_difficulty': serializable,
            'config': vars(args),
            'has_claude': has_claude,
        }, f, indent=2)
    print(f"\nSaved raw results to {BENCH_RESULTS}")

    plot_results(buckets, has_claude, BENCH_PLOT)

    print("\n" + "=" * 96)
    print("Benchmark complete")
    print("=" * 96)


if __name__ == '__main__':
    main()
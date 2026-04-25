#!/usr/bin/env python3
"""
compute_clc.py
==============
Computes Cross-Lingual Consistency (CLC) scores across language runs
for every model and every evaluation script type.

Run this AFTER automation.sh has finished.

Usage
-----
    python compute_clc.py                          # scans all known output dirs
    python compute_clc.py --output_dirs dir1 dir2  # only scan specific dirs
    python compute_clc.py --output_dir 1_call_cm_placeholder_corr
    python compute_clc.py --verbose                # print per-pair details

What it does
------------
For every <output_dir>/<model_name>/ folder it finds:
  1. Collects all detailed.json files across language sub-folders.
  2. Groups samples by (relation, index) to match the same factual probe
     across languages.
  3. Computes pairwise Jaccard overlap of correctly-answered indices
     between every pair of languages.
  4. Computes per-language CLC and overall CLC.
  5. Writes results to:
       <output_dir>/<model_name>/clc_results.json   (detailed CLC breakdown)
       <output_dir>/<model_name>/clc_summary.txt    (human-readable summary)

CLC formula
-----------
For languages A and B:
    CLC(A, B) = |correct_A ∩ correct_B| / |correct_A ∪ correct_B|

Per-language CLC:
    CLC(A) = mean over all B ≠ A of CLC(A, B)

Overall CLC:
    mean of all per-language CLC values
"""

import os
import json
import glob
import argparse
from collections import defaultdict
from itertools import combinations


# ── All output directories the automation script can produce ──────────────────
ALL_OUTPUT_DIRS = [
    "filter_knowns_live_baseline",
    "filter_knowns_live_with_obj_baseline",
    "1_call_pure_implicit_cm",
    "1_call_pure_implicit_en",
    "1_call_cm_placeholder_corr",
    "1_call_en_placeholder_corr_final",
    "2_call_cm_placeholder_corr_8",
    "2_call_en_placeholder_corr_final",
    "2_call_transliteration",
]


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard overlap between two sets. Returns 0 if both empty."""
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def load_correct_indices(detailed_path: str) -> dict:
    """
    Load a detailed.json and return a dict:
        { relation: set(correct_indices) }

    Handles both flat list format (most scripts) and
    the nested {"results": [...]} format (baseline scripts).
    """
    with open(detailed_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Baseline scripts wrap results in {"results": [...], ...}
    if isinstance(raw, dict) and "results" in raw:
        records = raw["results"]
    elif isinstance(raw, list):
        records = raw
    else:
        return {}

    relation_correct = defaultdict(set)
    for record in records:
        if record.get("is_correct", False):
            relation = record.get("relation", "__all__")
            idx = record.get("index")
            if idx is not None:
                relation_correct[relation].add(idx)

    return dict(relation_correct)


def load_all_indices(detailed_path: str) -> dict:
    """
    Load a detailed.json and return a dict:
        { relation: set(all_indices) }
    """
    with open(detailed_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "results" in raw:
        records = raw["results"]
    elif isinstance(raw, list):
        records = raw
    else:
        return {}

    relation_all = defaultdict(set)
    for record in records:
        relation = record.get("relation", "__all__")
        idx = record.get("index")
        if idx is not None:
            relation_all[relation].add(idx)

    return dict(relation_all)


def collect_lang_data(model_dir: str) -> dict:
    """
    Walk <model_dir>/ and find all language subdirectory detailed.json files.

    Returns:
        {
            lang_code: {
                "correct": { relation: set(indices) },
                "all":     { relation: set(indices) },
                "path":    str
            }
        }
    """
    lang_data = {}

    for entry in sorted(os.listdir(model_dir)):
        lang_dir = os.path.join(model_dir, entry)
        if not os.path.isdir(lang_dir):
            continue

        # Skip non-language dirs
        if entry in ("clc_results.json", "clc_summary.txt", "LIVE.json"):
            continue

        detailed_path = os.path.join(lang_dir, "detailed.json")
        if not os.path.exists(detailed_path):
            continue

        lang_data[entry] = {
            "correct": load_correct_indices(detailed_path),
            "all":     load_all_indices(detailed_path),
            "path":    detailed_path,
        }

    return lang_data


def compute_clc_for_model(model_dir: str, verbose: bool = False) -> dict | None:
    """
    Compute full CLC breakdown for one model directory.

    Returns a results dict, or None if fewer than 2 languages found.
    """
    lang_data = collect_lang_data(model_dir)
    langs = sorted(lang_data.keys())

    if len(langs) < 2:
        return None

    # ── Collect all relations present across any language ──────────────────
    all_relations = set()
    for ld in lang_data.values():
        all_relations |= set(ld["correct"].keys())
        all_relations |= set(ld["all"].keys())
    all_relations = sorted(all_relations)

    # ── Per-language flat correct-index sets (across all relations) ────────
    lang_correct_flat = {}
    lang_total_flat   = {}
    for lang in langs:
        correct_union = set()
        total_union   = set()
        for rel in all_relations:
            # Use (relation, index) tuples so indices don't collide across rels
            correct_union |= {(rel, i) for i in lang_data[lang]["correct"].get(rel, set())}
            total_union   |= {(rel, i) for i in lang_data[lang]["all"].get(rel, set())}
        lang_correct_flat[lang] = correct_union
        lang_total_flat[lang]   = total_union

    # ── Per-relation correct-index sets ───────────────────────────────────
    relation_lang_correct = defaultdict(dict)  # relation → lang → set(indices)
    for rel in all_relations:
        for lang in langs:
            relation_lang_correct[rel][lang] = lang_data[lang]["correct"].get(rel, set())

    # ── Pairwise CLC (overall, flat) ──────────────────────────────────────
    pairwise_clc = {}
    for lang_a, lang_b in combinations(langs, 2):
        score = jaccard(lang_correct_flat[lang_a], lang_correct_flat[lang_b])
        pairwise_clc[f"{lang_a} ↔ {lang_b}"] = round(score, 6)

    # ── Per-language CLC (mean over all partners) ──────────────────────────
    per_lang_clc = {}
    for lang in langs:
        partners = [l for l in langs if l != lang]
        scores = [
            jaccard(lang_correct_flat[lang], lang_correct_flat[p])
            for p in partners
        ]
        per_lang_clc[lang] = round(sum(scores) / len(scores), 6) if scores else 0.0

    overall_clc = round(
        sum(per_lang_clc.values()) / len(per_lang_clc), 6
    ) if per_lang_clc else 0.0

    # ── Per-language accuracy ──────────────────────────────────────────────
    per_lang_acc = {}
    for lang in langs:
        correct = len(lang_correct_flat[lang])
        total   = len(lang_total_flat[lang])
        per_lang_acc[lang] = {
            "correct": correct,
            "total":   total,
            "accuracy": round(correct / total, 6) if total else 0.0,
        }

    overall_correct = sum(v["correct"] for v in per_lang_acc.values())
    overall_total   = sum(v["total"]   for v in per_lang_acc.values())
    overall_acc     = round(overall_correct / overall_total, 6) if overall_total else 0.0

    # ── Per-relation CLC ───────────────────────────────────────────────────
    per_relation_clc = {}
    for rel in all_relations:
        rel_pairwise = {}
        for lang_a, lang_b in combinations(langs, 2):
            s = jaccard(
                relation_lang_correct[rel][lang_a],
                relation_lang_correct[rel][lang_b],
            )
            rel_pairwise[f"{lang_a} ↔ {lang_b}"] = round(s, 6)

        rel_per_lang = {}
        for lang in langs:
            partners = [l for l in langs if l != lang]
            scores = [
                jaccard(relation_lang_correct[rel][lang], relation_lang_correct[rel][p])
                for p in partners
            ]
            rel_per_lang[lang] = round(sum(scores) / len(scores), 6) if scores else 0.0

        rel_overall = round(
            sum(rel_per_lang.values()) / len(rel_per_lang), 6
        ) if rel_per_lang else 0.0

        per_relation_clc[rel] = {
            "overall_clc":   rel_overall,
            "per_lang_clc":  rel_per_lang,
            "pairwise_clc":  rel_pairwise,
        }

    results = {
        "model_dir":        model_dir,
        "languages":        langs,
        "overall_accuracy": overall_acc,
        "overall_clc":      overall_clc,
        "per_language_accuracy": per_lang_acc,
        "per_language_clc":      per_lang_clc,
        "pairwise_clc":          pairwise_clc,
        "per_relation_clc":      per_relation_clc,
    }

    if verbose:
        print(f"\n  Languages : {langs}")
        print(f"  Overall acc : {overall_acc:.2%}")
        print(f"  Overall CLC : {overall_clc:.2%}")
        print(f"  Per-language CLC:")
        for lang, score in per_lang_clc.items():
            print(f"    {lang:12s}: {score:.2%}")
        print(f"  Pairwise CLC:")
        for pair, score in pairwise_clc.items():
            print(f"    {pair}: {score:.2%}")

    return results


def write_clc_results(model_dir: str, results: dict) -> None:
    """Write clc_results.json and clc_summary.txt into model_dir."""

    # ── JSON ──────────────────────────────────────────────────────────────
    json_path = os.path.join(model_dir, "clc_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # ── Human-readable TXT ────────────────────────────────────────────────
    txt_path = os.path.join(model_dir, "clc_summary.txt")
    lines = []
    lines.append("=" * 60)
    lines.append(f"CLC SUMMARY")
    lines.append(f"Model dir : {results['model_dir']}")
    lines.append(f"Languages : {', '.join(results['languages'])}")
    lines.append("=" * 60)
    lines.append(f"\nOverall Accuracy : {results['overall_accuracy']:.2%}")
    lines.append(f"Overall CLC      : {results['overall_clc']:.2%}")

    lines.append("\n── Per-Language Accuracy ──")
    for lang, stats in results["per_language_accuracy"].items():
        lines.append(
            f"  {lang:12s}: {stats['accuracy']:.2%}  "
            f"({stats['correct']}/{stats['total']})"
        )

    lines.append("\n── Per-Language CLC ──")
    for lang, score in results["per_language_clc"].items():
        lines.append(f"  {lang:12s}: {score:.2%}")

    lines.append("\n── Pairwise CLC ──")
    for pair, score in results["pairwise_clc"].items():
        lines.append(f"  {pair}: {score:.2%}")

    lines.append("\n── Per-Relation Overall CLC ──")
    for rel, data in sorted(results["per_relation_clc"].items(),
                             key=lambda x: -x[1]["overall_clc"]):
        lines.append(f"  {rel:35s}: {data['overall_clc']:.2%}")

    lines.append("\n" + "=" * 60)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  ✅  Saved → {json_path}")
    print(f"  ✅  Saved → {txt_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute CLC scores across language runs for all models."
    )
    parser.add_argument(
        "--output_dirs",
        nargs="+",
        default=None,
        help=(
            "One or more output directories to scan. "
            "Defaults to all known dirs: " + ", ".join(ALL_OUTPUT_DIRS)
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Shorthand for scanning a single output directory.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-pair CLC scores to stdout.",
    )
    args = parser.parse_args()

    # Resolve which dirs to scan
    if args.output_dir:
        scan_dirs = [args.output_dir]
    elif args.output_dirs:
        scan_dirs = args.output_dirs
    else:
        scan_dirs = ALL_OUTPUT_DIRS

    total_processed = 0
    total_skipped   = 0

    for out_dir in scan_dirs:
        if not os.path.isdir(out_dir):
            print(f"\n⚠️  Skipping (not found): {out_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"📂  Scanning: {out_dir}")
        print(f"{'='*60}")

        # Each subdirectory inside out_dir is one model
        model_entries = sorted(
            e for e in os.listdir(out_dir)
            if os.path.isdir(os.path.join(out_dir, e))
        )

        if not model_entries:
            print("  (no model subdirectories found)")
            continue

        for model_entry in model_entries:
            model_dir = os.path.join(out_dir, model_entry)
            print(f"\n  🤖  Model: {model_entry}")

            results = compute_clc_for_model(model_dir, verbose=args.verbose)

            if results is None:
                print(
                    "  ⚠️  Skipped — fewer than 2 language subdirectories "
                    "with detailed.json found."
                )
                total_skipped += 1
                continue

            write_clc_results(model_dir, results)
            total_processed += 1

    print(f"\n{'='*60}")
    print(f"✅  Done.  Processed: {total_processed}  |  Skipped: {total_skipped}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

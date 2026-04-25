#!/bin/bash
# ============================================================
#  run_clc_all.sh
#
#  Runs compute_clc.py across all 9 prompting-setting output
#  directories. Execute this after automation.sh has finished.
#
#  Usage:
#    bash run_clc_all.sh              # all dirs, quiet
#    bash run_clc_all.sh --verbose    # all dirs, print per-pair scores
#    bash run_clc_all.sh --dir 1_call_cm_placeholder_corr        # one dir only
#    bash run_clc_all.sh --dir 1_call_cm_placeholder_corr --verbose
#
#  Output (written inside each <dir>/<model>/ folder):
#    clc_results.json   — full JSON breakdown
#    clc_summary.txt    — human-readable summary
#
#  To skip a setting: comment out its line in OUTPUT_DIRS below.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPUTE_CLC="${SCRIPT_DIR}/compute_clc.py"

# ── Verify compute_clc.py exists ────────────────────────────
if [[ ! -f "$COMPUTE_CLC" ]]; then
    echo "❌  compute_clc.py not found at: ${COMPUTE_CLC}"
    echo "    Place compute_clc.py in the same directory as this script."
    exit 1
fi

# ── Parse arguments ─────────────────────────────────────────
VERBOSE=""
SINGLE_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --verbose)  VERBOSE="--verbose"; shift ;;
        --dir)      SINGLE_DIR="$2";     shift 2 ;;
        *)          echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── All 9 output directories (comment out to skip) ──────────
OUTPUT_DIRS=(
    "filter_knowns_live_baseline"            # Baseline — no candidates
    "filter_knowns_live_with_obj_baseline"   # Baseline — with candidates
    "1_call_pure_implicit_cm"                # Implicit code-mix (1 call)
    "1_call_pure_implicit_en"                # Implicit English (1 call)
    "1_call_cm_placeholder_corr"             # Explicit code-mix (1 call)
    "1_call_en_placeholder_corr_final"       # Explicit English (1 call)
    "2_call_cm_placeholder_corr_8"           # 2-stage code-mix
    "2_call_en_placeholder_corr_final"       # 2-stage English
    "2_call_transliteration"                 # 2-stage transliteration
)

# ── If --dir was passed, override to a single directory ─────
if [[ -n "$SINGLE_DIR" ]]; then
    OUTPUT_DIRS=("$SINGLE_DIR")
fi

# ── Counters ────────────────────────────────────────────────
FOUND=0
MISSING=0
FAILED=0

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║         CLC Computation — All Prompting Settings     ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

for DIR in "${OUTPUT_DIRS[@]}"; do
    FULL_PATH="${SCRIPT_DIR}/${DIR}"

    if [[ ! -d "$FULL_PATH" ]]; then
        echo "⚠️   Not found (skipping): ${DIR}"
        ((MISSING++)) || true
        continue
    fi

    echo "────────────────────────────────────────────────────"
    echo "📂  ${DIR}"
    echo "────────────────────────────────────────────────────"

    if python "${COMPUTE_CLC}" --output_dir "${FULL_PATH}" ${VERBOSE}; then
        ((FOUND++)) || true
    else
        echo "❌  compute_clc.py failed for: ${DIR}"
        ((FAILED++)) || true
    fi

    echo ""
done

# ── Final summary ────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Done."
printf  "║  Processed : %-3d  Missing : %-3d  Failed : %-3d      ║\n" \
        "$FOUND" "$MISSING" "$FAILED"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

if [[ $FAILED -gt 0 ]]; then
    exit 1
fi

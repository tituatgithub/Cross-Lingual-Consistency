#!/bin/bash
# ============================================================
#  run_all_evaluations.sh
#  Runs all 8 evaluation scripts for:
#    1. Bengali   (ben)
#    2. Assamese  (asm)
#    3. Odia      (ori)
#  For each language: iterates over all models × all scripts.
#  All runs on CUDA device 1.
# ============================================================

set -euo pipefail

# ──────────────────────────────────────────
#  Force all scripts onto GPU 1
# ──────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=1

# ──────────────────────────────────────────
#  MODELS — shared across all languages
# ──────────────────────────────────────────
MODELS=(
    "meta-llama/Llama-3.1-8B"
    "Qwen/Qwen2.5-7B"
    "google/gemma-7b"
)

SEED=12345
DATA_DIR="cm_klar"

# ──────────────────────────────────────────
#  HELPER: timestamped log file path
# ──────────────────────────────────────────
logfile() {
    local lang="$1" script="$2" model="$3"
    local safe_model="${model//\//_}"
    local log_dir="logs/${lang}"
    mkdir -p "$log_dir"
    echo "${log_dir}/${script}__${safe_model}__$(date +%Y%m%d_%H%M%S).log"
}

# ──────────────────────────────────────────
#  HELPER: run a script, tee output to log
# ──────────────────────────────────────────
run_script() {
    local label="$1"; shift
    local log="$1";   shift
    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  ▶  ${label}"
    echo "  📄  Log: ${log}"
    echo "  🖥️   GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "════════════════════════════════════════════════════"
    CUDA_VISIBLE_DEVICES=1 python "$@" 2>&1 | tee "$log"
    echo "  ✅  Done: ${label}"
}

# ──────────────────────────────────────────
#  CORE: run all 8 scripts for one language
#  Usage: run_language <lang_code> <source_lang> <source_script> <cm_target>
# ──────────────────────────────────────────
run_language() {
    local LANG_CODE="$1"
    local SOURCE_LANG="$2"
    local SOURCE_SCRIPT="$3"
    local TARGET_LANG_CM="$4"
    local TARGET_LANG_EN="English"

    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║  🌐  Starting language: ${SOURCE_LANG} (${LANG_CODE})"
    echo "╚══════════════════════════════════════════════════════╝"

    for MODEL in "${MODELS[@]}"; do

        echo ""
        echo "┌──────────────────────────────────────────────────────"
        echo "│  🤖  Model: ${MODEL}"
        echo "└──────────────────────────────────────────────────────"

        # ── 1. Baseline_filter_knowns ──────────────────────────
        run_script \
            "Baseline_filter_knowns | ${SOURCE_LANG} | ${MODEL}" \
            "$(logfile "$LANG_CODE" "Baseline_filter_knowns" "$MODEL")" \
            Evaluation-Scripts/Baseline_filter_knowns.py \
                --model_name    "$MODEL" \
                --seed          "$SEED" \
                --lang_codes    "$LANG_CODE"

        # ── 2. Baseline_filter_knowns_with_obj ────────────────
        run_script \
            "Baseline_filter_knowns_with_obj | ${SOURCE_LANG} | ${MODEL}" \
            "$(logfile "$LANG_CODE" "Baseline_filter_knowns_with_obj" "$MODEL")" \
            Evaluation-Scripts/Baseline_filter_knowns_with_obj.py \
                --model_name    "$MODEL" \
                --seed          "$SEED" \
                --lang_codes    "$LANG_CODE"

        # ── 3. Implicit_CM ─────────────────────────────────────
        run_script \
            "Implicit_CM | ${SOURCE_LANG} | ${MODEL}" \
            "$(logfile "$LANG_CODE" "Implicit_CM" "$MODEL")" \
            Evaluation-Scripts/Implicit_CM.py \
                --model_name    "$MODEL" \
                --seed          "$SEED" \
                --lang          "$LANG_CODE" \
                --mode          "hinglish"

        # ── 4. Implicit_EN ─────────────────────────────────────
        run_script \
            "Implicit_EN | ${SOURCE_LANG} | ${MODEL}" \
            "$(logfile "$LANG_CODE" "Implicit_EN" "$MODEL")" \
            Evaluation-Scripts/Implicit_EN.py \
                --model_name    "$MODEL" \
                --seed          "$SEED" \
                --lang          "$LANG_CODE" \
                --mode          "english"

        # ── 5. 1_Call_CM ───────────────────────────────────────
        run_script \
            "1_Call_CM | ${SOURCE_LANG} | ${MODEL}" \
            "$(logfile "$LANG_CODE" "1_Call_CM" "$MODEL")" \
            Evaluation-Scripts/1_Call_CM.py \
                --model_name    "$MODEL" \
                --seed          "$SEED" \
                --lang_code     "$LANG_CODE" \
                --data_dir      "$DATA_DIR" \
                --source_lang   "$SOURCE_LANG" \
                --source_script "$SOURCE_SCRIPT" \
                --target_lang   "$TARGET_LANG_CM"

        # ── 6. 1_Call_EN ───────────────────────────────────────
        run_script \
            "1_Call_EN | ${SOURCE_LANG} | ${MODEL}" \
            "$(logfile "$LANG_CODE" "1_Call_EN" "$MODEL")" \
            Evaluation-Scripts/1_Call_EN.py \
                --model_name    "$MODEL" \
                --seed          "$SEED" \
                --data_dir      "$DATA_DIR" \
                --lang_codes    "$LANG_CODE" \
                --source_lang   "$SOURCE_LANG" \
                --source_script "$SOURCE_SCRIPT" \
                --target_lang   "$TARGET_LANG_EN"

        # ── 7. 2_Call_CM ───────────────────────────────────────
        run_script \
            "2_Call_CM | ${SOURCE_LANG} | ${MODEL}" \
            "$(logfile "$LANG_CODE" "2_Call_CM" "$MODEL")" \
            Evaluation-Scripts/2_Call_CM.py \
                --model_name    "$MODEL" \
                --seed          "$SEED" \
                --lang_code     "$LANG_CODE" \
                --data_dir      "$DATA_DIR" \
                --source_lang   "$SOURCE_LANG" \
                --source_script "$SOURCE_SCRIPT" \
                --target_lang   "$TARGET_LANG_CM"

        # ── 8. 2_Call_EN ───────────────────────────────────────
        run_script \
            "2_Call_EN | ${SOURCE_LANG} | ${MODEL}" \
            "$(logfile "$LANG_CODE" "2_Call_EN" "$MODEL")" \
            Evaluation-Scripts/2_Call_EN.py \
                --model_name    "$MODEL" \
                --seed          "$SEED" \
                --data_dir      "$DATA_DIR" \
                --lang_codes    "$LANG_CODE" \
                --source_lang   "$SOURCE_LANG" \
                --source_script "$SOURCE_SCRIPT" \
                --target_lang   "$TARGET_LANG_EN"

    done

    echo ""
    echo "  🏁  Finished all models for: ${SOURCE_LANG} (${LANG_CODE})"
}

# ============================================================
#  RUN ALL LANGUAGES IN SEQUENCE
#  Args: lang_code  source_lang  source_script  cm_target_lang
# ============================================================

run_language  "ben"  "Bengali"   "Bengali"   "Banglish"
run_language  "asm"  "Assamese"  "Assamese"  "Assamglish"
run_language  "ori"  "Odia"      "Odia"      "Odiglish"

# ============================================================
#  DONE
# ============================================================
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  🎉  All evaluations complete!                       ║"
echo "║  Bengali  logs → logs/ben/                           ║"
echo "║  Assamese logs → logs/asm/                           ║"
echo "║  Odia     logs → logs/ori/                           ║"
echo "╚══════════════════════════════════════════════════════╝"

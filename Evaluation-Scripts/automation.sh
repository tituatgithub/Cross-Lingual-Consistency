#!/bin/bash
# ============================================================
#  automation_2.sh
#  Runs all evaluation scripts for multiple languages.
#
#  Scripts run per language:
#    1.  Baseline_filter_knowns           (filter_knowns_live.py)
#    2.  Baseline_filter_knowns_with_obj  (filter_knowns_live_obj.py)
#    3.  Implicit_CM                      (1_call_pure_implicit_cm.py)
#    4.  Implicit_EN                      (1_call_pure_implicit_en.py)
#    5.  1_Call_CM                        (1_call_cm_placeholder_2.py)
#    6.  1_Call_EN                        (1_call_en_placeholder.py)
#    7.  2_Call_CM                        (2_call_cm_placeholder_correct.py)
#    8.  2_Call_EN                        (2_call_en_placeholder.py)
#    9.  2_Call_Transliteration           (2_call_transliteration.py)
#
#  NOTE on lang_codes:
#    Baseline scripts  → use BOTH native + english dirs (e.g. ben,ben-en)
#                        EXCEPT for "en" → uses only "en" (no "en-en")
#    All other scripts → use native dir only            (e.g. ben)
#    English (en)      → CM / transliteration scripts are SKIPPED
#                        (no code-mixing or transliteration for English)
#
#  Usage:
#    bash automation_2.sh
#
#  To skip a language, comment out its run_language / run_language_en line
#  at the bottom of this file.
#  To skip a script inside the loop, comment out that run_script block.
#  To skip a model, comment out that model entry in MODELS array.
# ============================================================

set -euo pipefail

# ──────────────────────────────────────────
#  SCRIPT_DIR — absolute path to the folder
#  containing this script and all .py files.
#  No need to edit paths anywhere; just keep
#  all .py scripts in the same directory.
# ──────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ──────────────────────────────────────────
#  GPU — change to 0 if you want device 0
# ──────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=1

# ──────────────────────────────────────────
#  BATCH_SIZE — prompts per vLLM batch
#  (default matches original hardcoded values
#   in each script; override here globally)
# ──────────────────────────────────────────
BATCH_SIZE=64

# ──────────────────────────────────────────
#  GPU_MEM_UTIL — fraction of GPU memory
#  that vLLM may use (0.0 – 1.0)
# ──────────────────────────────────────────
GPU_MEM_UTIL=0.55

# ──────────────────────────────────────────
#  MODELS — comment/uncomment as needed
#  All active models run for every language.
# ──────────────────────────────────────────
MODELS=(
    # ── Small / tiny models ────────────────
    # "Qwen/Qwen2.5-0.5B-Instruct"
    # "Qwen/Qwen2.5-1.5B-Instruct"
    # "Qwen/Qwen2.5-3B"
    # "Qwen/Qwen2.5-3B-Instruct"
    # "meta-llama/Llama-3.2-1B"
    # "meta-llama/Llama-3.2-1B-Instruct"
    # "meta-llama/Llama-3.2-3B"
    # "meta-llama/Llama-3.2-3B-Instruct"
    # "google/gemma-3-270m"
    # "google/gemma-3-270m-it"
    # "google/gemma-3-1b-pt"   #only has hin
    # "google/gemma-3-1b-it"  #only has hin

    # ── Mid-size models ────────────────────
    # "meta-llama/Llama-3.1-8B"   # missing ben
    "meta-llama/Llama-3.1-8B-Instruct"
    # "Qwen/Qwen2.5-7B"
    "Qwen/Qwen2.5-7B-Instruct"
    # "google/gemma-7b"
    # "google/gemma-3-4b-it"  #only has hin
    # "google/gemma-3-4b-pt"  #only has hin
    # "Qwen/Qwen3-8B"  #only has hin
    "google/gemma-3-12b-it"  #only has hin
    # "google/gemma-3-12b-pt"  #everything missing

    # ── Large models ───────────────────────
    # "Qwen/Qwen2.5-14B"
    "Qwen/Qwen2.5-14B-Instruct"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

    # ── Qwen3 Instruct / Thinking ──────────
    # "Qwen/Qwen3-4B-Instruct-2507"  #only hindi
    # "Qwen/Qwen3-4B-Thinking-2507"  #only hindi


     # "Qwen/Qwen3-4B"
    # "Qwen/Qwen3-4B-Base"
     # "Qwen/Qwen3-1.7B"
     # "Qwen/Qwen3-0.6B"
     # "Qwen/Qwen3-8B"
    #  "Qwen 3.5 models"
     "Qwen/Qwen3.5-9B"
    #  "Qwen/Qwen3.5-9B-Base"
     # "Qwen/Qwen3.5-4B"
    #  "Qwen/Qwen3.5-4B-Base"
     # "Qwen/Qwen3.5-2B"
    #  "Qwen/Qwen3.5-2B-Base"
     # "Qwen/Qwen3.5-0.8B"
    #  "Qwen/Qwen3.5-0.8B-Base"

)

SEED=12345
DATA_DIR="cm_klar"

# ──────────────────────────────────────────
#  HELPER: produce a timestamped log path
#  Usage: logfile <lang_code> <script_tag> <model_name>
# ──────────────────────────────────────────
logfile() {
    local lang="$1" script="$2" model="$3"
    local safe_model="${model//\//_}"
    local log_dir="logs/${lang}"
    mkdir -p "$log_dir"
    echo "${log_dir}/${script}__${safe_model}__$(date +%Y%m%d_%H%M%S).log"
}

# ──────────────────────────────────────────
#  HELPER: print a banner, run a python script,
#          tee output to a log file.
#  Usage: run_script <label> <log_path> <python_script> [args...]
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
    # Swap the line below with the commented version to make failures fatal:
    # CUDA_VISIBLE_DEVICES=1 python "$@" 2>&1 | tee "$log"
    CUDA_VISIBLE_DEVICES=1 python "$@" 2>&1 | tee "$log" || true
    echo "  ✅  Done: ${label}"
}

# ============================================================
#  run_language — for non-English languages (ben, hin, asm …)
#
#  Runs ALL 9 scripts:
#    Baselines 1+2 use   lang_code,lang_code-en  (both dirs)
#    Scripts   3–9 use   lang_code               (native dir only)
#
#  Usage:
#    run_language <lang_code> <source_lang> <source_script> <cm_target_lang>
#
#  Example:
#    run_language "hin" "Hindi" "Hindi" "Hinglish"
#    run_language "hin-en" "Hindi" "Hindi" "Hinglish"   ← also valid
# ============================================================
run_language() {
    local LANG_CODE="$1"          # e.g. ben | hin | hin-en
    local SOURCE_LANG="$2"        # e.g. Bengali | Hindi
    local SOURCE_SCRIPT="$3"      # e.g. Bengali | Hindi  (script name)
    local TARGET_LANG_CM="$4"     # e.g. Banglish | Hinglish
    local TARGET_LANG_EN="English"

    # ── Baseline lang_codes: try native-en dir first; fall back to native only ──
    # e.g.  ben → "ben,ben-en" if cm_klar/ben-en/ exists, else just "ben"
    local EN_DIR="${DATA_DIR}/${LANG_CODE}-en"
    local LANG_CODE_WITH_EN
    if [ -d "${SCRIPT_DIR}/${EN_DIR}" ]; then
        LANG_CODE_WITH_EN="${LANG_CODE},${LANG_CODE}-en"
        echo "  ℹ️   Found ${EN_DIR} — baselines will use: ${LANG_CODE_WITH_EN}"
    else
        LANG_CODE_WITH_EN="${LANG_CODE}"
        echo "  ⚠️   ${EN_DIR} not found — baselines will use native only: ${LANG_CODE_WITH_EN}"
    fi

    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║  🌐  Starting language: ${SOURCE_LANG} (${LANG_CODE})"
    echo "╚══════════════════════════════════════════════════════╝"

    for MODEL in "${MODELS[@]}"; do

        echo ""
        echo "┌──────────────────────────────────────────────────────"
        echo "│  🤖  Model: ${MODEL}"
        echo "└──────────────────────────────────────────────────────"

        # ── Script 1: Baseline_filter_knowns  (native + en dirs) ──────────
        run_script \
            "Baseline_filter_knowns | ${SOURCE_LANG} | ${MODEL}" \
            "$(logfile "$LANG_CODE" "Baseline_filter_knowns" "$MODEL")" \
            ${SCRIPT_DIR}/filter_knowns_live.py \
                --model_name              "$MODEL" \
                --seed                    "$SEED" \
                --lang_codes              "$LANG_CODE_WITH_EN" \
                --batch_size              "$BATCH_SIZE" \
                --gpu_memory_utilization  "$GPU_MEM_UTIL"

        # ── Script 2: Baseline_filter_knowns_with_obj  (native + en) ──────
        run_script \
            "Baseline_filter_knowns_with_obj | ${SOURCE_LANG} | ${MODEL}" \
            "$(logfile "$LANG_CODE" "Baseline_filter_knowns_with_obj" "$MODEL")" \
            ${SCRIPT_DIR}/filter_knowns_live_obj.py \
                --model_name              "$MODEL" \
                --seed                    "$SEED" \
                --lang_codes              "$LANG_CODE_WITH_EN" \
                --batch_size              "$BATCH_SIZE" \
                --gpu_memory_utilization  "$GPU_MEM_UTIL"

        # ── Script 3: Implicit_CM  (native only) ──────────────────────────
        run_script \
            "Implicit_CM | ${SOURCE_LANG} | ${MODEL}" \
            "$(logfile "$LANG_CODE" "Implicit_CM" "$MODEL")" \
            ${SCRIPT_DIR}/1_call_pure_implicit_cm.py \
                --model_name              "$MODEL" \
                --seed                    "$SEED" \
                --lang_code               "$LANG_CODE" \
                --data_dir                "$DATA_DIR" \
                --source_lang             "$SOURCE_LANG" \
                --source_script           "$SOURCE_SCRIPT" \
                --target_lang             "$TARGET_LANG_CM" \
                --batch_size              "$BATCH_SIZE" \
                --gpu_memory_utilization  "$GPU_MEM_UTIL"

        # ── Script 4: Implicit_EN  (native only) ──────────────────────────
        run_script \
            "Implicit_EN | ${SOURCE_LANG} | ${MODEL}" \
            "$(logfile "$LANG_CODE" "Implicit_EN" "$MODEL")" \
            ${SCRIPT_DIR}/1_call_pure_implicit_en.py \
                --model_name              "$MODEL" \
                --seed                    "$SEED" \
                --lang_code               "$LANG_CODE" \
                --data_dir                "$DATA_DIR" \
                --source_lang             "$SOURCE_LANG" \
                --source_script           "$SOURCE_SCRIPT" \
                --target_lang             "$TARGET_LANG_EN" \
                --batch_size              "$BATCH_SIZE" \
                --gpu_memory_utilization  "$GPU_MEM_UTIL"

        # ── Script 5: 1_Call_CM  (native only) ────────────────────────────
        run_script \
            "1_Call_CM | ${SOURCE_LANG} | ${MODEL}" \
            "$(logfile "$LANG_CODE" "1_Call_CM" "$MODEL")" \
            ${SCRIPT_DIR}/1_call_cm_placeholder_2.py \
                --model_name              "$MODEL" \
                --seed                    "$SEED" \
                --lang_code               "$LANG_CODE" \
                --data_dir                "$DATA_DIR" \
                --source_lang             "$SOURCE_LANG" \
                --source_script           "$SOURCE_SCRIPT" \
                --target_lang             "$TARGET_LANG_CM" \
                --batch_size              "$BATCH_SIZE" \
                --gpu_memory_utilization  "$GPU_MEM_UTIL"

        # ── Script 6: 1_Call_EN  (native only) ────────────────────────────
        run_script \
            "1_Call_EN | ${SOURCE_LANG} | ${MODEL}" \
            "$(logfile "$LANG_CODE" "1_Call_EN" "$MODEL")" \
            ${SCRIPT_DIR}/1_call_en_placeholder.py \
                --model_name              "$MODEL" \
                --seed                    "$SEED" \
                --data_dir                "$DATA_DIR" \
                --lang_codes              "$LANG_CODE" \
                --source_lang             "$SOURCE_LANG" \
                --source_script           "$SOURCE_SCRIPT" \
                --target_lang             "$TARGET_LANG_EN" \
                --batch_size              "$BATCH_SIZE" \
                --gpu_memory_utilization  "$GPU_MEM_UTIL"

        # ── Script 7: 2_Call_CM  (native only) ────────────────────────────
        run_script \
            "2_Call_CM | ${SOURCE_LANG} | ${MODEL}" \
            "$(logfile "$LANG_CODE" "2_Call_CM" "$MODEL")" \
            ${SCRIPT_DIR}/2_call_cm_placeholder_correct.py \
                --model_name              "$MODEL" \
                --seed                    "$SEED" \
                --lang_code               "$LANG_CODE" \
                --data_dir                "$DATA_DIR" \
                --source_lang             "$SOURCE_LANG" \
                --source_script           "$SOURCE_SCRIPT" \
                --target_lang             "$TARGET_LANG_CM" \
                --batch_size              "$BATCH_SIZE" \
                --gpu_memory_utilization  "$GPU_MEM_UTIL"

        # ── Script 8: 2_Call_EN  (native only) ────────────────────────────
        run_script \
            "2_Call_EN | ${SOURCE_LANG} | ${MODEL}" \
            "$(logfile "$LANG_CODE" "2_Call_EN" "$MODEL")" \
            ${SCRIPT_DIR}/2_call_en_placeholder.py \
                --model_name              "$MODEL" \
                --seed                    "$SEED" \
                --data_dir                "$DATA_DIR" \
                --lang_codes              "$LANG_CODE" \
                --source_lang             "$SOURCE_LANG" \
                --source_script           "$SOURCE_SCRIPT" \
                --target_lang             "$TARGET_LANG_EN" \
                --batch_size              "$BATCH_SIZE" \
                --gpu_memory_utilization  "$GPU_MEM_UTIL"

        # ── Script 9: 2_Call_Transliteration  (native only) ───────────────
        run_script \
            "2_Call_Transliteration | ${SOURCE_LANG} | ${MODEL}" \
            "$(logfile "$LANG_CODE" "2_Call_Transliteration" "$MODEL")" \
            ${SCRIPT_DIR}/2_call_transliteration.py \
                --model_name              "$MODEL" \
                --seed                    "$SEED" \
                --lang_code               "$LANG_CODE" \
                --data_dir                "$DATA_DIR" \
                --source_lang             "$SOURCE_LANG" \
                --source_script           "$SOURCE_SCRIPT" \
                --target_lang             "Transliterated" \
                --batch_size              "$BATCH_SIZE" \
                --gpu_memory_utilization  "$GPU_MEM_UTIL"

        # ── Cleanup: delete model from HuggingFace cache after all scripts ─
        # e.g.  Qwen/Qwen2.5-7B  →  models--Qwen--Qwen2.5-7B
        CACHE_DIR="$HOME/.cache/huggingface/hub"
        SAFE_MODEL="${MODEL//\//--}"
        MODEL_PATH="${CACHE_DIR}/models--${SAFE_MODEL}"

        echo "🧹 Deleting cached model: ${MODEL_PATH}"
        rm -rf "$MODEL_PATH" || true
        echo "✅ Deleted: ${MODEL}"

    done

    echo ""
    echo "  🏁  Finished all models for: ${SOURCE_LANG} (${LANG_CODE})"
}

# ============================================================
#  run_language_en -- special handler for English ("en")
#
#  English does NOT have a paired "-en" data directory, so the
#  baseline scripts receive only "--lang_codes en" (not "en,en-en").
#
#  Only the two baselines are run here -- all other scripts
#  (Implicit_CM/EN, 1_Call_CM/EN, 2_Call_CM/EN, Transliteration)
#  are designed for native-language input (Hindi, Bengali, etc.)
#  and are NOT applicable when the source is already English.
#
#  Scripts run:
#    1. Baseline_filter_knowns           (lang_codes = "en")
#    2. Baseline_filter_knowns_with_obj  (lang_codes = "en")
#
#  Usage:
#    run_language_en   (no arguments -- always uses "en")
# ============================================================
run_language_en() {
    local LANG_CODE="en"
    local SOURCE_LANG="English"

    echo ""
    echo "============================================================"
    echo "  Starting language: English (en) [baselines only]"
    echo "============================================================"

    for MODEL in "${MODELS[@]}"; do

        echo ""
        echo "------------------------------------------------------"
        echo "  Model: ${MODEL}"
        echo "------------------------------------------------------"

        # -- Script 1: Baseline_filter_knowns  (en only -- no en-en dir) --
        run_script \
            "Baseline_filter_knowns | English | ${MODEL}" \
            "$(logfile "$LANG_CODE" "Baseline_filter_knowns" "$MODEL")" \
            ${SCRIPT_DIR}/filter_knowns_live.py \
                --model_name              "$MODEL" \
                --seed                    "$SEED" \
                --lang_codes              "$LANG_CODE" \
                --batch_size              "$BATCH_SIZE" \
                --gpu_memory_utilization  "$GPU_MEM_UTIL"

        # -- Script 2: Baseline_filter_knowns_with_obj  (en only) ---------
        run_script \
            "Baseline_filter_knowns_with_obj | English | ${MODEL}" \
            "$(logfile "$LANG_CODE" "Baseline_filter_knowns_with_obj" "$MODEL")" \
            ${SCRIPT_DIR}/filter_knowns_live_obj.py \
                --model_name              "$MODEL" \
                --seed                    "$SEED" \
                --lang_codes              "$LANG_CODE" \
                --batch_size              "$BATCH_SIZE" \
                --gpu_memory_utilization  "$GPU_MEM_UTIL"

        # -- [SKIP] Scripts 3-9 -- not applicable for English --------------
        # All remaining scripts (Implicit_CM/EN, 1_Call_CM/EN, 2_Call_CM/EN,
        # Transliteration) convert native-language data into code-mixed or
        # English answers. English as source has no such step -- skip all.

        # -- Cleanup: delete model from HuggingFace cache ------------------
        # e.g.  Qwen/Qwen2.5-7B  ->  models--Qwen--Qwen2.5-7B
        CACHE_DIR="$HOME/.cache/huggingface/hub"
        SAFE_MODEL="${MODEL//\//--}"
        MODEL_PATH="${CACHE_DIR}/models--${SAFE_MODEL}"

        echo "Deleting cached model: ${MODEL_PATH}"
        rm -rf "$MODEL_PATH" || true
        echo "Deleted: ${MODEL}"

    done

    echo ""
    echo "  Finished all models for: English (en)"
}

# ============================================================
#  RUN ALL LANGUAGES IN SEQUENCE
#
#  ┌──────────────────────────────────────────────────────┐
#  │  To skip a language  → comment out its line below    │
#  │  To skip a script    → comment it out inside the     │
#  │                         run_language() function above │
#  │  To skip a model     → comment it out in MODELS[]    │
#  └──────────────────────────────────────────────────────┘
#
#  Format for non-English:
#    run_language  "<lang_code>"  "<Source_Lang>"  "<Source_Script>"  "<CM_target>"
#
#  English must use run_language_en (no args).
# ============================================================

# ── English (special handler — no CM / transliteration) ───────────────────────
run_language_en

# ── Indic languages ───────────────────────────────────────────────────────────
run_language  "asm"    "Assamese"   "Assamese"   "Assamglish"       
run_language  "ben"    "Bengali"    "Bengali"    "Banglish"         
run_language  "doi"    "Dogri"      "Dogri"      "Dogrilish"        
run_language  "guj"    "Gujarati"   "Gujarati"   "Gujlish"          
run_language  "hin"    "Hindi"      "Hindi"      "Hinglish"         
run_language  "kan"    "Kannada"    "Kannada"    "Kanglish"         
run_language  "kon"    "Konkani"    "Konkani"    "Konklish"
run_language  "mai"    "Maithili"   "Maithili"   "Maithilish"
run_language  "mal"    "Malayalam"  "Malayalam"  "Malyalamglish"
run_language  "mar"    "Marathi"    "Marathi"    "Marglish"
run_language  "nep"    "Nepali"     "Nepali"     "Nepglish"
run_language  "ori"    "Odia"       "Odia"       "Odiglish"         
run_language  "pan"    "Punjabi"    "Punjabi"    "Punglish"
run_language  "san"    "Sanskrit"   "Sanskrit"   "Sanglish"
run_language  "snd"    "Sindhi"     "Sindhi"     "Sindlish"
run_language  "tam"    "Tamil"      "Tamil"      "Tamlish"
run_language  "tel"    "Telugu"     "Telugu"     "Teluglish"
run_language  "urd"    "Urdu"       "Urdu"       "Urglish"


# ============================================================
#  DONE
# ============================================================
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  🎉  All evaluations complete!                       ║"
echo "║  Logs → logs/<lang_code>/                            ║"
echo "╚══════════════════════════════════════════════════════╝"

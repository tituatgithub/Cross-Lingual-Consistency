# Cross-Lingual Consistency Evaluation

A research evaluation framework that tests how well large language models answer factual questions written in Indic languages — with variants that use **code-mixing**, **transliteration**, and **English translation** as intermediate reasoning steps.

---

## Table of Contents

1. [Repo Placement](#1-repo-placement)
2. [Data Placement](#2-data-placement)
3. [What is GPU Utilization?](#3-what-is-gpu-utilization)
4. [Getting Started](#4-getting-started)
   - [Create a Virtual Environment](#41-create-a-virtual-environment)
   - [Install Dependencies](#42-install-dependencies)
   - [Run the Automation Script](#43-run-the-automation-script)
5. [What Each File Does](#5-what-each-file-does)
6. [How Results Are Saved](#6-how-results-are-saved)
7. [Adding Models](#7-adding-models)
8. [Using Hugging Face Model Names](#8-using-hugging-face-model-names)
9. [Language & Script Reference](#9-language--script-reference)
10. [Quick Troubleshooting](#10-quick-troubleshooting)

---

## 1. Repo Placement

Clone this repository **anywhere on your machine** — there is no hard-coded absolute path anywhere in the codebase. All paths inside the scripts are relative.

```bash
git clone https://github.com/tituatgithub/cross-lingual-consistency.git
cd cross-lingual-consistency/Evaluation-Scripts
```

> **Important:** Every command you run — including the bash script and all Python scripts — must be executed from inside the `Evaluation-Scripts/` directory. The scripts glob for data using relative paths like `cm_klar/*/*.json`, so your working directory matters.

The expected layout after setup:

```
cross-lingual-consistency/
└── Evaluation-Scripts/
    ├── cm_klar/                        ← your data goes here (see Section 2)
    │   ├── hin/
    │   │   ├── capital.json
    │   │   ├── place_of_birth.json
    │   │   └── ...
    │   ├── hin-en/
    │   ├── ben/
    │   ├── ben-en/
    │   └── ...
    ├── logs/                           ← auto-created when you run scripts
    ├── 1_call_cm_placeholder.py
    ├── 1_call_en_placeholder.py
    ├── 1_call_pure_implicit_cm.py
    ├── 1_call_pure_implicit_en.py
    ├── 2_call_cm_placeholder_correct.py
    ├── 2_call_en_placeholder.py
    ├── 2_call_transliteration.py
    ├── filter_knowns_live.py
    ├── filter_knowns_live_obj.py
    └── automation.sh
```

---

## 2. Data Placement

All data must live inside a folder called **`cm_klar/`** placed directly inside `Evaluation-Scripts/`. The scripts glob for JSON files at the pattern `cm_klar/*/*.json`, where the intermediate folder name is the **language code**.

### Expected folder structure inside `cm_klar/`

```
cm_klar/
├── hin/                    ← Hindi (native script)
│   ├── capital.json
│   ├── place_of_birth.json
│   └── ... (one file per relation)
├── hin-en/                 ← Hindi questions translated to English
│   ├── capital.json
│   └── ...
├── ben/                    ← Bengali (native script)
├── ben-en/                 ← Bengali → English
├── asm/                    ← Assamese
├── asm-en/
├── ori/                    ← Odia
├── ori-en/
├── guj/                    ← Gujarati
├── guj-en/
├── tel/                    ← Telugu
├── mal/                    ← Malayalam
└── en/                     ← English baseline
```

### JSON file format (each relation file)

Each `.json` file must follow this schema:

```json
{
  "prompt_templates": [
    "<subject> का जन्म स्थान <mask> है।"
  ],
  "samples": [
    {
      "index": 0,
      "subject": "महात्मा गांधी",
      "object": "पोरबंदर",
      "object_candidates": ["पोरबंदर", "दिल्ली", "मुंबई", "चेन्नई"]
    }
  ]
}
```

- `prompt_templates[0]` — the question template; `<subject>` and `<mask>` are placeholders.
- `samples` — list of individual factual probes.
- `object_candidates` — optional list of answer choices shown to the model; if absent, the scripts fall back to open-ended prefix matching.

### Supported relations

The scripts filter for exactly these 20 relation names (must match the filename, without `.json`):

```
applies_to_jurisdiction, capital, capital_of, continent,
country_of_citizenship, developer, field_of_work, headquarters_location,
instrument, language_of_work_or_name, languages_spoken, location_of_formation,
manufacturer, native_language, occupation, official_language,
owned_by, place_of_birth, place_of_death, religion
```

---

## 3. What is GPU Utilization?

Every script loads the model via **vLLM** and passes a `gpu_memory_utilization` parameter, for example:

```python
llm = LLM(
    model=model_name,
    trust_remote_code=True,
    gpu_memory_utilization=0.20,
    max_model_len=4096,
    max_num_seqs=16
)
```

`gpu_memory_utilization=0.20` means vLLM is allowed to use **20% of your GPU's VRAM** for the KV cache (the memory used to store intermediate attention states during generation). The model weights themselves are loaded on top of this.

| Setting | Meaning | When to use |
|---|---|---|
| `0.20` | 20% for KV cache | Multiple experiments running in parallel; small models |
| `0.45` | 45% for KV cache | Single experiment; medium models (default in `2_call_transliteration.py`) |
| `0.90` | 90% for KV cache | Single experiment; large models, maximum throughput |

**If you run out of VRAM**, lower `gpu_memory_utilization` or reduce `max_num_seqs`. If you want faster throughput on a free GPU, increase it toward `0.9`.

The GPU used is controlled by the environment variable at the top of `automation.sh`:

```bash
export CUDA_VISIBLE_DEVICES=1   # change to 0 for the first GPU
```

---

## 4. Getting Started

### 4.1 Create a Virtual Environment

```bash
# From inside Evaluation-Scripts/
python3 -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows
```

### 4.2 Install Dependencies

```bash
pip install --upgrade pip

pip install \
    vllm \
    torch \
    transformers \
    datasets \
    tqdm \
    numpy
```

> **Note on vLLM:** vLLM requires a CUDA-capable GPU and a matching CUDA toolkit. Install the version of `torch` that matches your CUDA version before installing `vllm`. See [vllm.ai](https://docs.vllm.ai/en/latest/getting_started/installation.html) for details.

For Hugging Face gated models (e.g. Llama, Gemma), log in once:

```bash
pip install huggingface_hub
huggingface-cli login
```

Paste your Hugging Face access token when prompted.

### 4.3 Run the Automation Script

`automation.sh` is the **single entry point** that runs all 9 evaluation scripts for every language–model combination you configure.

```bash
# Make it executable (first time only)
chmod +x automation.sh

# Run it
bash automation.sh
```

All output — including stdout and stderr for every experiment — is automatically saved to `logs/<lang_code>/`.

**To run a single script manually** (useful for debugging):

```bash
python 1_call_cm_placeholder.py \
    --model_name "google/gemma-3-270m" \
    --lang_code "hin" \
    --data_dir "cm_klar" \
    --source_lang "Hindi" \
    --source_script "Hindi" \
    --target_lang "Hinglish"
```

**To watch live progress** while a run is in progress, open a second terminal:

```bash
tail -f <output_dir>/<model_name>/LIVE.json
```

---

## 5. What Each File Does

### `automation.sh` — Master Orchestrator

The shell script that ties everything together. It defines:
- Which **models** to run (the `MODELS` array).
- Which **languages** to run (the `run_language` / `run_language_en` calls at the bottom).
- Which **GPU** to use (`CUDA_VISIBLE_DEVICES`).

It calls all 9 evaluation scripts in sequence for each model–language pair, logs output to `logs/`, and deletes the model from the HuggingFace cache after each model finishes to free disk space.

---

### Baseline Scripts

#### `filter_knowns_live.py` — Baseline (No Candidates)

Plain few-shot prompting in the native language. No candidate list is shown to the model. The model must generate the answer freely; correctness is measured by prefix matching against the ground truth.

- **Input:** Native-language questions (`hin`, `ben`, etc.) **and** their English translations (`hin-en`, `ben-en`, etc.) — both passed via `--lang_codes`.
- **Output dir:** `filter_knowns_live_baseline/<model>/`

#### `filter_knowns_live_obj.py` — Baseline (With Candidates)

Same as above, but the candidate answer list (`object_candidates`) is appended to each prompt. The model selects from the list; correctness uses candidate-aware matching.

- **Input:** Same as above — both native and `-en` directories.
- **Output dir:** `filter_knowns_live_with_obj_baseline/<model>/`

---

### Single-Call Scripts (`1_call_*`)

These scripts make **one LLM call per question** and ask the model to output a structured JSON with two fields: `"translation"` (the intermediate reasoning step) and `"answer"` (the final answer in the source script). Guided JSON decoding via vLLM's `StructuredOutputsParams` ensures the output is always valid JSON.

#### `1_call_cm_placeholder.py` — Explicit Code-Mix + Answer (1 Call)

Prompts the model to first produce a code-mixed (e.g. Hinglish) version of the native-language question, then answer in the native script — all in one JSON output.

- **Arguments:** `--source_lang`, `--source_script`, `--target_lang` (the code-mixed language, e.g. `Hinglish`)
- **Output dir:** `1_call_cm_placeholder_corr/<model>/<lang>/`

#### `1_call_en_placeholder.py` — Explicit English Translation + Answer (1 Call)

Same structure, but the intermediate step is a full English translation of the question.

- **Arguments:** `--source_lang`, `--source_script`, `--target_lang English`
- **Output dir:** `1_call_en_placeholder_corr_final/<model>/<lang>/`

#### `1_call_pure_implicit_cm.py` — Implicit Code-Mix (1 Call)

The model is instructed to **internally** perform a code-mixed transformation of the question (without outputting it), then output only the answer. The JSON schema has only one field: `"answer"`.

- **Output dir:** `1_call_pure_implicit_cm/<model>/<lang>/`

#### `1_call_pure_implicit_en.py` — Implicit English Translation (1 Call)

Same as above but the internal step is English translation. The model reasons in English internally but outputs the answer in the native script.

- **Output dir:** `1_call_pure_implicit_en/<model>/<lang>/`

---

### Two-Call Scripts (`2_call_*`)

These scripts make **two separate LLM calls per question**: Stage 1 produces the intermediate form; Stage 2 receives that intermediate form and produces the final answer. This allows studying each stage independently and gives cleaner signal on where errors occur.

#### `2_call_cm_placeholder_correct.py` — 2-Stage Code-Mix Pipeline

- **Stage 1:** Native script → Code-mixed (e.g. Hinglish) in Roman script.
- **Stage 2:** Code-mixed question + candidates → Answer in native script.
- **Output dir:** `2_call_cm_placeholder_corr_8/<model>/<lang>/`

#### `2_call_en_placeholder.py` — 2-Stage English Translation Pipeline

- **Stage 1:** Native script → English translation.
- **Stage 2:** English question + candidates → Answer in native script.
- **Output dir:** `2_call_en_placeholder_corr_final/<model>/<lang>/`

#### `2_call_transliteration.py` — 2-Stage Transliteration Pipeline

- **Stage 1:** Native script → Phonetic romanization (no translation, no code-mixing; every word is kept but rewritten in Roman letters).
- **Stage 2:** Romanized question + candidates → Answer in native script.
- **Output dir:** `2_call_transliteration/<model>/<lang>/`

---

## 6. How Results Are Saved

Every script creates a structured output directory automatically. The layout is:

```
<OUTPUT_DIR>/
└── <model_name_with_slashes_replaced_by_underscores>/
    └── <lang_code>/
        ├── summary.json    ← overall + per-language accuracy and CLC scores
        ├── detailed.json   ← one entry per sample with full prompt, prediction, and correctness
        └── LIVE.json       ← updated in real time during the run; safe to tail -f
```

### `summary.json`

```json
{
  "overall_acc": 0.632,
  "overall_clc": 0.581,
  "per_language_acc": {
    "hin": 0.71,
    "ben": 0.55
  },
  "per_language_clc": {
    "hin": 0.60,
    "ben": 0.56
  }
}
```

- **`overall_acc`** — fraction of questions answered correctly across all languages.
- **`per_language_acc`** — accuracy broken down by language code.
- **`overall_clc`** — Cross-Lingual Consistency score: how often the model gets the same questions right across languages (Jaccard overlap of correct indices).
- **`per_language_clc`** — CLC per language (average overlap with all other languages).

### `detailed.json`

A JSON array where each entry contains:

| Field | Description |
|---|---|
| `index` | Sample index from the source file |
| `relation` | Relation name (e.g. `place_of_birth`) |
| `subject` | The subject entity |
| `question_<lang>` | The question as shown to the model |
| `<target_lang>_question` | The intermediate translation / code-mix (where applicable) |
| `model_prediction` | Raw parsed answer from the model |
| `matched_candidate` | Which candidate was matched (if candidates were used) |
| `object_candidates` | Full candidate list |
| `ground_truth` | Correct answer |
| `is_correct` | Boolean correctness |
| `raw_output` | Exact text the model generated |
| `final_prompt` | The complete prompt sent to the model |

For two-call scripts, `stage1_prompt`, `stage1_raw_output`, `stage2_prompt`, and `stage2_raw_output` are also included.

### `LIVE.json`

Written after every batch during the run. Contains running accuracy and the last 50 results. Use it to monitor progress:

```bash
watch -n 2 "python -c \"import json; d=json.load(open('LIVE.json')); print(d['progress'], d['running_accuracy'])\""
# or simply:
tail -f LIVE.json
```

### Log files

`automation.sh` saves a timestamped `.log` file for every script execution under:

```
logs/<lang_code>/<script_tag>__<model_name>__<timestamp>.log
```

---

## 7. Adding Models

### In `automation.sh`

Open `automation.sh` and add your model to the `MODELS` array:

```bash
MODELS=(
    "google/gemma-3-270m"       # already there
    "meta-llama/Llama-3.1-8B"  # add new models here
    "Qwen/Qwen2.5-7B"
)
```

Comment out any model you want to skip. The script will iterate over every active model for every active language.

### Adding a new language

At the bottom of `automation.sh`, call `run_language` with your new language's parameters:

```bash
run_language  "<lang_code>"  "<Source_Lang>"  "<Source_Script>"  "<CM_target_lang>"

# Examples:
run_language  "hin"   "Hindi"      "Hindi"      "Hinglish"
run_language  "ben"   "Bengali"    "Bengali"    "Banglish"
run_language  "tel"   "Telugu"     "Telugu"     "Teluglish"
run_language  "mal"   "Malayalam"  "Malayalam"  "Malyalamglish"
```

For English (which has no code-mixed variant), use the special handler:

```bash
run_language_en
```

---

## 8. Using Hugging Face Model Names

**Yes, you can use direct Hugging Face Hub model IDs** — vLLM accepts them natively and will download the model automatically on first run.

```bash
# These all work as-is:
"meta-llama/Llama-3.1-8B"
"meta-llama/Llama-3.1-8B-Instruct"
"Qwen/Qwen2.5-7B"
"google/gemma-3-270m"
"google/gemma-7b"
"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
```

Models are cached by HuggingFace in `~/.cache/huggingface/hub/`. After each model finishes, `automation.sh` automatically deletes the cache entry to free disk space:

```bash
CACHE_DIR="$HOME/.cache/huggingface/hub"
SAFE_MODEL="${MODEL//\//--}"
rm -rf "${CACHE_DIR}/models--${SAFE_MODEL}"
```

Comment out those three lines inside `automation.sh` if you want to keep the model cached between runs.

### Gated models (Llama, Gemma)

Some models require you to accept a license on Hugging Face and provide an access token:

```bash
huggingface-cli login
# Paste your token from https://huggingface.co/settings/tokens
```

Your token is stored in `~/.cache/huggingface/token` and is picked up automatically by vLLM.

### Local model paths

You can also point to a local directory instead of a Hub ID:

```bash
MODELS=(
    "/data/models/my-finetuned-llama"
)
```

vLLM will load from the local path directly — no internet required.

---

## 9. Language & Script Reference

| Lang code | Language | Script name (for `--source_script`) | CM target (for `--target_lang`) |
|---|---|---|---|
| `hin` | Hindi | Hindi | Hinglish |
| `ben` | Bengali | Bengali | Banglish |
| `asm` | Assamese | Assamese | Assamglish |
| `ori` | Odia | Odia | Odiglish |
| `guj` | Gujarati | Gujarati | Gujlish |
| `tel` | Telugu | Telugu | Teluglish |
| `mal` | Malayalam | Malayalam | Malyalamglish |
| `mai` | Maithili | Maithili | Maithilish |
| `nep` | Nepali | Nepali | Nepglish |
| `en` | English | — | — (use `run_language_en`) |

The `-en` suffix variants (e.g. `hin-en`, `ben-en`) are subdirectories in `cm_klar/` containing the English-translated versions of the questions in that language. These are used only by the two baseline scripts.

---

## 10. Quick Troubleshooting

**`cuda out of memory`**
Lower `gpu_memory_utilization` in the relevant `.py` script (e.g. from `0.45` to `0.20`), or reduce `max_num_seqs`.

**`No samples loaded` / empty dataset**
Check that your `cm_klar/` directory is inside `Evaluation-Scripts/` and that the subfolder names match the `--lang_code` values exactly (e.g. `hin`, not `hindi`).

**`ValueError: No languages found for model`**
Occurs in `1_call_en_placeholder.py` and `2_call_en_placeholder.py` when neither `--lang_codes` nor `--lang_file` is provided. Always pass `--lang_codes` when calling these scripts manually.

**Model downloads slowly or fails**
Run `huggingface-cli login` and ensure your token has access to the model. For Llama and Gemma models, you must first accept the license on the model's Hugging Face page.

**Guided JSON decoding errors**
Ensure you are using a version of vLLM that supports `StructuredOutputsParams`. Install the latest stable release: `pip install --upgrade vllm`.

**Script exits with `set -euo pipefail` error**
Each `run_script` call inside `automation.sh` ends with `|| true` to prevent one failure from stopping the entire run. If you want failures to be fatal, remove the `|| true` suffix on the `tee` line inside `run_script`.

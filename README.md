# Evaluation Scripts — Code-Mixed & Cross-Lingual Knowledge Probing

This repository contains a suite of Python evaluation scripts for probing factual knowledge in Large Language Models (LLMs) across **code-mixed** and **multilingual** settings. The experiments cover Indic languages (Hindi, Bengali, Gujarati, Malayalam, Odia, Assamese, etc.) and test how well LLMs answer questions when the input is written in a script other than English.

---

## Background & Motivation

When an LLM is asked a factual question in Hindi (Devanagari script), Bengali (Bengali script), Gujarati (Gujarati script), Malayalam (Malayalam script), or Odia (Odia script), does it actually *understand* the question, or does it fail? This codebase investigates that question by:

1. Presenting the model with questions in a **source language** (e.g., Hindi in Devanagari script).
2. Optionally prompting the model to translate the question into a **target language** (e.g., Hinglish — Hindi grammar with English vocabulary written in Roman script — or plain English) before answering.
3. Measuring accuracy and **cross-lingual consistency (CLC)** — whether the model gets the same questions right across different languages.

The scripts vary along two axes:
- **Translation strategy**: Implicit (model translates internally), 1-Call (single API call for translation + answer), or 2-Call (separate API calls for translation and answer).
- **Target language**: Code-mixed / Hinglish (`CM`) or English (`EN`).

---

## Dataset Structure

All scripts load data from a directory of JSON files organized as:

```
<data_dir>/
  <language_code>/
    <relation>.json
    ...
```

For example:
```
cm_klar/
  hin/
    capital.json
    place_of_birth.json
    ...
  ben/
    capital.json
    ...
```

Each JSON file contains:
```json
{
  "prompt_templates": ["<subject> ka capital kya hai?"],
  "samples": [
    {
      "index": 0,
      "subject": "India",
      "object": "New Delhi",
      "object_candidates": ["New Delhi", "Mumbai", "Kolkata", "Chennai"]
    }
  ]
}
```

- **`prompt_templates`**: A fill-in-the-blank question template. `<subject>` is replaced with the entity name, and `<mask>` (if present) is removed.
- **`samples`**: Individual knowledge probes.
- **`object`**: The ground-truth answer.
- **`object_candidates`**: A list of candidate answers (used for multiple-choice-style evaluation). When present, the model must pick one.

### Relations Evaluated

All scripts evaluate over the same 20 Wikidata relations:

`applies_to_jurisdiction`, `capital`, `capital_of`, `continent`, `country_of_citizenship`, `developer`, `field_of_work`, `headquarters_location`, `instrument`, `language_of_work_or_name`, `languages_spoken`, `location_of_formation`, `manufacturer`, `native_language`, `occupation`, `official_language`, `owned_by`, `place_of_birth`, `place_of_death`, `religion`

---

## Few-Shot Prompting Strategy

All scripts use **3-shot prompting** (configurable). The 3 demonstration examples are always sampled from the **same language and relation** as the test question — meaning the few-shot examples are in the **same source language script** as the question being asked.

For example, if the test question is a Hindi question about a person's `place_of_birth`, the 3 demonstrations will also be Hindi `place_of_birth` questions (with their correct answers). This ensures the model sees in-distribution examples without any cross-lingual leakage in the demonstrations.

The demonstrations show:
```
Q: <subject> ka place of birth kya hai?
Candidates: Porbandar, Mumbai, Delhi, Pune
Output: {"answer": "Porbandar"}
```

The test question is then appended and the model generates the `Output:` field.

---

## Evaluation Metrics

### Accuracy
Matching is done with a candidate-aware strategy (in priority order):
1. **Exact match** (case-insensitive)
2. **Candidate is prefix of prediction** (e.g., prediction = "Microsoft Corp", candidate = "Microsoft")
3. **Prediction is prefix of candidate** (e.g., prediction = "Micro", candidate = "Microsoft")
4. **Candidate appears as substring in prediction**

If no candidate matches, a fallback prefix match against the ground-truth string is used.

### Cross-Lingual Consistency (CLC)
For each language, CLC measures how much overlap there is in the *set of correctly answered question indices* between that language and every other language — averaged using the Jaccard similarity (intersection over union). This captures whether the model's knowledge is consistent across languages, not just accurate in isolation.

---

## Output Files

Each script writes its results to a structured output directory:

```
<OUTPUT_DIR>/<model_safe_name>/<lang_code>/
  summary.json     ← overall accuracy, per-language accuracy, CLC scores
  detailed.json    ← full record for every example
  LIVE.json        ← updates in real-time during a run (tail -f to monitor)
```

You can watch a run in progress with:
```bash
tail -f <OUTPUT_DIR>/<model>/<lang>/LIVE.json
```

---

## Script Reference

---

### `Baseline_filter_knowns.py` — Open-Generation Baseline (No Candidates)

**What it does:**
The simplest baseline. The model is given a few-shot prompt in the source language and must generate the answer **freely** (no candidate list is shown). Accuracy is measured using prefix matching against the ground truth.

**When to use:** To measure raw factual recall without any answer-selection scaffolding.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model_name` | str | `Qwen/Qwen2.5-7B` | HuggingFace model name or path |
| `--seed` | int | `12345` | Random seed for reproducibility |
| `--lang_codes` | str | *(required)* | Comma-separated language codes to evaluate, e.g. `hin,ben,guj` |

**Example:**
```bash
python Baseline_filter_knowns.py \
  --model_name meta-llama/Llama-3.1-8B \
  --lang_codes hin,ben \
  --seed 42
```

**Output directory:** `Baseline_filter_knowns/`

---

### `Baseline_filter_knowns_with_obj.py` — Baseline with Candidate List

**What it does:**
Same as `Baseline_filter_knowns.py` but the model is shown the candidate answers alongside the question. The model still generates freely, but matching uses the candidate-aware strategy (exact, prefix, substring) to find the best-matching candidate. This typically gives higher accuracy than the open-generation baseline.

**When to use:** To measure factual recall when candidates are provided as a hint, but without any explicit translation step.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model_name` | str | `meta-llama/Llama-3.1-8B` | HuggingFace model name or path |
| `--seed` | int | `12345` | Random seed |
| `--lang_codes` | str | *(required)* | Comma-separated language codes, e.g. `asm,ben,guj` |

**Example:**
```bash
python Baseline_filter_knowns_with_obj.py \
  --model_name Qwen/Qwen2.5-7B \
  --lang_codes hin,ben,guj,mal,ori \
  --seed 12345
```

**Output directory:** `Baseline_with_opt/`

---

### `Implicit_CM.py` — Implicit Code-Mixed Translation (Single Call)

**What it does:**
The model is instructed (via the system preamble) to **mentally** translate the source-language question into a code-mixed form (e.g., Hinglish: Hindi grammar + English vocabulary in Roman script) internally, then directly output only the answer in JSON. The translation is **never written** in the output — it happens implicitly inside the model's reasoning.

The prompt structure is:
```
[System preamble: "Mentally translate to Hinglish, then answer as JSON"]
Q: <Hindi question>
Candidates: <list>
Output: {"answer": "..."}   ← model generates this
```

**Key idea:** Tests whether telling the model to use code-mixed as an internal reasoning step (without requiring it to write out the translation) improves answer accuracy over a plain baseline.

**Language config:** Reads from `configs/lang/<lang_code>.json` to determine source language, script, and target language.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model_name` | str | `meta-llama/Llama-3.1-8B` | HuggingFace model name or path |
| `--seed` | int | `12345` | Random seed |
| `--mode` | str | `hinglish` | Translation target mode: `hinglish`, `english`, or `native` |
| `--lang` | str | `hin` | Single language code to evaluate, e.g. `hin`, `ben`, `guj` |

**Example:**
```bash
python Implicit_CM.py \
  --model_name google/gemma-7b \
  --lang hin \
  --mode hinglish
```

**Output directory:** `Implicit_CM/`

---

### `Implicit_EN.py` — Implicit English Translation (Single Call)

**What it does:**
Identical in structure to `Implicit_CM.py`, but the model is instructed to mentally translate the source-language question into **English** (not code-mixed Hinglish). It then outputs only the answer JSON, without writing the English translation.

**Key idea:** Tests whether implicit English as the reasoning pivot language helps the model answer source-language questions.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model_name` | str | `Qwen/Qwen2.5-7B` | HuggingFace model name or path |
| `--seed` | int | `12345` | Random seed |
| `--mode` | str | `english` | Translation target mode: `english`, `hinglish`, or `native` |
| `--lang` | str | `hin` | Comma-separated language codes, e.g. `hin` or `hin,ben` |

**Example:**
```bash
python Implicit_EN.py \
  --model_name meta-llama/Llama-3.1-8B \
  --lang ben \
  --mode english
```

**Output directory:** `Implicit_EN/`

---

### `1_Call_CM.py` — Explicit Code-Mixed Translation + Answer (Single API Call)

**What it does:**
The model performs translation and answering **explicitly** in a single API call, outputting both fields in one JSON response:

```json
{
  "translation": "<Hinglish version of the question>",
  "answer": "<answer in source script>"
}
```

The `translation` field captures the code-mixed (Hinglish) form of the source-language question in Roman script. The `answer` field contains the factual answer chosen from the candidates, written back in the **source script** (e.g., Devanagari for Hindi).

**Key idea:** By forcing the model to write out the translation explicitly before answering, it may use the code-mixed form as a reasoning step that improves answer accuracy. This is the "chain-of-thought via code-mixing" approach.

**Guided decoding:** Uses `StructuredOutputsParams` (via vLLM) to constrain model output to valid JSON with exactly `translation` and `answer` fields.

**Optional dictionary hint:** If `DICTIONARY_PATH` is set (pointing to a JSON file of source-language → English word mappings), those mappings are included in the prompt to assist with code-mixed generation.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model_name` | str | `google/gemma-7b` | HuggingFace model name or path |
| `--seed` | int | `12345` | Random seed |
| `--lang_code` | str | `hin` | Comma-separated language codes, e.g. `hin` or `ben` |
| `--data_dir` | str | `cm_klar` | Root data directory containing language subdirectories |
| `--output_dir` | str | `filter_knowns_implicit-trans` | *(overridden by hardcoded `OUTPUT_DIR = "1_Call_CM"`)* |
| `--source_lang` | str | *(required)* | Full source language name, e.g. `Hindi`, `Bengali` |
| `--source_script` | str | *(required)* | Script name, e.g. `Devanagari`, `Bengali`, `Odia` |
| `--target_lang` | str | *(required)* | Code-mixed target language name, e.g. `Hinglish`, `Benglish` |

**Internal constants (edit in script):**

| Constant | Description |
|---|---|
| `OUTPUT_DIR` | Top-level output directory (default: `"1_Call_CM"`) |
| `DATA_DIR` | Data root directory (default: `"cm_klar"`) |
| `DICTIONARY_PATH` | Path to a JSON lexicon for dictionary hints, or `None` |

**Example:**
```bash
python 1_Call_CM.py \
  --model_name meta-llama/Llama-3.1-8B \
  --lang_code hin \
  --source_lang Hindi \
  --source_script Devanagari \
  --target_lang Hinglish
```

**Output directory:** `1_Call_CM/`

---

### `1_Call_EN.py` — Explicit English Translation + Answer (Single API Call)

**What it does:**
Same concept as `1_Call_CM.py`, but the model translates into **plain English** instead of a code-mixed language. The single JSON output is:

```json
{
  "translation": "<English translation of the question>",
  "answer": "<answer in source script>"
}
```

This directly tests whether English as an intermediate pivot language (made explicit in the output) helps the model answer source-language factual questions.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model_name` | str | `meta-llama/Llama-3.1-8B` | HuggingFace model name or path |
| `--seed` | int | `12345` | Random seed |
| `--data_dir` | str | `cm_klar` | Root data directory |
| `--lang_codes` | str | *(one of these required)* | Comma-separated language codes, e.g. `hin,ben` |
| `--lang_file` | str | `None` | Path to a JSON file mapping model names → language lists |
| `--source_lang` | str | `Hindi` | Full source language name |
| `--source_script` | str | `Devanagari` | Script name |
| `--target_lang` | str | `English` | Translation target (typically `English`) |

> **Note:** Provide either `--lang_codes` or `--lang_file`, not both. `--lang_codes` takes priority.

**Lang file format** (for `--lang_file`):
```json
{
  "meta-llama/Llama-3.1-8B": ["hin", "ben"],
  "Qwen/Qwen2.5-7B": ["guj", "mal"]
}
```

**Example (Bengali):**
```bash
python 1_Call_EN.py \
  --model_name Qwen/Qwen2.5-7B \
  --lang_codes ben \
  --source_lang Bengali \
  --source_script Bengali \
  --target_lang English
```

**Example (Gujarati):**
```bash
python 1_Call_EN.py \
  --model_name Qwen/Qwen2.5-7B \
  --lang_codes guj \
  --source_lang Gujarati \
  --source_script Gujarati \
  --target_lang English
```

Each language has its own native script. Pass the appropriate `--source_lang` and `--source_script` pair for the language you are evaluating. See the table below for supported combinations:

| Language | `--lang_codes` | `--source_lang` | `--source_script` |
|---|---|---|---|
| Hindi | `hin` | `Hindi` | `Devanagari` |
| Bengali | `ben` | `Bengali` | `Bengali` |
| Gujarati | `guj` | `Gujarati` | `Gujarati` |
| Malayalam | `mal` | `Malayalam` | `Malayalam` |
| Odia | `ori` | `Odia` | `Odia` |
| Assamese | `asm` | `Assamese` | `Bengali` |

**Output directory:** `1_Call_EN/`

---

### `2_Call_CM.py` — Two-Stage Code-Mixed Translation Pipeline

**What it does:**
Splits the translation and answering into **two separate LLM calls**:

**Stage 1 — Translation:**
The model receives the source-language question and outputs only the code-mixed translation:
```json
{"translation": "<Hinglish question>"}
```

**Stage 2 — Answer:**
The translated (code-mixed) question from Stage 1 is fed into a new prompt, and the model outputs only the answer:
```json
{"answer": "<answer in source script>"}
```

Both stages use few-shot demonstrations drawn from the dataset (in the source language). The Stage 2 prompt uses the *translated* question from Stage 1 as the test question, not the original source-language question.

**Key idea:** A fully explicit two-stage pipeline tests whether decoupling translation from answering — giving the model a dedicated call for each — yields better results than doing both in one call.

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model_name` | str | `google/gemma-7b` | HuggingFace model name or path |
| `--seed` | int | `12345` | Random seed |
| `--lang_code` | str | `hin` | Comma-separated language codes |
| `--data_dir` | str | `cm_klar` | Root data directory |
| `--output_dir` | str | `filter_knowns_implicit-trans` | *(overridden by `OUTPUT_DIR = "2_Call_CM"`)* |
| `--source_lang` | str | *(required)* | Source language name |
| `--source_script` | str | *(required)* | Source script name |
| `--target_lang` | str | *(required)* | Code-mixed target language name |

**Internal constants:**

| Constant | Description |
|---|---|
| `OUTPUT_DIR` | Top-level output directory (default: `"2_Call_CM"`) |
| `DATA_DIR` | Data root (default: `"cm_klar"`) |
| `DICTIONARY_PATH` | Optional path to a lexicon JSON file |

**Example:**
```bash
python 2_Call_CM.py \
  --model_name Qwen/Qwen2.5-7B \
  --lang_code mal \
  --source_lang Malayalam \
  --source_script Malayalam \
  --target_lang Manglish
```

**Output directory:** `2_Call_CM/`

---

### `2_Call_EN.py` — Two-Stage English Translation Pipeline

**What it does:**
Identical two-stage structure as `2_Call_CM.py`, but Stage 1 translates into **English** instead of a code-mixed language:

- **Stage 1:** Source language → English translation
- **Stage 2:** English question → answer in source script

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model_name` | str | `meta-llama/Llama-3.1-8B` | HuggingFace model name or path |
| `--seed` | int | `12345` | Random seed |
| `--data_dir` | str | `cm_klar` | Root data directory |
| `--lang_codes` | str | *(one required)* | Comma-separated language codes |
| `--lang_file` | str | `None` | JSON file mapping model names → language lists |
| `--source_lang` | str | `Hindi` | Source language name |
| `--source_script` | str | `Devanagari` | Source script name |
| `--target_lang` | str | `English` | Target language (always English here) |

**Example:**
```bash
python 2_Call_EN.py \
  --model_name meta-llama/Llama-3.1-8B \
  --lang_codes hin \
  --source_lang Hindi \
  --source_script Devanagari \
  --target_lang English
```

**Output directory:** `2_Call_EN/`

---

## Comparison of All Scripts

| Script | Calls | Translation Written? | Target Language | Candidates Shown? |
|---|---|---|---|---|
| `Baseline_filter_knowns.py` | 1 | No | None | No |
| `Baseline_filter_knowns_with_obj.py` | 1 | No | None | Yes |
| `Implicit_CM.py` | 1 | No (internal only) | Code-mixed (Hinglish etc.) | Yes |
| `Implicit_EN.py` | 1 | No (internal only) | English | Yes |
| `1_Call_CM.py` | 1 | Yes (in output JSON) | Code-mixed | Yes |
| `1_Call_EN.py` | 1 | Yes (in output JSON) | English | Yes |
| `2_Call_CM.py` | 2 | Yes (Stage 1 output) | Code-mixed | Yes |
| `2_Call_EN.py` | 2 | Yes (Stage 1 output) | English | Yes |

---

## Common Setup

### Requirements

```bash
pip install vllm torch transformers datasets tqdm
```

### Model Support

All scripts are compatible with any model loadable via `vllm.LLM`. The scripts have been tested with:
- `meta-llama/Llama-3.1-8B`
- `Qwen/Qwen2.5-7B`
- `google/gemma-7b`

### GPU Memory

All scripts use `gpu_memory_utilization=0.25` and `max_model_len=4096` for conservative GPU usage. Increase `gpu_memory_utilization` if your GPU allows it (e.g., `0.85` for a dedicated 80GB A100).

### Language Config Files (for Implicit scripts)

`Implicit_CM.py` and `Implicit_EN.py` require per-language config files at:
```
configs/lang/<lang_code>.json
```

Each language needs its own config file. Examples:

`configs/lang/hin.json`:
```json
{
  "source_lang": "Hindi",
  "source_script": "Devanagari",
  "target_lang": "Hinglish",
  "target_script": "Latin"
}
```

`configs/lang/ben.json`:
```json
{
  "source_lang": "Bengali",
  "source_script": "Bengali",
  "target_lang": "Benglish",
  "target_script": "Latin"
}
```

`configs/lang/guj.json`:
```json
{
  "source_lang": "Gujarati",
  "source_script": "Gujarati",
  "target_lang": "Gujlish",
  "target_script": "Latin"
}
```

`configs/lang/mal.json`:
```json
{
  "source_lang": "Malayalam",
  "source_script": "Malayalam",
  "target_lang": "Manglish",
  "target_script": "Latin"
}
```

The `target_lang` is the name for the code-mixed variety of that language (source grammar + English vocabulary in Roman script). For the `Implicit_EN.py` script, the `target_lang` is always overridden to `"English"` regardless of what's in the config file.

---

## Example: Running a Full Experiment

```bash
# Baseline — no translation, no candidates
python Baseline_filter_knowns.py --model_name meta-llama/Llama-3.1-8B --lang_codes hin

# Baseline — no translation, with candidates
python Baseline_filter_knowns_with_obj.py --model_name meta-llama/Llama-3.1-8B --lang_codes hin

# Implicit: model translates internally to Hinglish, outputs only answer
python Implicit_CM.py --model_name meta-llama/Llama-3.1-8B --lang hin --mode hinglish

# Implicit: model translates internally to English, outputs only answer
python Implicit_EN.py --model_name meta-llama/Llama-3.1-8B --lang hin --mode english

# 1-Call: model outputs Hinglish translation + answer in one call
python 1_Call_CM.py \
  --model_name meta-llama/Llama-3.1-8B \
  --lang_code hin \
  --source_lang Hindi \
  --source_script Devanagari \
  --target_lang Hinglish

# 2-Call: separate calls for translation (→ Hinglish) and answering
python 2_Call_CM.py \
  --model_name meta-llama/Llama-3.1-8B \
  --lang_code hin \
  --source_lang Hindi \
  --source_script Devanagari \
  --target_lang Hinglish
```

---

## Notes

- **Batch size** is set to `1` in all scripts with structured output decoding (guided JSON). The baseline scripts use batch sizes of `1` or `32`. Adjust as needed for throughput.
- **Structured output** is enforced via `vllm.sampling_params.StructuredOutputsParams` with a JSON schema, ensuring the model always returns valid, parseable JSON.
- The `translation` field in 1-Call and 2-Call scripts is **logged only** — it is never fed back into the evaluation logic. Only the `answer` field affects accuracy.
- When `object_candidates` is absent from a sample, the scripts fall back to open-generation with prefix matching.

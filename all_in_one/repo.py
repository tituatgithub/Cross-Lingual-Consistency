================================================
FILE: Evaluation-Scripts/1_call_cm_placeholder.py
================================================
import os
import json
import glob
import random
import argparse
import numpy
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
from tqdm import tqdm
import time

# ==== CONFIGURATION - SET THIS ONCE ====
OUTPUT_DIR      = "1_call_cm_placeholder_corr"
DATA_DIR        = "cm_klar"
DICTIONARY_PATH = None   # e.g. "dicts/hin_eng_dict.json" or None to disable
# ======================================


# ==== Argument parser ====
parser = argparse.ArgumentParser()
parser.add_argument("--model_name",    type=str, default="google/gemma-7b")
parser.add_argument("--seed",          type=int, default=12345)
parser.add_argument("--lang_code",     type=str, default="hin")
parser.add_argument("--data_dir",      type=str, default="cm_klar")
parser.add_argument("--output_dir",    type=str, default="filter_knowns_implicit-trans")
parser.add_argument("--source_lang", type=str, required=True)
parser.add_argument("--source_script", type=str, required=True)
parser.add_argument("--target_lang", type=str, required=True)


args = parser.parse_args()

SOURCE_LANG   = args.source_lang
SOURCE_SCRIPT = args.source_script
TARGET_LANG   = args.target_lang

model_name = args.model_name

# ==== Set random seed ====
def set_seed(seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)

# ==== Define valid languages and relations ====
# To run a different model/language: add/uncomment its entry here.
# The lang code must match a key in CM_TEMPLATES above.
languages = {
    "meta-llama/Llama-3.1-8B": ["ben", "asm", "guj", "mal", "ori"],
    "Qwen/Qwen2.5-7B":         ["ben", "asm", "guj", "mal", "ori"],
    "google/gemma-7b":         ["ben", "asm", "guj", "mal", "ori"],
}

relations = [
    "applies_to_jurisdiction", "capital", "capital_of", "continent",
    "country_of_citizenship", "developer", "field_of_work", "headquarters_location",
    "instrument", "language_of_work_or_name", "languages_spoken", "location_of_formation",
    "manufacturer", "native_language", "occupation", "official_language",
    "owned_by", "place_of_birth", "place_of_death", "religion"
]


valid_langs = set(c.strip() for c in args.lang_code.split(",") if c.strip())
valid_rels  = set(relations)

# ==== Language config lookup ====
# Maps lang code → source_lang, source_script, target_lang.
# To add a new language: add one entry here, nothing else changes.
lang_code     = list(valid_langs)[0]           # e.g. "hin", "ben", "mal"
src_key       = SOURCE_LANG.lower()            # → result field: "question_hindi"
tgt_key       = TARGET_LANG.lower()            # → result field: "hinglish_question"

print(f"[Config] model={model_name}  lang={lang_code}  {SOURCE_LANG} → {TARGET_LANG}")
print(f"[Config] DATA_DIR={DATA_DIR}  OUTPUT_DIR={OUTPUT_DIR}")



# ==== Load CM dictionary (optional lexicon hints) ====
cm_dictionary = {}
if DICTIONARY_PATH and os.path.exists(DICTIONARY_PATH):
    with open(DICTIONARY_PATH, "r", encoding="utf-8") as f:
        cm_dictionary = json.load(f)
    print(f"[Dictionary] Loaded {len(cm_dictionary)} entries from {DICTIONARY_PATH}")
else:
    print(f"[Dictionary] None loaded — proceeding without lexicon hints")

# ==== Group file paths by relation ====
json_paths = glob.glob(f"{DATA_DIR}/*/*.json")
path_map   = defaultdict(dict)

for path in json_paths:
    lang = os.path.basename(os.path.dirname(path))
    rel  = os.path.splitext(os.path.basename(path))[0]
    if lang in valid_langs and rel in valid_rels:
        path_map[rel][lang] = path

# ==== Load samples ====
samples = []

for rel, lang_paths in path_map.items():
    if not all(lang in lang_paths for lang in valid_langs):
        continue
    for lang in valid_langs:
        with open(lang_paths[lang], "r", encoding="utf-8") as f:
            content        = json.load(f)
            loaded_samples = content["samples"]
            template       = content["prompt_templates"][0]
            for sample in loaded_samples:
                new_sample = {
                    "subject":           sample["subject"],
                    "object":            sample["object"],
                    "object_candidates": sample.get("object_candidates", None),
                    "language":          lang,
                    "relation":          rel,
                    "template":          template,
                    "index":             sample["index"]
                }
                samples.append(new_sample)

dataset = Dataset.from_list(samples)
print(f"[Dataset] Loaded {len(samples)} samples across {len(path_map)} relations")

# ==== Model loading ====
llm = LLM(
    model=model_name,
    trust_remote_code=True,
    gpu_memory_utilization=0.20,
    max_model_len=4096,
    max_num_seqs=16
)

# ==== Guided JSON schema ====
# The model outputs a two-field JSON in a single call:
#   "translation" — TARGET_LANG codemix form of the SOURCE_LANG question (logging only)
#   "answer"      — SOURCE_LANG answer in SOURCE_SCRIPT, selected from object_candidates
GUIDED_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "translation": {"type": "string"},
        "answer":      {"type": "string"}
    },
    "required": ["translation", "answer"],
    "additionalProperties": False
}

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=120,
    structured_outputs=StructuredOutputsParams(json=json.dumps(GUIDED_JSON_SCHEMA))
)


# ==== Helpers ====
def parse_candidates(raw) -> list[str]:
    """
    Accept either:
      - a list   ["Boeing", "IBM", ...]
      - a string "Boeing, IBM, Google, ..."
    Returns a list of stripped candidate strings.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [c.strip() for c in raw if c.strip()]
    return [c.strip() for c in raw.split(",") if c.strip()]


def is_nontrivial_prefix(prediction: str, target: str) -> bool:
    target     = target.lower().strip()
    prediction = prediction.lower().strip()
    return len(prediction) > 0 and target.startswith(prediction)


def overlapping_ratio(list1, list2):
    set1         = set(list1)
    set2         = set(list2)
    intersection = set1.intersection(set2)
    union        = set1.union(set2)
    return len(intersection) / len(union) if union else 0


def match_against_candidates(prediction: str, candidates: list[str], target: str) -> tuple[bool, str]:
    """
    Try to find the best-matching candidate for the model's raw prediction.
    Strategy (in order):
      1. Exact match (case-insensitive)
      2. Candidate is a prefix of prediction  (e.g. pred="Microsoft Corp" cand="Microsoft")
      3. Prediction is a prefix of candidate  (existing nontrivial-prefix logic)
      4. Candidate appears as a substring in prediction
    Returns (is_correct, matched_candidate).
    If no candidate matches, falls back to the original prefix logic against target.
    """
    pred_low  = prediction.lower().strip()
    best_cand = None

    for cand in candidates:
        cand_low = cand.lower().strip()
        if not cand_low:
            continue
        if pred_low == cand_low:
            best_cand = cand; break
        if pred_low.startswith(cand_low):
            best_cand = cand; break
        if cand_low.startswith(pred_low) and len(pred_low) > 0:
            best_cand = cand; break
        if cand_low in pred_low:
            best_cand = cand; break

    if best_cand is not None:
        is_correct = best_cand.lower().strip() == target.lower().strip()
        return is_correct, best_cand

    is_correct = is_nontrivial_prefix(prediction, target) or is_nontrivial_prefix(target, prediction)
    return is_correct, prediction


def parse_json_output(raw_text: str) -> tuple[str, str]:
    """
    Parse the guided-JSON output from the model.
    Returns (translation, answer). Falls back gracefully on malformed output.
    """
    try:
        parsed      = json.loads(raw_text.strip())
        translation = parsed.get("translation", "").strip()
        answer      = parsed.get("answer", "").strip()
        return translation, answer
    except (json.JSONDecodeError, AttributeError):
        return "", raw_text.strip()


# ==== Evaluation ====
def evaluate(llm, dataset, max_new_tokens=10, n_shot=3):
    test_data        = list(dataset)
    correct_total    = 0
    total_total      = 0
    per_lang_results = defaultdict(lambda: {"correct": 0, "total": 0, "correct_indices": []})
    detailed_results = []

    model_safe_name = model_name.replace("/", "_")
    lang_name       = list(valid_langs)[0]
    output_subdir   = os.path.join(OUTPUT_DIR, model_safe_name, lang_name)
    os.makedirs(output_subdir, exist_ok=True)

    summary_path  = os.path.join(output_subdir, "summary.json")
    detailed_path = os.path.join(output_subdir, "detailed.json")
    live_path     = os.path.join(output_subdir, "LIVE.json")

    candidates_by_key = defaultdict(list)
    for i, ex in enumerate(test_data):
        key = (ex.get("relation"), ex.get("language"))
        candidates_by_key[key].append((i, ex))

    # ==== Base prompt — fully templated via SOURCE_LANG / SOURCE_SCRIPT / TARGET_LANG ====
    # All three are resolved from CM_TEMPLATES at the top of the script.
    dict_hint = ""
    if cm_dictionary:
        sample_entries = list(cm_dictionary.items())[:10]
        dict_hint = (
            f"\nDictionary hint — use these {SOURCE_LANG}→English mappings when writing the {TARGET_LANG} translation:\n"
            + "\n".join(f"  {k} → {v}" for k, v in sample_entries)
            + "\n"
        )


    # base_prompt = (
    #     f"You are answering factual questions written in {SOURCE_LANG} ({SOURCE_SCRIPT} script).\n"
    #     f"Step 1 (internal only): Mentally translate the {SOURCE_LANG} question into {TARGET_LANG} "
    #     f"({SOURCE_LANG} grammar + English content words in Roman script) to understand it clearly.\n"
    #     "Step 2: Based on your understanding, output a JSON object with exactly two fields:\n"
    #     f"  \"translation\": the {TARGET_LANG} translation of the question,\n"
    #     f"  \"answer\": the correct answer in {SOURCE_SCRIPT} script, chosen from the provided candidates.\n\n"
    #     "Rules:\n"
    #     "- The answer MUST be one of the listed candidates.\n"
    #     f"- The answer MUST be written in {SOURCE_SCRIPT} script only.\n"
    #     f"- Do NOT use English in the answer field.\n"
    #     f"{dict_hint}\n"
    # )
    # base_prompt = (
    #     f"You are answering factual questions written in {SOURCE_LANG} ({SOURCE_SCRIPT} script).\n"
    #     f"First, translate the {SOURCE_LANG} question into {TARGET_LANG} "
    #     f"({SOURCE_LANG} grammar + English content words in Roman script).\n"
    #     f"Then answer based on that understanding.\n\n"
    #     "Output a JSON object with exactly two fields:\n"
    #     f"  \"translation\": the {TARGET_LANG} translation (Roman script, mixed {SOURCE_LANG}+English),\n"
    #     f"  \"answer\": the correct answer in {SOURCE_SCRIPT} script, chosen from candidates.\n\n"
    #     "Rules:\n"
    #     f"- The translation MUST use {SOURCE_LANG} grammar with English content words\n"
    #     f"- The translation MUST be in Roman/Latin script (not {SOURCE_SCRIPT})\n"
    #     "- The answer MUST be one of the listed candidates\n"
    #     f"- The answer MUST be in {SOURCE_SCRIPT} script only\n"
    #     f"{dict_hint}\n"
    # )

    base_prompt = (
        f"You are answering factual questions written in {SOURCE_LANG} ({SOURCE_SCRIPT} script).\n\n"

        f"You MUST perform TWO steps in ONE response:\n"
        f"1. Convert the question into {TARGET_LANG} (code-mixed form)\n"
        f"2. Provide the correct answer\n\n"

        f"STEP 1 — CODE-MIXED TRANSLATION (CRITICAL):\n"
        f"- First convert the input into a romanized version preserving ALL words\n"
        f"- Then replace ONLY a few content words (nouns, main verbs, adjectives) with English\n"
        f"- Keep function words (question words, particles, auxiliaries, postpositions) unchanged\n"
        f"- Preserve word order exactly\n"
        f"- DO NOT rewrite or paraphrase\n\n"

        f"ILLUSTRATION:\n"
        f"Input: Bharat ki rajdhani kya hai?\n"
        f"Romanized: Bharat ki rajdhani kya hai?\n"
        f"Code-mixed: Bharat ki capital kya hai?\n\n"

        f"RULES FOR TRANSLATION:\n"
        f"- The translation MUST NOT be a fluent English sentence\n"
        f"- The translation MUST NOT be a full translation\n"
        f"- The translation MUST preserve structure of the original sentence\n"
        f"- At least 50% of the words must remain from the original (after romanization)\n"
        f"- Do NOT convert into known English question templates\n\n"

        f"STEP 2 — ANSWERING:\n"
        f"- Answer based on the meaning of the question\n"
        f"- The answer MUST be consistent with the generated translation\n"
        f"- Both translation and answer must reflect the same interpretation\n"
        f"- Answer MUST be one of the candidates\n"
        f"- Answer MUST be in {SOURCE_SCRIPT} script only\n"
        f"- Do NOT use English in the answer\n\n"

        f"OUTPUT FORMAT (STRICT):\n"
        f"{{\n"
        f"  \"translation\": \"<code-mixed question in Roman script>\",\n"
        f"  \"answer\": \"<answer in {SOURCE_SCRIPT} script>\"\n"
        f"}}\n\n"

        f"{dict_hint}\n"
    )

    print(f"\n[Evaluating {len(test_data)} examples with {n_shot}-shot prompts]")
    print(f"📁 Output directory: {output_subdir}")
    print(f"💡 Tip: Run 'tail -f {live_path}' in another terminal to watch progress\n")

    prompts         = []
    prompt_metadata = []
    batch_size      = 8

    live_data = {
        "progress":        "0/0",
        "percent":         "0%",
        "current_example": None,
        "results_so_far":  []
    }
    with open(live_path, "w", encoding="utf-8") as f:
        json.dump(live_data, f, indent=2, ensure_ascii=False)

    pbar = tqdm(total=len(test_data), desc="Processing", unit="ex")

    for idx, ex in enumerate(test_data):
        lang     = ex.get("language", "unknown")
        relation = ex.get("relation", "unknown")
        index    = ex.get("index", None)

        key               = (relation, lang)
        candidates_pool   = [c[1] for c in candidates_by_key[key] if c[0] != idx]
        demonstrations    = random.sample(candidates_pool, min(n_shot, len(candidates_pool)))
        object_candidates = parse_candidates(ex.get("object_candidates"))

        # ---- Build few-shot prompt ----
        # demo_trans_fn() pulls the language-correct codemix question from CM_TEMPLATES.
        # e.g. hin: "Gandhi ka place of birth kya hai?"
        #      ben: "Tagore er place of birth ki?"
        prompt = base_prompt
        for d in demonstrations:
            demo_question    = d["template"].replace("<subject>", d["subject"]).replace("<mask>", "").strip()
            demo_candidates  = parse_candidates(d.get("object_candidates"))
            demo_cands_str   = ", ".join(demo_candidates) if demo_candidates else d["object"]
            prompt += (
                f"Q: {demo_question}\n"
                f"Candidates: {demo_cands_str}\n"
                f"Output: {{\"answer\": \"{d['object']}\"}}\n\n"
            )

        # ---- Test question ----
        test_question  = ex["template"].replace("<subject>", ex["subject"]).replace("<mask>", "").strip()
        candidates_str = ", ".join(object_candidates) if object_candidates else ""


        prompt += (
            f"Q: {test_question}\n"
            f"Candidates: {candidates_str}\n"
            f"Output:"
        )

        prompts.append(prompt)
        prompt_metadata.append((ex, lang, index, object_candidates, prompt))

        if len(prompts) == batch_size or idx == len(test_data) - 1:
            outputs = llm.generate(prompts, sampling_params)

            for i, output in enumerate(outputs):
                ex, lang, index, object_candidates, final_prompt = prompt_metadata[i]
                raw_prediction = output.outputs[0].text

                target   = ex["object"].strip()
                question = ex["template"].replace("<subject>", ex["subject"]).replace("<mask>", "").strip()

                # tgt_question = "translation" field → TARGET_LANG codemix form (logged only)
                # prediction   = "answer" field     → SOURCE_LANG in SOURCE_SCRIPT
                tgt_question, prediction = parse_json_output(raw_prediction)

                if object_candidates:
                    match, matched_candidate = match_against_candidates(prediction, object_candidates, target)
                else:
                    match = is_nontrivial_prefix(prediction, target) or is_nontrivial_prefix(target, prediction)
                    matched_candidate = prediction

                correct_total += int(match)
                total_total   += 1
                per_lang_results[lang]["correct"] += int(match)
                per_lang_results[lang]["total"]   += 1
                if match:
                    per_lang_results[lang]["correct_indices"].append(index)

                # Field names auto-derived from CM_TEMPLATES:
                result_entry = {
                    "index":                   index,
                    "relation":                relation,
                    "subject":                 ex["subject"],
                    f"question_{src_key}":     question,
                    f"{tgt_key}_question":     tgt_question,
                    "model_prediction":        prediction,
                    "matched_candidate":       matched_candidate,
                    "object_candidates":       object_candidates if object_candidates else None,
                    "used_candidates":         bool(object_candidates),
                    "ground_truth":            target,
                    "is_correct":              bool(match),
                    "raw_output":              raw_prediction,
                    "final_prompt":            final_prompt
                }

                detailed_results.append(result_entry)

                # ==== LIVE UPDATE ====
                progress_percent = (total_total / len(test_data)) * 100
                live_data = {
                    "progress": f"{total_total}/{len(test_data)}",
                    "percent":  f"{progress_percent:.1f}%",
                    "current_example": {
                        "index":               index,
                        "subject":             ex["subject"],
                        f"question_{src_key}": question,
                        f"{tgt_key}_question": tgt_question,
                        "model_prediction":    prediction,
                        "matched_candidate":   matched_candidate,
                        "ground_truth":        target,
                        "is_correct":          bool(match),
                        "final_prompt":        final_prompt
                    },
                    "running_accuracy": f"{(correct_total/total_total)*100:.2f}%" if total_total > 0 else "0%",
                    "results_so_far":   detailed_results[-50:]
                }

                with open(live_path, "w", encoding="utf-8") as f:
                    json.dump(live_data, f, indent=2, ensure_ascii=False)

                status = "✅" if match else "❌"
                print(f"\n{status} [{total_total}/{len(test_data)}] Subject: {ex['subject']}")
                print(f"   {SOURCE_LANG} Q:    {question}")
                print(f"   {TARGET_LANG}:      {tgt_question}")
                print(f"   Candidates:        {', '.join(object_candidates) if object_candidates else 'N/A (open gen)'}")
                print(f"   Raw prediction:    {prediction}")
                print(f"   Matched candidate: {matched_candidate}")
                print(f"   Ground truth:      {target}")
                print(f"   Running Acc:       {(correct_total/total_total)*100:.2f}%")

            prompts.clear()
            prompt_metadata.clear()
            pbar.update(batch_size)

    pbar.close()

    overall_acc = correct_total / total_total if total_total > 0 else 0

    results = {
        "overall_acc":      overall_acc,
        "overall_clc":      None,
        "per_language_acc": {},
        "per_language_clc": {}
    }

    for lang, res in sorted(per_lang_results.items()):
        lang_acc = res["correct"] / res["total"] if res["total"] > 0 else 0
        results["per_language_acc"][lang] = lang_acc

    langs = sorted(list(per_lang_results.keys()))
    for lang in langs:
        others = [l for l in langs if l != lang]
        scores = [
            overlapping_ratio(
                per_lang_results[lang]["correct_indices"],
                per_lang_results[other]["correct_indices"]
            ) for other in others
        ]
        consistency = sum(scores) / len(scores) if scores else 0
        results["per_language_clc"][lang] = consistency

    results["overall_clc"] = (
        sum(results["per_language_clc"].values()) / len(results["per_language_clc"].values())
        if results["per_language_clc"] else 0
    )

    print(f"\n📊 Overall Accuracy: {overall_acc:.2%}")
    for lang, res in sorted(per_lang_results.items()):
        lang_acc = res["correct"] / res["total"] if res["total"] > 0 else 0
        print(f"  {lang}: {lang_acc:.2%} ({res['correct']} / {res['total']})")
    print(f'\n📊 Overall CLC: {results["overall_clc"]:.2%}')

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=4, ensure_ascii=False)

    live_data["status"]         = "COMPLETED"
    live_data["final_accuracy"] = f"{overall_acc:.2f}"
    with open(live_path, "w", encoding="utf-8") as f:
        json.dump(live_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved:")
    print(f"   Summary  → {summary_path}")
    print(f"   Detailed → {detailed_path}")
    print(f"   Live Log → {live_path}")

    return results


# Run evaluation
evaluate(llm, dataset, max_new_tokens=10, n_shot=3)



================================================
FILE: Evaluation-Scripts/1_call_en_placeholder.py
================================================
import os
import json
import glob
import random
import argparse
import numpy
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
from tqdm import tqdm
import time

# ==== CONFIGURATION - SET THIS ONCE ====
OUTPUT_DIR = "1_call_en_placeholder_corr_final"
# ======================================

# ==== Argument parser ====
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B",
                    help="Model name / HF hub path, e.g. meta-llama/Llama-3.1-8B")
parser.add_argument("--lang_file", type=str, default=None,
                    help="Path to a JSON file mapping model names to language lists, "
                         "e.g. {\"meta-llama/Llama-3.1-8B\": [\"hin\", \"ben\"]}")
parser.add_argument("--seed", type=int, default=12345)
# ==== Dataset control: which directory and which language-code subdirs to load ====
parser.add_argument("--data_dir", type=str, default="cm_klar",
                    help="Root data directory to glob for JSON files, e.g. cm_klar, ben_klar")
parser.add_argument("--lang_codes", type=str, default=None,
                    help="Comma-separated language-code subdirs to load, e.g. hin,ben,ori. "
                         "Overrides --lang_file / inline languages dict for this run.")
# ==== Prompt placeholders: e.g. --source_lang Bengali --source_script Bengali --target_lang English ====
parser.add_argument("--source_lang", type=str, default="Hindi",
                    help="Source language name, e.g. Hindi, Bengali, Odia")
parser.add_argument("--source_script", type=str, default="Devanagari",
                    help="Script name for the source language, e.g. Devanagari, Bengali, Odia")
parser.add_argument("--target_lang", type=str, default="English",
                    help="Target language for the translation field, e.g. English")
args = parser.parse_args()

model_name    = args.model_name
SOURCE_LANG   = args.source_lang    # e.g. "Hindi", "Bengali"
SOURCE_SCRIPT = args.source_script  # e.g. "Devanagari", "Bengali"
TARGET_LANG   = args.target_lang    # e.g. "English"

# ==== Key slugs for result dict fields (derived from language args) ====
# e.g. SOURCE_LANG="Bengali" → src_key="bengali" → field "question_bengali"
src_key = SOURCE_LANG.lower()   # used as: f"question_{src_key}"
tgt_key = TARGET_LANG.lower()   # used as: f"{tgt_key}_question"

# ==== Set random seed ====
def set_seed(seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)


if args.lang_codes:
    valid_langs = set(c.strip() for c in args.lang_codes.split(",") if c.strip())

elif args.lang_file:
    with open(args.lang_file, "r", encoding="utf-8") as _lf:
        languages = json.load(_lf)
    valid_langs = set(languages[model_name])

else:
    raise ValueError("You must provide either --lang_codes or --lang_file")

relations = [
    "applies_to_jurisdiction", "capital", "capital_of", "continent",
    "country_of_citizenship", "developer", "field_of_work", "headquarters_location",
    "instrument", "language_of_work_or_name", "languages_spoken", "location_of_formation",
    "manufacturer", "native_language", "occupation", "official_language",
    "owned_by", "place_of_birth", "place_of_death", "religion"
]
valid_rels = set(relations)

print(f"[Dataset] data_dir={args.data_dir}  lang_codes={sorted(valid_langs)}")

# ==== Group file paths by relation ====
json_paths = glob.glob(f"{args.data_dir}/*/*.json")
path_map = defaultdict(dict)

for path in json_paths:
    lang = os.path.basename(os.path.dirname(path))
    rel = os.path.splitext(os.path.basename(path))[0]
    if lang in valid_langs and rel in valid_rels:
        path_map[rel][lang] = path

# ==== Load samples ====
samples = []

for rel, lang_paths in path_map.items():
    if not all(lang in lang_paths for lang in valid_langs):
        continue
    for lang in valid_langs:
        with open(lang_paths[lang], "r", encoding="utf-8") as f:
            content = json.load(f)
            loaded_samples = content["samples"]
            template = content["prompt_templates"][0]
            for sample in loaded_samples:
                new_sample = {
                    "subject": sample["subject"],
                    "object": sample["object"],
                    "object_candidates": sample.get("object_candidates", None),
                    "language": lang,
                    "relation": rel,
                    "template": template,
                    "index": sample["index"]
                }
                samples.append(new_sample)

dataset = Dataset.from_list(samples)

# ==== Model loading ====
llm = LLM(
    model=model_name,
    trust_remote_code=True,
    gpu_memory_utilization=0.20,
    max_model_len=4096,
    max_num_seqs=16
)

# ==== Guided JSON schema ====
# The model outputs a two-field JSON in a single call:
#   "translation" — {TARGET_LANG} rendering of the {SOURCE_LANG} question (for logging only)
#   "answer"      — {SOURCE_LANG} answer in {SOURCE_SCRIPT}, selected from object_candidates
GUIDED_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "translation": {"type": "string"},
        "answer":      {"type": "string"}
    },
    "required": ["translation", "answer"],
    "additionalProperties": False
}



# guided_decoding = GuidedDecodingParams(json=GUIDED_JSON_SCHEMA)


sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=120,
    structured_outputs=StructuredOutputsParams(json=json.dumps(GUIDED_JSON_SCHEMA))
)



# ==== Helpers ====
def parse_candidates(raw) -> list[str]:
    """
    Accept either:
      - a list   ["Boeing", "IBM", ...]
      - a string "Boeing, IBM, Google, ..."
    Returns a list of stripped candidate strings.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [c.strip() for c in raw if c.strip()]
    # string form
    return [c.strip() for c in raw.split(",") if c.strip()]


def is_nontrivial_prefix(prediction: str, target: str) -> bool:
    target = target.lower().strip()
    prediction = prediction.lower().strip()
    return len(prediction) > 0 and target.startswith(prediction)


def overlapping_ratio(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0


def match_against_candidates(prediction: str, candidates: list[str], target: str) -> tuple[bool, str]:
    """
    Try to find the best-matching candidate for the model's raw prediction.

    Strategy (in order):
      1. Exact match (case-insensitive)
      2. Candidate is a prefix of prediction  (e.g. pred="Microsoft Corp" cand="Microsoft")
      3. Prediction is a prefix of candidate  (existing nontrivial-prefix logic)
      4. Candidate appears as a substring in prediction

    Returns (is_correct, matched_candidate).
    If no candidate matches, falls back to the original prefix logic against target.
    """
    pred_low = prediction.lower().strip()

    best_cand = None
    for cand in candidates:
        cand_low = cand.lower().strip()
        if not cand_low:
            continue
        # exact
        if pred_low == cand_low:
            best_cand = cand
            break
        # candidate is prefix of prediction  ("Microsoft" in "Microsoft Corporation")
        if pred_low.startswith(cand_low):
            best_cand = cand
            break
        # prediction is prefix of candidate  ("Micro" for "Microsoft")
        if cand_low.startswith(pred_low) and len(pred_low) > 0:
            best_cand = cand
            break
        # substring
        if cand_low in pred_low:
            best_cand = cand
            break

    if best_cand is not None:
        is_correct = best_cand.lower().strip() == target.lower().strip()
        return is_correct, best_cand

    # No candidate matched — fall back to original logic against ground truth
    is_correct = is_nontrivial_prefix(prediction, target) or is_nontrivial_prefix(target, prediction)
    return is_correct, prediction


def parse_json_output(raw_text: str) -> tuple[str, str]:
    """
    Parse the guided-JSON output from the model.
    Returns (translation, answer). Falls back gracefully on malformed output.
    """
    try:
        parsed = json.loads(raw_text.strip())
        translation = parsed.get("translation", "").strip()
        answer      = parsed.get("answer", "").strip()
        return translation, answer
    except (json.JSONDecodeError, AttributeError):
        # Malformed output — return empty translation, raw text as answer
        return "", raw_text.strip()


# ==== Evaluation ====
def evaluate(llm, dataset, max_new_tokens=10, n_shot=3):
    test_data = list(dataset)
    correct_total = 0
    total_total = 0
    per_lang_results = defaultdict(lambda: {"correct": 0, "total": 0, "correct_indices": []})
    detailed_results = []

    model_safe_name = model_name.replace("/", "_")
    lang_name = list(valid_langs)[0]
    output_subdir = os.path.join(OUTPUT_DIR, model_safe_name, lang_name)
    os.makedirs(output_subdir, exist_ok=True)

    summary_path  = os.path.join(output_subdir, "summary.json")
    detailed_path = os.path.join(output_subdir, "detailed.json")
    live_path     = os.path.join(output_subdir, "LIVE.json")

    candidates_by_key = defaultdict(list)
    for i, ex in enumerate(test_data):
        key = (ex.get("relation"), ex.get("language"))
        candidates_by_key[key].append((i, ex))

    # base_prompt = (
    #     f"You are answering factual questions written in {SOURCE_LANG} ({SOURCE_SCRIPT} script).\n"
    #     f"First, translate the {SOURCE_LANG} question into {TARGET_LANG}.\n"
    #     f"Then answer based on that understanding.\n\n"
    #     "Output a JSON object with exactly two fields:\n"
    #     f"  \"translation\": the {TARGET_LANG} translation,\n"
    #     f"  \"answer\": the correct answer in {SOURCE_LANG} {SOURCE_SCRIPT} script, chosen from candidates.\n\n"
    #     "Rules:\n"
    #     f"- The translation MUST be in {TARGET_LANG} only (not {SOURCE_SCRIPT} script)\n"
    #     "- The answer MUST be one of the listed candidates\n"
    #     f"- The answer MUST be in {SOURCE_LANG} {SOURCE_SCRIPT} script only\n"
    #     f"- Do NOT use {TARGET_LANG} in the answer field\n\n"
    # )


# some native some english
    # base_prompt = (
    #     f"You are answering factual questions written in {SOURCE_LANG} ({SOURCE_SCRIPT} script).\n\n"

    #     f"You MUST perform BOTH steps:\n"
    #     f"1. Translate the question into {TARGET_LANG}\n"
    #     f"2. Answer using the candidates based on that translation\n\n"

    #     "Output MUST be a JSON object with EXACTLY two fields:\n"
    #     f"  \"translation\": the {TARGET_LANG} translation of the question,\n"
    #     f"  \"answer\": the correct answer in {SOURCE_LANG} ({SOURCE_SCRIPT} script)\n\n"

    #     "IMPORTANT RULES:\n"
    #     "- BOTH fields are REQUIRED\n"
    #     "- Missing \"translation\" makes the output INVALID\n"
    #     "- Generate \"translation\" FIRST, then \"answer\"\n"
    #     "- Do NOT skip translation even if examples omit it\n"
    #     f"- The translation MUST be in {TARGET_LANG} only (not {SOURCE_SCRIPT} script)\n"
    #     "- The answer MUST be one of the listed candidates\n"
    #     f"- The answer MUST be written in {SOURCE_LANG} ({SOURCE_SCRIPT} script) only\n"
    #     f"- Do NOT use {TARGET_LANG} in the answer field\n\n"

    #     "Start your output exactly like this:\n"
    #     "{\"translation\":"
    # )
    base_prompt = (
        f"You are answering factual questions written in {SOURCE_LANG} ({SOURCE_SCRIPT} script).\n\n"

        f"You MUST do BOTH of the following:\n"
        f"1. Translate the question into {TARGET_LANG}\n"
        f"2. Select the correct answer from the candidates using that understanding\n\n"

        "You MUST output a JSON object with EXACTLY two fields:\n"
        f"  \"translation\": a fluent {TARGET_LANG} sentence translating the question\n"
        f"  \"answer\": the correct answer in {SOURCE_LANG} ({SOURCE_SCRIPT} script)\n\n"

        "STRICT RULES:\n"
        "- BOTH fields are REQUIRED\n"
        "- Do NOT omit \"translation\"\n"
        "- Do NOT copy the original question into \"translation\"\n"
        "- The translation MUST be a proper natural sentence in the target language\n"
        "- The translation MUST NOT contain source script text\n"
        "- The answer MUST be chosen EXACTLY from the candidates\n"
        "- The answer MUST be in source script only\n"
        "- Do NOT include explanations\n\n"

        "FORMAT REQUIREMENT:\n"
        "{\"translation\": \"<English sentence>\", \"answer\": \"<candidate>\"}\n\n"
    )



    print(f"\n[Evaluating {len(test_data)} examples with {n_shot}-shot prompts]")
    print(f"📁 Output directory: {output_subdir}")
    print(f"💡 Tip: Run 'tail -f {live_path}' in another terminal to watch progress\n")

    prompts = []
    prompt_metadata = []
    batch_size = 8

    live_data = {
        "progress": "0/0",
        "percent": "0%",
        "current_example": None,
        "results_so_far": []
    }
    with open(live_path, "w", encoding="utf-8") as f:
        json.dump(live_data, f, indent=2, ensure_ascii=False)

    pbar = tqdm(total=len(test_data), desc="Processing", unit="ex")

    for idx, ex in enumerate(test_data):
        lang     = ex.get("language", "unknown")
        relation = ex.get("relation", "unknown")
        index    = ex.get("index", None)

        key = (relation, lang)
        candidates_pool = [c[1] for c in candidates_by_key[key] if c[0] != idx]
        demonstrations  = random.sample(candidates_pool, min(n_shot, len(candidates_pool)))

        object_candidates = parse_candidates(ex.get("object_candidates"))

        # ---- Build single-call few-shot prompt ----
        # Few-shot demos illustrate the JSON output format.
        # Each demo shows a {SOURCE_LANG} question + candidates → JSON with translation + {SOURCE_LANG} answer.
        prompt = base_prompt
        for d in demonstrations:
            demo_question = d["template"].replace("<subject>", d["subject"]).replace("<mask>", "").strip()
            demo_candidates = parse_candidates(d.get("object_candidates"))
            demo_cands_str = ", ".join(demo_candidates) if demo_candidates else d["object"]
            prompt += (
                f"Q: {demo_question}\n"
                f"Candidates: {demo_cands_str}\n"
                f"Output: {{\"answer\": \"{d['object']}\"}}\n\n"
            )
            # prompt += (
            #     f"Q ({SOURCE_LANG}): {demo_question}\n"
            #     f"Candidates: {demo_cands_str}\n"
            #     # f"Output: {{\"translation\": \"({TARGET_LANG} translation)\", \"answer\": \"{d['object']}\"}}\n\n"
            #     f"Output: {{\"translation\": \"What is the {d['relation'].replace('_', ' ')} of {d['subject']}?\", \"answer\": \"{d['object']}\"}}\n\n"
            # )

        # ---- Test question (always with candidates) ----
        test_question = ex["template"].replace("<subject>", ex["subject"]).replace("<mask>", "").strip()
        candidates_str = ", ".join(object_candidates) if object_candidates else ""

        # prompt += (
        #     # f"Q ({SOURCE_LANG}): {test_question}\n"
        #     f"Candidates: {candidates_str}\n"
        #     f"Output:"
        # )
        prompt += (
            f"Q: {test_question}\n"
            f"Candidates: {candidates_str}\n"
            f"Output:"
        )


        prompts.append(prompt)
        prompt_metadata.append((ex, lang, index, object_candidates, prompt))

        if len(prompts) == batch_size or idx == len(test_data) - 1:
            outputs = llm.generate(prompts, sampling_params)

            for i, output in enumerate(outputs):
                ex, lang, index, object_candidates, final_prompt = prompt_metadata[i]
                raw_prediction = output.outputs[0].text

                target   = ex["object"].strip()
                question = ex["template"].replace("<subject>", ex["subject"]).replace("<mask>", "").strip()

                # ---- Parse the single JSON output ----
                tgt_question, prediction = parse_json_output(raw_prediction)
                # prediction   = the "answer" field ({SOURCE_LANG} {SOURCE_SCRIPT})
                # tgt_question = the "translation" field ({TARGET_LANG}, logged only)

                # ---- Candidate-aware matching on the {SOURCE_LANG} answer ----
                if object_candidates:
                    match, matched_candidate = match_against_candidates(prediction, object_candidates, target)
                else:
                    match = is_nontrivial_prefix(prediction, target) or is_nontrivial_prefix(target, prediction)
                    matched_candidate = prediction

                correct_total += int(match)
                total_total   += 1
                per_lang_results[lang]["correct"] += int(match)
                per_lang_results[lang]["total"]   += 1
                if match:
                    per_lang_results[lang]["correct_indices"].append(index)

                result_entry = {
                    "index":              index,
                    "relation":           relation,
                    "subject":            ex["subject"],
                    f"question_{src_key}": question,
                    f"{tgt_key}_question": tgt_question,    # from "translation" field (logging only)
                    "model_prediction":   prediction,         # from "answer" field ({SOURCE_LANG} {SOURCE_SCRIPT})
                    "matched_candidate":  matched_candidate,
                    "object_candidates":  object_candidates if object_candidates else None,
                    "used_candidates":    bool(object_candidates),
                    "ground_truth":       target,
                    "is_correct":         bool(match),
                    "raw_output":         raw_prediction,
                    "final_prompt":       final_prompt
                }

                detailed_results.append(result_entry)

                # ==== LIVE UPDATE ====
                progress_percent = (total_total / len(test_data)) * 100
                live_data = {
                    "progress": f"{total_total}/{len(test_data)}",
                    "percent":  f"{progress_percent:.1f}%",
                    "current_example": {
                        "index":              index,
                        "subject":            ex["subject"],
                        f"question_{src_key}": question,
                        f"{tgt_key}_question": tgt_question,
                        "model_prediction":   prediction,
                        "matched_candidate":  matched_candidate,
                        "ground_truth":       target,
                        "is_correct":         bool(match),
                        "final_prompt":       final_prompt
                    },
                    "running_accuracy": f"{(correct_total/total_total)*100:.2f}%" if total_total > 0 else "0%",
                    "results_so_far":   detailed_results[-50:]
                }

                with open(live_path, "w", encoding="utf-8") as f:
                    json.dump(live_data, f, indent=2, ensure_ascii=False)

                status = "✅" if match else "❌"
                print(f"\n{status} [{total_total}/{len(test_data)}] Subject: {ex['subject']}")
                print(f"   {SOURCE_LANG} Q:    {question}")
                print(f"   {TARGET_LANG}:      {tgt_question}")
                print(f"   Candidates:      {', '.join(object_candidates) if object_candidates else 'N/A (open gen)'}")
                print(f"   Raw prediction:  {prediction}")
                print(f"   Matched cand.:   {matched_candidate}")
                print(f"   Ground Truth:    {target}")
                print(f"   Running Acc:     {(correct_total/total_total)*100:.2f}%")

            prompts.clear()
            prompt_metadata.clear()
            pbar.update(batch_size)

    pbar.close()

    overall_acc = correct_total / total_total if total_total > 0 else 0

    results = {
        "overall_acc":        overall_acc,
        "overall_clc":        None,
        "per_language_acc":   {},
        "per_language_clc":   {}
    }

    for lang, res in sorted(per_lang_results.items()):
        lang_acc = res["correct"] / res["total"] if res["total"] > 0 else 0
        results["per_language_acc"][lang] = lang_acc

    langs = sorted(list(per_lang_results.keys()))
    for lang in langs:
        others = [l for l in langs if l != lang]
        scores = [
            overlapping_ratio(
                per_lang_results[lang]["correct_indices"],
                per_lang_results[other]["correct_indices"]
            ) for other in others
        ]
        consistency = sum(scores) / len(scores) if scores else 0
        results["per_language_clc"][lang] = consistency

    results["overall_clc"] = (
        sum(results["per_language_clc"].values()) / len(results["per_language_clc"].values())
        if results["per_language_clc"] else 0
    )

    print(f"\n📊 Overall Accuracy: {overall_acc:.2%}")
    for lang, res in sorted(per_lang_results.items()):
        lang_acc = res["correct"] / res["total"] if res["total"] > 0 else 0
        print(f"  {lang}: {lang_acc:.2%} ({res['correct']} / {res['total']})")
    print(f'\n📊 Overall CLC: {results["overall_clc"]:.2%}')

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=4, ensure_ascii=False)

    live_data["status"]           = "COMPLETED"
    live_data["final_accuracy"]   = f"{overall_acc:.2f}"
    with open(live_path, "w", encoding="utf-8") as f:
        json.dump(live_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved:")
    print(f"   Summary  → {summary_path}")
    print(f"   Detailed → {detailed_path}")
    print(f"   Live Log → {live_path}")

    return results

# Run evaluation
evaluate(llm, dataset, max_new_tokens=10, n_shot=3)



================================================
FILE: Evaluation-Scripts/1_call_pure_implicit_cm.py
================================================
import os
import json
import glob
import random
import argparse
import numpy
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
from tqdm import tqdm
import time

# ==== CONFIGURATION - SET THIS ONCE ====
OUTPUT_DIR      = "1_call_pure_implicit_cm"
DATA_DIR        = "cm_klar"
DICTIONARY_PATH = None   # e.g. "dicts/hin_eng_dict.json" or None to disable
# ======================================

# ==== Argument parser ====
parser = argparse.ArgumentParser()
parser.add_argument("--model_name",    type=str, default="meta-llama/Llama-3.1-8B")
parser.add_argument("--seed",          type=int, default=12345)
parser.add_argument("--lang_code",     type=str, default="hin")
parser.add_argument("--data_dir",      type=str, default="cm_klar")
parser.add_argument("--output_dir",    type=str, default="1_call_pure_implicit_cm")
parser.add_argument("--source_lang",   type=str, required=True)
parser.add_argument("--source_script", type=str, required=True)
parser.add_argument("--target_lang",   type=str, required=True)
args = parser.parse_args()

SOURCE_LANG   = args.source_lang
SOURCE_SCRIPT = args.source_script
TARGET_LANG   = args.target_lang


model_name = args.model_name

# ==== Set random seed ====
def set_seed(seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)



valid_langs = set(c.strip() for c in args.lang_code.split(",") if c.strip())


relations = [
    "applies_to_jurisdiction", "capital", "capital_of", "continent",
    "country_of_citizenship", "developer", "field_of_work", "headquarters_location",
    "instrument", "language_of_work_or_name", "languages_spoken", "location_of_formation",
    "manufacturer", "native_language", "occupation", "official_language",
    "owned_by", "place_of_birth", "place_of_death", "religion"
]
# valid_langs = set(languages[model_name])
valid_rels = set(relations)

# ==== Group file paths by relation ====
json_paths = glob.glob(f"{DATA_DIR}/*/*.json")
path_map = defaultdict(dict)

for path in json_paths:
    lang = os.path.basename(os.path.dirname(path))
    rel = os.path.splitext(os.path.basename(path))[0]
    if lang in valid_langs and rel in valid_rels:
        path_map[rel][lang] = path

# ==== Load samples ====
samples = []

for rel, lang_paths in path_map.items():
    if not all(lang in lang_paths for lang in valid_langs):
        continue
    for lang in valid_langs:
        with open(lang_paths[lang], "r", encoding="utf-8") as f:
            content = json.load(f)
            loaded_samples = content["samples"]
            template = content["prompt_templates"][0]
            for sample in loaded_samples:
                new_sample = {
                    "subject": sample["subject"],
                    "object": sample["object"],
                    "object_candidates": sample.get("object_candidates", None),
                    "language": lang,
                    "relation": rel,
                    "template": template,
                    "index": sample["index"]
                }
                samples.append(new_sample)

dataset = Dataset.from_list(samples)

# ==== Model loading ====
llm = LLM(
    model=model_name,
    trust_remote_code=True,
    gpu_memory_utilization=0.20,
    max_model_len=4096,
    max_num_seqs=16
)

# ==== Guided JSON schema ====
# The model outputs ONLY the answer field.
# Hinglish translation is NOT in the prompt at all — the model processes Hindi
# internally and directly outputs the answer. Fully implicit.
GUIDED_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"}
    },
    "required": ["answer"],
    "additionalProperties": False
}

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=80,
    structured_outputs=StructuredOutputsParams(json=json.dumps(GUIDED_JSON_SCHEMA))
)


# ==== Helpers ====
def parse_candidates(raw) -> list[str]:
    """
    Accept either:
      - a list   ["Boeing", "IBM", ...]
      - a string "Boeing, IBM, Google, ..."
    Returns a list of stripped candidate strings.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [c.strip() for c in raw if c.strip()]
    # string form
    return [c.strip() for c in raw.split(",") if c.strip()]


def is_nontrivial_prefix(prediction: str, target: str) -> bool:
    target = target.lower().strip()
    prediction = prediction.lower().strip()
    return len(prediction) > 0 and target.startswith(prediction)


def overlapping_ratio(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0


def match_against_candidates(prediction: str, candidates: list[str], target: str) -> tuple[bool, str]:
    """
    Try to find the best-matching candidate for the model's raw prediction.

    Strategy (in order):
      1. Exact match (case-insensitive)
      2. Candidate is a prefix of prediction  (e.g. pred="Microsoft Corp" cand="Microsoft")
      3. Prediction is a prefix of candidate  (existing nontrivial-prefix logic)
      4. Candidate appears as a substring in prediction

    Returns (is_correct, matched_candidate).
    If no candidate matches, falls back to the original prefix logic against target.
    """
    pred_low = prediction.lower().strip()

    best_cand = None
    for cand in candidates:
        cand_low = cand.lower().strip()
        if not cand_low:
            continue
        # exact
        if pred_low == cand_low:
            best_cand = cand
            break
        # candidate is prefix of prediction  ("Microsoft" in "Microsoft Corporation")
        if pred_low.startswith(cand_low):
            best_cand = cand
            break
        # prediction is prefix of candidate  ("Micro" for "Microsoft")
        if cand_low.startswith(pred_low) and len(pred_low) > 0:
            best_cand = cand
            break
        # substring
        if cand_low in pred_low:
            best_cand = cand
            break

    if best_cand is not None:
        is_correct = best_cand.lower().strip() == target.lower().strip()
        return is_correct, best_cand

    # No candidate matched — fall back to original logic against ground truth
    is_correct = is_nontrivial_prefix(prediction, target) or is_nontrivial_prefix(target, prediction)
    return is_correct, prediction


def parse_json_output(raw_text: str) -> str:
    """
    Parse the guided-JSON output from the model.
    Returns the answer string. Falls back gracefully on malformed output.
    """
    try:
        parsed = json.loads(raw_text.strip())
        answer = parsed.get("answer", "").strip()
        return answer
    except (json.JSONDecodeError, AttributeError):
        # Malformed output — return raw text as answer
        return raw_text.strip()



# ==== Evaluation ====
def evaluate(llm, dataset, max_new_tokens=10, n_shot=3):
    test_data = list(dataset)
    correct_total = 0
    total_total = 0
    per_lang_results = defaultdict(lambda: {"correct": 0, "total": 0, "correct_indices": []})
    detailed_results = []

    # ==== Build base prompt using SOURCE_LANG / SOURCE_SCRIPT / TARGET_LANG ====
    base_prompt = (
        f"You are answering factual questions written in {SOURCE_LANG} ({SOURCE_SCRIPT} script).\n\n"

        f"You MUST internally perform code-mixed transformation before answering.\n\n"

        f"INTERNAL STEP (DO NOT OUTPUT):\n"
        f"- Convert the question into {TARGET_LANG} using {SOURCE_LANG} grammar with English content words\n"
        f"- First romanize the original sentence preserving ALL words\n"
        f"- Then replace ONLY a few content words (nouns, main verbs, adjectives) with English\n"
        f"- Keep function words (question words, particles, auxiliaries, postpositions) unchanged\n"
        f"- Preserve word order exactly\n"
        f"- Do NOT rewrite or paraphrase\n\n"

        f"ILLUSTRATION (for understanding the process):\n"
        f"Input: Bharat ki rajdhani kya hai?\n"
        f"Internal form: Bharat ki capital kya hai?\n\n"

        f"RULES FOR INTERNAL TRANSFORMATION:\n"
        f"- It MUST NOT be a fluent English sentence\n"
        f"- It MUST NOT be a full translation\n"
        f"- It MUST preserve the original sentence structure\n"
        f"- At least 50% of the words should remain from the original (after romanization)\n"
        f"- Do NOT convert into known English question templates\n\n"

        f"ANSWERING:\n"
        f"- Answer based on the meaning of the question\n"
        f"- The answer MUST be consistent with the internal code-mixed interpretation\n"
        f"- The answer MUST be one of the listed candidates\n"
        f"- The answer MUST be written in {SOURCE_SCRIPT} script only\n"
        f"- Do NOT use English in the answer\n\n"

        f"OUTPUT FORMAT (STRICT):\n"
        f"Output ONLY a JSON object with a single field:\n"
        f"{{\"answer\": \"<answer in {SOURCE_SCRIPT} script>\"}}\n\n"

        f"- Do NOT output the {TARGET_LANG} translation\n"
        f"- Do NOT output any explanation\n"
    )

    model_safe_name = model_name.replace("/", "_")
    lang_name = list(valid_langs)[0]
    output_subdir = os.path.join(OUTPUT_DIR, model_safe_name, lang_name)
    os.makedirs(output_subdir, exist_ok=True)

    summary_path  = os.path.join(output_subdir, "summary.json")
    detailed_path = os.path.join(output_subdir, "detailed.json")
    live_path     = os.path.join(output_subdir, "LIVE.json")

    candidates_by_key = defaultdict(list)
    for i, ex in enumerate(test_data):
        key = (ex.get("relation"), ex.get("language"))
        candidates_by_key[key].append((i, ex))

    print(f"\n[Evaluating {len(test_data)} examples with {n_shot}-shot prompts]")
    print(f"📁 Output directory: {output_subdir}")
    print(f"💡 Tip: Run 'tail -f {live_path}' in another terminal to watch progress\n")

    prompts = []
    prompt_metadata = []
    batch_size = 8

    live_data = {
        "progress": "0/0",
        "percent": "0%",
        "current_example": None,
        "results_so_far": []
    }
    with open(live_path, "w", encoding="utf-8") as f:
        json.dump(live_data, f, indent=2, ensure_ascii=False)

    pbar = tqdm(total=len(test_data), desc="Processing", unit="ex")

    for idx, ex in enumerate(test_data):
        lang     = ex.get("language", "unknown")
        relation = ex.get("relation", "unknown")
        index    = ex.get("index", None)

        key = (relation, lang)
        candidates_pool = [c[1] for c in candidates_by_key[key] if c[0] != idx]
        demonstrations  = random.sample(candidates_pool, min(n_shot, len(candidates_pool)))

        object_candidates = parse_candidates(ex.get("object_candidates"))

        prompt = base_prompt

        for d in demonstrations:
            demo_question = d["template"].replace("<subject>", d["subject"]).replace("<mask>", "").strip()
            demo_candidates = parse_candidates(d.get("object_candidates"))
            demo_cands_str = ", ".join(demo_candidates) if demo_candidates else d["object"]

            # Hinglish is built for logging/analysis only — NOT shown in the prompt.
            # The model sees only the Hindi question and must answer implicitly.
            prompt += (
                f"Q: {demo_question}\n"
                f"Candidates: {demo_cands_str}\n"
                f"Output: {{\"answer\": \"{d['object']}\"}}\n\n"
            )

        # ---- Test question (always with candidates) ----
        test_question = ex["template"].replace("<subject>", ex["subject"]).replace("<mask>", "").strip()
        candidates_str = ", ".join(object_candidates) if object_candidates else ""

        # Only the Hindi question and candidates are shown — no Hinglish in prompt.
        prompt += (
            f"Q: {test_question}\n"
            f"Candidates: {candidates_str}\n"
            f"Output:"
        )

        prompts.append(prompt)
        # Store the pre-built hinglish alongside other metadata for logging
        prompt_metadata.append((ex, lang, index, object_candidates, prompt))

        if len(prompts) == batch_size or idx == len(test_data) - 1:
            outputs = llm.generate(prompts, sampling_params)

            for i, output in enumerate(outputs):
                ex, lang, index, object_candidates, final_prompt= prompt_metadata[i]
                raw_prediction = output.outputs[0].text

                target   = ex["object"].strip()
                question = ex["template"].replace("<subject>", ex["subject"]).replace("<mask>", "").strip()

                # ---- Parse the single JSON output (only "answer" field now) ----
                prediction = parse_json_output(raw_prediction)

                # ---- Candidate-aware matching on the Hindi answer ----
                if object_candidates:
                    match, matched_candidate = match_against_candidates(prediction, object_candidates, target)
                else:
                    match = is_nontrivial_prefix(prediction, target) or is_nontrivial_prefix(target, prediction)
                    matched_candidate = prediction

                correct_total += int(match)
                total_total   += 1
                per_lang_results[lang]["correct"] += int(match)
                per_lang_results[lang]["total"]   += 1
                if match:
                    per_lang_results[lang]["correct_indices"].append(index)

                result_entry = {
                    "index":              index,
                    "relation":           relation,
                    "subject":            ex["subject"],
                    f"question_{SOURCE_LANG.lower()}": question,
                    "model_prediction":   prediction,
                    "matched_candidate":  matched_candidate,
                    "object_candidates":  object_candidates if object_candidates else None,
                    "used_candidates":    bool(object_candidates),
                    "ground_truth":       target,
                    "is_correct":         bool(match),
                    "raw_output":         raw_prediction,
                    "final_prompt":       final_prompt
                }

                detailed_results.append(result_entry)

                # ==== LIVE UPDATE ====
                progress_percent = (total_total / len(test_data)) * 100
                live_data = {
                    "progress": f"{total_total}/{len(test_data)}",
                    "percent":  f"{progress_percent:.1f}%",
                    "current_example": {
                        "index":              index,
                        "subject":            ex["subject"],
                        f"question_{SOURCE_LANG.lower()}": question,
                        "model_prediction":   prediction,
                        "matched_candidate":  matched_candidate,
                        "ground_truth":       target,
                        "is_correct":         bool(match)
                    },
                    "running_accuracy": f"{(correct_total/total_total)*100:.2f}%" if total_total > 0 else "0%",
                    "results_so_far":   detailed_results[-50:]
                }

                with open(live_path, "w", encoding="utf-8") as f:
                    json.dump(live_data, f, indent=2, ensure_ascii=False)

                status = "✅" if match else "❌"
                print(f"\n{status} [{total_total}/{len(test_data)}] Subject: {ex['subject']}")
                # print(f"   Hindi Q:         {question}")
                print(f"   {SOURCE_LANG} Q:  {question}")
                print(f"   Candidates:      {', '.join(object_candidates) if object_candidates else 'N/A (open gen)'}")
                print(f"   Raw prediction:  {prediction}")
                print(f"   Matched cand.:   {matched_candidate}")
                print(f"   Ground Truth:    {target}")
                print(f"   Running Acc:     {(correct_total/total_total)*100:.2f}%")

            prompts.clear()
            prompt_metadata.clear()
            pbar.update(batch_size)

    pbar.close()

    overall_acc = correct_total / total_total if total_total > 0 else 0

    results = {
        "overall_acc":        overall_acc,
        "overall_clc":        None,
        "per_language_acc":   {},
        "per_language_clc":   {}
    }

    for lang, res in sorted(per_lang_results.items()):
        lang_acc = res["correct"] / res["total"] if res["total"] > 0 else 0
        results["per_language_acc"][lang] = lang_acc

    langs = sorted(list(per_lang_results.keys()))
    for lang in langs:
        others = [l for l in langs if l != lang]
        scores = [
            overlapping_ratio(
                per_lang_results[lang]["correct_indices"],
                per_lang_results[other]["correct_indices"]
            ) for other in others
        ]
        consistency = sum(scores) / len(scores) if scores else 0
        results["per_language_clc"][lang] = consistency

    results["overall_clc"] = (
        sum(results["per_language_clc"].values()) / len(results["per_language_clc"].values())
        if results["per_language_clc"] else 0
    )

    print(f"\n📊 Overall Accuracy: {overall_acc:.2%}")
    for lang, res in sorted(per_lang_results.items()):
        lang_acc = res["correct"] / res["total"] if res["total"] > 0 else 0
        print(f"  {lang}: {lang_acc:.2%} ({res['correct']} / {res['total']})")
    print(f'\n📊 Overall CLC: {results["overall_clc"]:.2%}')

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=4, ensure_ascii=False)

    live_data["status"]           = "COMPLETED"
    live_data["final_accuracy"]   = f"{overall_acc:.2f}"
    with open(live_path, "w", encoding="utf-8") as f:
        json.dump(live_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved:")
    print(f"   Summary  → {summary_path}")
    print(f"   Detailed → {detailed_path}")
    print(f"   Live Log → {live_path}")

    return results

# Run evaluation
evaluate(llm, dataset, max_new_tokens=10, n_shot=3)



================================================
FILE: Evaluation-Scripts/1_call_pure_implicit_en.py
================================================
import os
import json
import glob
import random
import argparse
import numpy
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
from tqdm import tqdm
import time

# ==== CONFIGURATION - SET THIS ONCE ====
OUTPUT_DIR      = "1_call_pure_implicit_en"
DATA_DIR        = "cm_klar"
DICTIONARY_PATH = None   # e.g. "dicts/hin_eng_dict.json" or None to disable
# ======================================

# ==== Argument parser ====
parser = argparse.ArgumentParser()
parser.add_argument("--model_name",    type=str, default="meta-llama/Llama-3.1-8B")
parser.add_argument("--seed",          type=int, default=12345)
parser.add_argument("--lang_code",     type=str, default="hin")
parser.add_argument("--data_dir",      type=str, default="cm_klar")
parser.add_argument("--output_dir",    type=str, default="1_call_pure_implicit_en")
parser.add_argument("--source_lang",   type=str, required=True)
parser.add_argument("--source_script", type=str, required=True)
parser.add_argument("--target_lang",   type=str, required=True)
args = parser.parse_args()

SOURCE_LANG   = args.source_lang
SOURCE_SCRIPT = args.source_script
TARGET_LANG   = args.target_lang


model_name = args.model_name

# ==== Set random seed ====
def set_seed(seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)




valid_langs = set(c.strip() for c in args.lang_code.split(",") if c.strip())


relations = [
    "applies_to_jurisdiction", "capital", "capital_of", "continent",
    "country_of_citizenship", "developer", "field_of_work", "headquarters_location",
    "instrument", "language_of_work_or_name", "languages_spoken", "location_of_formation",
    "manufacturer", "native_language", "occupation", "official_language",
    "owned_by", "place_of_birth", "place_of_death", "religion"
]
# valid_langs = set(languages[model_name])
valid_rels = set(relations)

# ==== Group file paths by relation ====
json_paths = glob.glob(f"{DATA_DIR}/*/*.json")
path_map = defaultdict(dict)

for path in json_paths:
    lang = os.path.basename(os.path.dirname(path))
    rel = os.path.splitext(os.path.basename(path))[0]
    if lang in valid_langs and rel in valid_rels:
        path_map[rel][lang] = path

# ==== Load samples ====
samples = []

for rel, lang_paths in path_map.items():
    if not all(lang in lang_paths for lang in valid_langs):
        continue
    for lang in valid_langs:
        with open(lang_paths[lang], "r", encoding="utf-8") as f:
            content = json.load(f)
            loaded_samples = content["samples"]
            template = content["prompt_templates"][0]
            for sample in loaded_samples:
                new_sample = {
                    "subject": sample["subject"],
                    "object": sample["object"],
                    "object_candidates": sample.get("object_candidates", None),
                    "language": lang,
                    "relation": rel,
                    "template": template,
                    "index": sample["index"]
                }
                samples.append(new_sample)

dataset = Dataset.from_list(samples)

# ==== Model loading ====
llm = LLM(
    model=model_name,
    trust_remote_code=True,
    gpu_memory_utilization=0.20,
    max_model_len=4096,
    max_num_seqs=16
)

# ==== Guided JSON schema ====
# The model outputs ONLY the answer field.
# Hinglish translation is NOT in the prompt at all — the model processes Hindi
# internally and directly outputs the answer. Fully implicit.
GUIDED_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"}
    },
    "required": ["answer"],
    "additionalProperties": False
}

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=80,
    structured_outputs=StructuredOutputsParams(json=json.dumps(GUIDED_JSON_SCHEMA))
)


# ==== Helpers ====
def parse_candidates(raw) -> list[str]:
    """
    Accept either:
      - a list   ["Boeing", "IBM", ...]
      - a string "Boeing, IBM, Google, ..."
    Returns a list of stripped candidate strings.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [c.strip() for c in raw if c.strip()]
    # string form
    return [c.strip() for c in raw.split(",") if c.strip()]


def is_nontrivial_prefix(prediction: str, target: str) -> bool:
    target = target.lower().strip()
    prediction = prediction.lower().strip()
    return len(prediction) > 0 and target.startswith(prediction)


def overlapping_ratio(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0


def match_against_candidates(prediction: str, candidates: list[str], target: str) -> tuple[bool, str]:
    """
    Try to find the best-matching candidate for the model's raw prediction.

    Strategy (in order):
      1. Exact match (case-insensitive)
      2. Candidate is a prefix of prediction  (e.g. pred="Microsoft Corp" cand="Microsoft")
      3. Prediction is a prefix of candidate  (existing nontrivial-prefix logic)
      4. Candidate appears as a substring in prediction

    Returns (is_correct, matched_candidate).
    If no candidate matches, falls back to the original prefix logic against target.
    """
    pred_low = prediction.lower().strip()

    best_cand = None
    for cand in candidates:
        cand_low = cand.lower().strip()
        if not cand_low:
            continue
        # exact
        if pred_low == cand_low:
            best_cand = cand
            break
        # candidate is prefix of prediction  ("Microsoft" in "Microsoft Corporation")
        if pred_low.startswith(cand_low):
            best_cand = cand
            break
        # prediction is prefix of candidate  ("Micro" for "Microsoft")
        if cand_low.startswith(pred_low) and len(pred_low) > 0:
            best_cand = cand
            break
        # substring
        if cand_low in pred_low:
            best_cand = cand
            break

    if best_cand is not None:
        is_correct = best_cand.lower().strip() == target.lower().strip()
        return is_correct, best_cand

    # No candidate matched — fall back to original logic against ground truth
    is_correct = is_nontrivial_prefix(prediction, target) or is_nontrivial_prefix(target, prediction)
    return is_correct, prediction


def parse_json_output(raw_text: str) -> str:
    """
    Parse the guided-JSON output from the model.
    Returns the answer string. Falls back gracefully on malformed output.
    """
    try:
        parsed = json.loads(raw_text.strip())
        answer = parsed.get("answer", "").strip()
        return answer
    except (json.JSONDecodeError, AttributeError):
        # Malformed output — return raw text as answer
        return raw_text.strip()


# ==== Evaluation ====
def evaluate(llm, dataset, max_new_tokens=10, n_shot=3):
    test_data = list(dataset)
    correct_total = 0
    total_total = 0
    per_lang_results = defaultdict(lambda: {"correct": 0, "total": 0, "correct_indices": []})
    detailed_results = []

    # ==== Build base prompt using SOURCE_LANG / SOURCE_SCRIPT / TARGET_LANG ====
    base_prompt = (
        f"You are answering factual questions written in {SOURCE_LANG} ({SOURCE_SCRIPT} script).\n\n"

        f"You MUST internally translate the question into {TARGET_LANG} to fully understand its meaning.\n"
        f"The translation must preserve the exact factual meaning of the question.\n\n"

        "IMPORTANT:\n"
        f"- This translation is ONLY for internal reasoning\n"
        f"- Do NOT output the {TARGET_LANG} translation under any circumstances\n\n"

        "You MUST then select the correct answer based on this understanding.\n\n"

        "Output MUST be a JSON object with EXACTLY one field:\n"
        "  \"answer\": the correct answer chosen from the provided candidates\n\n"

        "STRICT RULES:\n"
        "- The answer MUST be chosen EXACTLY from the candidate list\n"
        f"- The answer MUST be written in {SOURCE_LANG} ({SOURCE_SCRIPT} script) only\n"
        f"- Do NOT use {TARGET_LANG} in the answer\n"
        "- Do NOT output the translation\n"
        "- Do NOT include any explanation\n"
        "- Do NOT include any extra fields\n\n"

        "FORMAT REQUIREMENT:\n"
        "{\"answer\": \"<candidate>\"}\n\n"
    )

    model_safe_name = model_name.replace("/", "_")
    lang_name = list(valid_langs)[0]
    output_subdir = os.path.join(OUTPUT_DIR, model_safe_name, lang_name)
    os.makedirs(output_subdir, exist_ok=True)

    summary_path  = os.path.join(output_subdir, "summary.json")
    detailed_path = os.path.join(output_subdir, "detailed.json")
    live_path     = os.path.join(output_subdir, "LIVE.json")

    candidates_by_key = defaultdict(list)
    for i, ex in enumerate(test_data):
        key = (ex.get("relation"), ex.get("language"))
        candidates_by_key[key].append((i, ex))

    print(f"\n[Evaluating {len(test_data)} examples with {n_shot}-shot prompts]")
    print(f"📁 Output directory: {output_subdir}")
    print(f"💡 Tip: Run 'tail -f {live_path}' in another terminal to watch progress\n")

    prompts = []
    prompt_metadata = []
    batch_size = 8

    live_data = {
        "progress": "0/0",
        "percent": "0%",
        "current_example": None,
        "results_so_far": []
    }
    with open(live_path, "w", encoding="utf-8") as f:
        json.dump(live_data, f, indent=2, ensure_ascii=False)

    pbar = tqdm(total=len(test_data), desc="Processing", unit="ex")

    for idx, ex in enumerate(test_data):
        lang     = ex.get("language", "unknown")
        relation = ex.get("relation", "unknown")
        index    = ex.get("index", None)

        key = (relation, lang)
        candidates_pool = [c[1] for c in candidates_by_key[key] if c[0] != idx]
        demonstrations  = random.sample(candidates_pool, min(n_shot, len(candidates_pool)))

        object_candidates = parse_candidates(ex.get("object_candidates"))

        prompt = base_prompt

        for d in demonstrations:
            demo_question = d["template"].replace("<subject>", d["subject"]).replace("<mask>", "").strip()
            demo_candidates = parse_candidates(d.get("object_candidates"))
            demo_cands_str = ", ".join(demo_candidates) if demo_candidates else d["object"]

            # Hinglish is built for logging/analysis only — NOT shown in the prompt.
            # The model sees only the Hindi question and must answer implicitly.
            prompt += (
                f"Q: {demo_question}\n"
                f"Candidates: {demo_cands_str}\n"
                f"Output: {{\"answer\": \"{d['object']}\"}}\n\n"
            )

        # ---- Test question (always with candidates) ----
        test_question = ex["template"].replace("<subject>", ex["subject"]).replace("<mask>", "").strip()
        candidates_str = ", ".join(object_candidates) if object_candidates else ""

        # Only the Hindi question and candidates are shown — no Hinglish in prompt.
        prompt += (
            f"Q: {test_question}\n"
            f"Candidates: {candidates_str}\n"
            f"Output:"
        )

        prompts.append(prompt)
        # Store the pre-built hinglish alongside other metadata for logging
        prompt_metadata.append((ex, lang, index, object_candidates, prompt))

        if len(prompts) == batch_size or idx == len(test_data) - 1:
            outputs = llm.generate(prompts, sampling_params)

            for i, output in enumerate(outputs):
                ex, lang, index, object_candidates, final_prompt= prompt_metadata[i]
                raw_prediction = output.outputs[0].text

                target   = ex["object"].strip()
                question = ex["template"].replace("<subject>", ex["subject"]).replace("<mask>", "").strip()

                # ---- Parse the single JSON output (only "answer" field now) ----
                prediction = parse_json_output(raw_prediction)

                # ---- Candidate-aware matching on the Hindi answer ----
                if object_candidates:
                    match, matched_candidate = match_against_candidates(prediction, object_candidates, target)
                else:
                    match = is_nontrivial_prefix(prediction, target) or is_nontrivial_prefix(target, prediction)
                    matched_candidate = prediction

                correct_total += int(match)
                total_total   += 1
                per_lang_results[lang]["correct"] += int(match)
                per_lang_results[lang]["total"]   += 1
                if match:
                    per_lang_results[lang]["correct_indices"].append(index)

                result_entry = {
                    "index":              index,
                    "relation":           relation,
                    "subject":            ex["subject"],
                    f"question_{SOURCE_LANG.lower()}": question,
                    "model_prediction":   prediction,
                    "matched_candidate":  matched_candidate,
                    "object_candidates":  object_candidates if object_candidates else None,
                    "used_candidates":    bool(object_candidates),
                    "ground_truth":       target,
                    "is_correct":         bool(match),
                    "raw_output":         raw_prediction,
                    "final_prompt":       final_prompt
                }

                detailed_results.append(result_entry)

                # ==== LIVE UPDATE ====
                progress_percent = (total_total / len(test_data)) * 100
                live_data = {
                    "progress": f"{total_total}/{len(test_data)}",
                    "percent":  f"{progress_percent:.1f}%",
                    "current_example": {
                        "index":              index,
                        "subject":            ex["subject"],
                        f"question_{SOURCE_LANG.lower()}": question,
                        "model_prediction":   prediction,
                        "matched_candidate":  matched_candidate,
                        "ground_truth":       target,
                        "is_correct":         bool(match)
                    },
                    "running_accuracy": f"{(correct_total/total_total)*100:.2f}%" if total_total > 0 else "0%",
                    "results_so_far":   detailed_results[-50:]
                }

                with open(live_path, "w", encoding="utf-8") as f:
                    json.dump(live_data, f, indent=2, ensure_ascii=False)

                status = "✅" if match else "❌"
                print(f"\n{status} [{total_total}/{len(test_data)}] Subject: {ex['subject']}")
                # print(f"   Hindi Q:         {question}")
                print(f"   {SOURCE_LANG} Q:  {question}")
                print(f"   Candidates:      {', '.join(object_candidates) if object_candidates else 'N/A (open gen)'}")
                print(f"   Raw prediction:  {prediction}")
                print(f"   Matched cand.:   {matched_candidate}")
                print(f"   Ground Truth:    {target}")
                print(f"   Running Acc:     {(correct_total/total_total)*100:.2f}%")

            prompts.clear()
            prompt_metadata.clear()
            pbar.update(batch_size)

    pbar.close()

    overall_acc = correct_total / total_total if total_total > 0 else 0

    results = {
        "overall_acc":        overall_acc,
        "overall_clc":        None,
        "per_language_acc":   {},
        "per_language_clc":   {}
    }

    for lang, res in sorted(per_lang_results.items()):
        lang_acc = res["correct"] / res["total"] if res["total"] > 0 else 0
        results["per_language_acc"][lang] = lang_acc

    langs = sorted(list(per_lang_results.keys()))
    for lang in langs:
        others = [l for l in langs if l != lang]
        scores = [
            overlapping_ratio(
                per_lang_results[lang]["correct_indices"],
                per_lang_results[other]["correct_indices"]
            ) for other in others
        ]
        consistency = sum(scores) / len(scores) if scores else 0
        results["per_language_clc"][lang] = consistency

    results["overall_clc"] = (
        sum(results["per_language_clc"].values()) / len(results["per_language_clc"].values())
        if results["per_language_clc"] else 0
    )

    print(f"\n📊 Overall Accuracy: {overall_acc:.2%}")
    for lang, res in sorted(per_lang_results.items()):
        lang_acc = res["correct"] / res["total"] if res["total"] > 0 else 0
        print(f"  {lang}: {lang_acc:.2%} ({res['correct']} / {res['total']})")
    print(f'\n📊 Overall CLC: {results["overall_clc"]:.2%}')

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=4, ensure_ascii=False)

    live_data["status"]           = "COMPLETED"
    live_data["final_accuracy"]   = f"{overall_acc:.2f}"
    with open(live_path, "w", encoding="utf-8") as f:
        json.dump(live_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved:")
    print(f"   Summary  → {summary_path}")
    print(f"   Detailed → {detailed_path}")
    print(f"   Live Log → {live_path}")

    return results

# Run evaluation
evaluate(llm, dataset, max_new_tokens=10, n_shot=3)



================================================
FILE: Evaluation-Scripts/2_call_cm_placeholder_correct.py
================================================
import os
import json
import glob
import random
import argparse
import numpy
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
from tqdm import tqdm
import time


def load_language_config(lang_code: str) -> dict:
    config_path = f"configs/lang/{lang_code}.json"

    if not os.path.exists(config_path):
        raise ValueError(f"No config found for language: {lang_code}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ==== CONFIGURATION - SET THIS ONCE ====
OUTPUT_DIR      = "2_call_cm_placeholder_corr_8"
DATA_DIR        = "cm_klar"
DICTIONARY_PATH = None   # e.g. "dicts/hin_eng_dict.json" or None to disable
# ======================================

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, default="google/gemma-7b")
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--lang_code", type=str, default="hin")
parser.add_argument("--data_dir", type=str, default="cm_klar")
parser.add_argument("--output_dir", type=str, default="filter_knowns_implicit-trans")

# ✅ MOVE HERE
parser.add_argument("--source_lang", type=str, required=True)
parser.add_argument("--source_script", type=str, required=True)
parser.add_argument("--target_lang", type=str, required=True)

args = parser.parse_args()

SOURCE_LANG   = args.source_lang
SOURCE_SCRIPT = args.source_script
TARGET_LANG   = args.target_lang

model_name = args.model_name

# ==== Set random seed ====
def set_seed(seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)



relations = [
    "applies_to_jurisdiction", "capital", "capital_of", "continent",
    "country_of_citizenship", "developer", "field_of_work", "headquarters_location",
    "instrument", "language_of_work_or_name", "languages_spoken", "location_of_formation",
    "manufacturer", "native_language", "occupation", "official_language",
    "owned_by", "place_of_birth", "place_of_death", "religion"
]

# valid_langs = set(languages[model_name])

valid_langs = set(c.strip() for c in args.lang_code.split(",") if c.strip())
valid_rels  = set(relations)

# ==== Language config lookup ====
# Maps lang code → source_lang, source_script, target_lang.
# To add a new language: add one entry here, nothing else changes.
lang_code     = list(valid_langs)[0]           # e.g. "hin", "ben", "mal"
src_key       = SOURCE_LANG.lower()            # → result field: "question_hindi"
tgt_key       = TARGET_LANG.lower()            # → result field: "hinglish_question"

print(f"[Config] model={model_name}  lang={lang_code}  {SOURCE_LANG} → {TARGET_LANG}")
print(f"[Config] DATA_DIR={DATA_DIR}  OUTPUT_DIR={OUTPUT_DIR}")


# ==== Load CM dictionary (optional lexicon hints) ====
cm_dictionary = {}
if DICTIONARY_PATH and os.path.exists(DICTIONARY_PATH):
    with open(DICTIONARY_PATH, "r", encoding="utf-8") as f:
        cm_dictionary = json.load(f)
    print(f"[Dictionary] Loaded {len(cm_dictionary)} entries from {DICTIONARY_PATH}")
else:
    print(f"[Dictionary] None loaded — proceeding without lexicon hints")

# ==== Group file paths by relation ====
json_paths = glob.glob(f"{DATA_DIR}/*/*.json")
path_map   = defaultdict(dict)

for path in json_paths:
    lang = os.path.basename(os.path.dirname(path))
    rel  = os.path.splitext(os.path.basename(path))[0]
    if lang in valid_langs and rel in valid_rels:
        path_map[rel][lang] = path

# ==== Load samples ====
samples = []

for rel, lang_paths in path_map.items():
    if not all(lang in lang_paths for lang in valid_langs):
        continue
    for lang in valid_langs:
        with open(lang_paths[lang], "r", encoding="utf-8") as f:
            content        = json.load(f)
            loaded_samples = content["samples"]
            template       = content["prompt_templates"][0]
            for sample in loaded_samples:
                new_sample = {
                    "subject":           sample["subject"],
                    "object":            sample["object"],
                    "object_candidates": sample.get("object_candidates", None),
                    "language":          lang,
                    "relation":          rel,
                    "template":          template,
                    "index":             sample["index"]
                }
                samples.append(new_sample)

dataset = Dataset.from_list(samples)
print(f"[Dataset] Loaded {len(samples)} samples across {len(path_map)} relations")

# ==== Model loading ====
llm = LLM(
    model=model_name,
    trust_remote_code=True,
    gpu_memory_utilization=0.20,
    max_model_len=4096,
    max_num_seqs=16
)

# ==== Guided JSON schemas ====
# Stage 1: Translation only
TRANSLATION_SCHEMA = {
    "type": "object",
    "properties": {
        "translation": {"type": "string"}
    },
    "required": ["translation"],
    "additionalProperties": False
}

# Stage 2: Answer only (based on Hinglish question)
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"}
    },
    "required": ["answer"],
    "additionalProperties": False
}


# ==== Helpers ====
def parse_candidates(raw) -> list[str]:
    """
    Accept either:
      - a list   ["Boeing", "IBM", ...]
      - a string "Boeing, IBM, Google, ..."
    Returns a list of stripped candidate strings.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [c.strip() for c in raw if c.strip()]
    return [c.strip() for c in raw.split(",") if c.strip()]


def is_nontrivial_prefix(prediction: str, target: str) -> bool:
    target     = target.lower().strip()
    prediction = prediction.lower().strip()
    return len(prediction) > 0 and target.startswith(prediction)


def overlapping_ratio(list1, list2):
    set1         = set(list1)
    set2         = set(list2)
    intersection = set1.intersection(set2)
    union        = set1.union(set2)
    return len(intersection) / len(union) if union else 0


def match_against_candidates(prediction: str, candidates: list[str], target: str) -> tuple[bool, str]:
    """
    Try to find the best-matching candidate for the model's raw prediction.
    Strategy (in order):
      1. Exact match (case-insensitive)
      2. Candidate is a prefix of prediction  (e.g. pred="Microsoft Corp" cand="Microsoft")
      3. Prediction is a prefix of candidate  (existing nontrivial-prefix logic)
      4. Candidate appears as a substring in prediction
    Returns (is_correct, matched_candidate).
    If no candidate matches, falls back to the original prefix logic against target.
    """
    pred_low  = prediction.lower().strip()
    best_cand = None

    for cand in candidates:
        cand_low = cand.lower().strip()
        if not cand_low:
            continue
        if pred_low == cand_low:
            best_cand = cand; break
        if pred_low.startswith(cand_low):
            best_cand = cand; break
        if cand_low.startswith(pred_low) and len(pred_low) > 0:
            best_cand = cand; break
        if cand_low in pred_low:
            best_cand = cand; break

    if best_cand is not None:
        is_correct = best_cand.lower().strip() == target.lower().strip()
        return is_correct, best_cand

    is_correct = is_nontrivial_prefix(prediction, target) or is_nontrivial_prefix(target, prediction)
    return is_correct, prediction


def parse_translation_output(raw_text: str) -> str:
    """Parse Stage 1: Translation output."""
    try:
        parsed = json.loads(raw_text.strip())
        return parsed.get("translation", "").strip()
    except (json.JSONDecodeError, AttributeError):
        return raw_text.strip()

def parse_answer_output(raw_text: str) -> str:
    """Parse Stage 2: Answer output."""
    try:
        parsed = json.loads(raw_text.strip())
        return parsed.get("answer", "").strip()
    except (json.JSONDecodeError, AttributeError):
        return raw_text.strip()


# ==== Evaluation ====
def evaluate(llm, dataset, max_new_tokens=10, n_shot=3):
    test_data        = list(dataset)
    correct_total    = 0
    total_total      = 0
    per_lang_results = defaultdict(lambda: {"correct": 0, "total": 0, "correct_indices": []})
    detailed_results = []

    model_safe_name = model_name.replace("/", "_")
    lang_name       = list(valid_langs)[0]
    output_subdir   = os.path.join(OUTPUT_DIR, model_safe_name, lang_name)
    os.makedirs(output_subdir, exist_ok=True)

    summary_path  = os.path.join(output_subdir, "summary.json")
    detailed_path = os.path.join(output_subdir, "detailed.json")
    live_path     = os.path.join(output_subdir, "LIVE.json")

    candidates_by_key = defaultdict(list)
    for i, ex in enumerate(test_data):
        key = (ex.get("relation"), ex.get("language"))
        candidates_by_key[key].append((i, ex))


    # ==== Stage 1: Translation Prompt ====
    dict_hint_stage1 = ""
    if cm_dictionary:
        sample_entries = list(cm_dictionary.items())[:10]
        dict_hint_stage1 = (
            f"\nDictionary hint — use these {SOURCE_LANG}→English mappings:\n"
            + "\n".join(f"  {k} → {v}" for k, v in sample_entries)
            + "\n"
        )

    # stage1_base_prompt = (
    #     f"You are a translator. Convert this {SOURCE_LANG} ({SOURCE_SCRIPT} script) question "
    #     f"into {TARGET_LANG} ({SOURCE_LANG} grammar + English content words in Roman/Latin script).\n\n"
    #     f"Rules:\n"
    #     f"- Preserve {SOURCE_LANG} sentence structure and grammar markers\n"
    #     f"- Replace only content words (nouns, verbs, adjectives) with English equivalents\n"
    #     f"- Write output in Roman/Latin script only\n"
    #     f"- Output ONLY: {{\"translation\": \"...\"}}\n\n"
    #     f"{dict_hint_stage1}"
    # )
    # stage1_base_prompt = (
    #     f"You are a translator. Convert this {SOURCE_LANG} ({SOURCE_SCRIPT} script) question "
    #     f"into {TARGET_LANG} ({SOURCE_LANG} grammar + English content words in Roman/Latin script).\n\n"
    #     f"CRITICAL INSTRUCTIONS:\n"
    #     f"1. Keep ALL {SOURCE_LANG} function words (question words, postpositions, auxiliaries, particles) in Romanized form\n"
    #     f"2. Replace ONLY content words (nouns, main verbs, adjectives) with English equivalents\n"
    #     f"3. Maintain {SOURCE_LANG} word order exactly - do NOT rearrange to English word order\n"
    #     f"4. The output should read like {SOURCE_LANG} sentence structure with English vocabulary mixed in\n"
    #     f"5. Write output in Roman/Latin script only\n"
    #     f"6. Output ONLY: {{\"translation\": \"...\"}}\n\n"
    #     f"{dict_hint_stage1}"
    # )
#     stage1_base_prompt = (
#     f"You are a STRICT translator performing code-mixing.\n\n"

#     f"Convert this {SOURCE_LANG} ({SOURCE_SCRIPT} script) question into {TARGET_LANG} "
#     f"({SOURCE_LANG} grammar + English content words in Roman/Latin script).\n\n"

#     f"DEFINITION OF CODE-MIXING (FOLLOW EXACTLY):\n"
#     f"1. Keep ALL {SOURCE_LANG} function words (question words, postpositions, auxiliaries, particles) in Romanized form\n"
#     f"2. Replace ONLY content words (nouns, main verbs, adjectives) with English equivalents\n"
#     f"3. Maintain {SOURCE_LANG} word order exactly - do NOT rearrange to English word order\n"
#     f"4. The output MUST be a MIX of {SOURCE_LANG} grammar and English words (not fully English, not fully {SOURCE_LANG})\n\n"

#     f"STRICT CONSTRAINTS:\n"
#     f"- This is NOT a question-answering task\n"
#     f"- DO NOT answer the question\n"
#     f"- DO NOT add any new information\n"
#     f"- DO NOT introduce any entities (places, names, answers) not present in the input\n"
#     f"- The output MUST remain a QUESTION\n"
#     f"- Do NOT convert the question into a statement\n\n"

#     f"SCRIPT RULES:\n"
#     f"- Write output in Roman/Latin script only\n"
#     f"- Do NOT use {SOURCE_SCRIPT} script\n\n"

#     f"OUTPUT FORMAT:\n"
#     f"{{\"translation\": \"<code-mixed question>\"}}\n\n"

#     f"{dict_hint_stage1}"
# )


    # stage1_base_prompt = (
    #     f"You are a STRICT translator performing code-mixing.\n\n"

    #     f"Convert this {SOURCE_LANG} ({SOURCE_SCRIPT} script) question into {TARGET_LANG} "
    #     f"({SOURCE_LANG} grammar + English content words in Roman/Latin script).\n\n"

    #     f"DEFINITION OF CODE-MIXING (MANDATORY):\n"
    #     f"- Keep function words from {SOURCE_LANG} (romanized)\n"
    #     f"- Replace ONLY content words with English equivalents\n"
    #     f"- Preserve original word order\n\n"

    #     f"HARD CONSTRAINTS (MUST FOLLOW):\n"
    #     f"- The output MUST contain at least ONE romanized {SOURCE_LANG} word\n"
    #     f"- The output MUST contain at least ONE English word\n"
    #     f"- The output MUST NOT be fully English\n"
    #     f"- The output MUST NOT be fully {SOURCE_LANG}\n\n"

    #     f"STRICT BEHAVIOR RULES:\n"
    #     f"- This is NOT a question-answering task\n"
    #     f"- DO NOT answer the question\n"
    #     f"- DO NOT add any new information\n"
    #     f"- DO NOT introduce any entities not present in the input\n"
    #     f"- The output MUST remain a question\n\n"

    #     f"SCRIPT RULES:\n"
    #     f"- Output MUST be in Roman/Latin script only\n"
    #     f"- Do NOT use {SOURCE_SCRIPT} script\n\n"

    #     f"VALIDITY CHECK (VERY IMPORTANT):\n"
    #     f"- If the output is fully English, it is WRONG\n"
    #     f"- If the output does not contain {SOURCE_LANG} words, it is WRONG\n"
    #     f"- If unsure, modify the output to include {SOURCE_LANG} words\n\n"

    #     f"OUTPUT FORMAT:\n"
    #     f"{{\"translation\": \"<code-mixed question>\"}}\n\n"

    #     f"{dict_hint_stage1}"
    # )

    # stage1_base_prompt = (
    #     f"You are a STRICT translator performing code-mixing.\n\n"

    #     f"Convert this {SOURCE_LANG} ({SOURCE_SCRIPT} script) question into {TARGET_LANG} "
    #     f"({SOURCE_LANG} grammar + English content words in Roman/Latin script).\n\n"

    #     f"DEFINITION OF CODE-MIXING (MANDATORY):\n"
    #     f"- Keep function words from {SOURCE_LANG} (romanized)\n"
    #     f"- Replace ONLY content words with English equivalents\n"
    #     f"- Preserve original word order\n\n"

    #     f"HARD CONSTRAINTS (MUST FOLLOW):\n"
    #     f"- The output MUST contain at least ONE romanized {SOURCE_LANG} function word\n"
    #     f"- Example function words include: ki, kothay, kahan, kab, ke, er, ka (use equivalents appropriate to the language)\n"
    #     f"- The output MUST contain at least ONE English word\n"
    #     f"- The output MUST NOT be fully English\n"
    #     f"- The output MUST NOT be fully {SOURCE_LANG}\n\n"

    #     f"STRICT BEHAVIOR RULES:\n"
    #     f"- This is NOT a question-answering task\n"
    #     f"- DO NOT answer the question\n"
    #     f"- DO NOT add any new information\n"
    #     f"- DO NOT introduce any entities not present in the input\n"
    #     f"- The output MUST remain a question\n"
    #     f"- The output should look like a partial word-by-word transformation, NOT a fluent English sentence\n\n"

    #     f"SCRIPT RULES:\n"
    #     f"- Output MUST be in Roman/Latin script only\n"
    #     f"- Do NOT use {SOURCE_SCRIPT} script\n\n"

    #     f"VALIDITY CHECK (VERY IMPORTANT):\n"
    #     f"- If the output is a fully fluent English sentence, it is INVALID\n"
    #     f"- If the output does not contain any romanized {SOURCE_LANG} function word, it is INVALID\n"
    #     f"- If invalid, modify the output to include {SOURCE_LANG} function words\n\n"

    #     f"OUTPUT FORMAT:\n"
    #     f"{{\"translation\": \"<code-mixed question>\"}}\n\n"

    #     f"{dict_hint_stage1}"
    # )

    # stage1_base_prompt = (
    #     f"You are a STRICT translator performing code-mixing.\n\n"

    #     f"Convert this {SOURCE_LANG} ({SOURCE_SCRIPT} script) question into {TARGET_LANG} "
    #     f"(Roman script) while preserving the original structure.\n\n"

    #     f"DEFINITION OF CODE-MIXING (MANDATORY):\n"
    #     f"- Do NOT fully translate the sentence\n"
    #     f"- Keep parts of the original sentence structure unchanged (in romanized form)\n"
    #     f"- Replace ONLY some content words with English equivalents\n"
    #     f"- Preserve word order exactly\n\n"

    #     f"HARD CONSTRAINTS (MUST FOLLOW):\n"
    #     f"- The output MUST visibly resemble the original sentence structure\n"
    #     f"- The output MUST NOT be a fully fluent English sentence\n"
    #     f"- The output MUST be a partial transformation, not a full translation\n\n"

    #     f"STRICT BEHAVIOR RULES:\n"
    #     f"- This is NOT a question-answering task\n"
    #     f"- DO NOT answer the question\n"
    #     f"- DO NOT add any new information\n"
    #     f"- DO NOT introduce any entities not present in the input\n"
    #     f"- The output MUST remain a question\n"
    #     f"- Do NOT rewrite the sentence into natural English\n\n"

    #     f"SCRIPT RULES:\n"
    #     f"- Output MUST be in Roman/Latin script only\n"
    #     f"- Do NOT use {SOURCE_SCRIPT} script\n\n"

    #     f"VALIDITY CHECK (VERY IMPORTANT):\n"
    #     f"- If the output is a fluent English sentence, it is INVALID\n"
    #     f"- If the output does not resemble the original sentence structure, it is INVALID\n"
    #     f"- If invalid, modify it to keep parts of the original structure\n\n"
    #     f"- The output MUST reuse parts of the input question (after romanization), not generate a completely new sentence\n"
    #     f"CRITICAL STEP:\n"
    #     f"- First convert the input into a romanized version preserving the original words\n"
    #     f"- Then modify that romanized sentence by replacing some content words with English\n\n"

    #     f"OUTPUT FORMAT:\n"
    #     f"{{\"translation\": \"<code-mixed question>\"}}\n\n"

    #     f"{dict_hint_stage1}"
    # )

    # stage1_base_prompt = (
    #     f"You are a STRICT translator performing code-mixing.\n\n"

    #     f"Convert this {SOURCE_LANG} ({SOURCE_SCRIPT} script) question into {TARGET_LANG} "
    #     f"(Roman script) while preserving the original structure.\n\n"

    #     f"DEFINITION OF CODE-MIXING (MANDATORY):\n"
    #     f"- Do NOT fully translate the sentence\n"
    #     f"- Keep parts of the original sentence structure unchanged (in romanized form)\n"
    #     f"- Replace ONLY some content words with English equivalents\n"
    #     f"- Preserve word order exactly\n\n"

    #     f"CRITICAL TRANSFORMATION PROCESS:\n"
    #     f"- First convert the input into a romanized version preserving original words\n"
    #     f"- Then replace some content words (nouns, main verbs, adjectives) with English\n"
    #     f"- Keep function words (question words, particles, auxiliaries, postpositions) unchanged\n\n"

    #     f"ILLUSTRATION (Hinglish example):\n"
    #     f"Input: Bharat ki rajdhani kya hai?\n"
    #     f"Romanized: Bharat ki rajdhani kya hai?\n"
    #     f"Code-mixed: Bharat ki capital kya hai?\n\n"

    #     f"Explanation:\n"
    #     f"- 'Bharat', 'ki', 'kya', 'hai' are function words or structural elements → kept unchanged\n"
    #     f"- 'rajdhani' (a noun / content word) → replaced with 'capital'\n"
    #     f"- Word order is unchanged\n"
    #     f"- Sentence remains a question\n\n"

    #     f"Apply the SAME logic to {SOURCE_LANG}:\n"
    #     f"- Identify function words (question words, particles, auxiliaries, postpositions) → keep them (romanized)\n"
    #     f"- Identify content words (nouns, main verbs, adjectives) → translate some into English\n"
    #     f"- Preserve structure and word order exactly\n\n"

    #     f"HARD CONSTRAINTS (MUST FOLLOW):\n"
    #     f"- The output MUST visibly resemble the original sentence structure\n"
    #     f"- The output MUST NOT be a fully fluent English sentence\n"
    #     f"- The output MUST be a partial transformation, not a full translation\n\n"

    #     f"STRICT BEHAVIOR RULES:\n"
    #     f"- This is NOT a question-answering task\n"
    #     f"- DO NOT answer the question\n"
    #     f"- DO NOT add any new information\n"
    #     f"- DO NOT introduce any entities not present in the input\n"
    #     f"- The output MUST remain a question\n"
    #     f"- Do NOT rewrite the sentence into natural English\n\n"

    #     f"SCRIPT RULES:\n"
    #     f"- Output MUST be in Roman/Latin script only\n"
    #     f"- Do NOT use {SOURCE_SCRIPT} script\n\n"

    #     f"VALIDITY CHECK (VERY IMPORTANT):\n"
    #     f"- If the output is a fluent English sentence, it is INVALID\n"
    #     f"- If the output does not resemble the original sentence structure, it is INVALID\n"
    #     f"- If invalid, modify it to keep parts of the original structure\n\n"

    #     f"CRITICAL ANCHORING RULE:\n"
    #     f"- The output MUST reuse the romanized form of the input sentence\n"
    #     f"- Do NOT generate a completely new sentence\n"
    #     f"- Modify only some words, do not rewrite everything\n\n"

    #     f"OUTPUT FORMAT:\n"
    #     f"{{\"translation\": \"<code-mixed question>\"}}\n\n"

    #     f"{dict_hint_stage1}"
    # )


    stage1_base_prompt = (
        f"You are a STRICT translator performing code-mixing.\n\n"

        f"Convert this {SOURCE_LANG} ({SOURCE_SCRIPT} script) question into {TARGET_LANG} "
        f"(Roman script) while preserving the original structure.\n\n"

        f"DEFINITION OF CODE-MIXING (MANDATORY):\n"
        f"- Do NOT fully translate the sentence\n"
        f"- Keep parts of the original sentence structure unchanged (in romanized form)\n"
        f"- Replace ONLY some content words with English equivalents\n"
        f"- Preserve word order exactly\n\n"

        f"CRITICAL TRANSFORMATION PROCESS:\n"
        f"- Step 1: Convert the input into a romanized version preserving ALL words\n"
        f"- Step 2: From this romanized sentence, replace ONLY a few content words with English\n"
        f"- Step 3: Keep the rest of the words unchanged\n"
        f"- DO NOT rewrite the sentence\n"
        f"- DO NOT paraphrase\n\n"

        f"ILLUSTRATION (Hinglish example):\n"
        f"Input: Bharat ki rajdhani kya hai?\n"
        f"Romanized: Bharat ki rajdhani kya hai?\n"
        f"Code-mixed: Bharat ki capital kya hai?\n\n"

        f"Explanation:\n"
        f"- 'Bharat', 'ki', 'kya', 'hai' are function/structural words → kept unchanged\n"
        f"- 'rajdhani' (a noun / content word) → replaced with 'capital'\n"
        f"- Word order is unchanged\n"
        f"- Sentence remains a question\n\n"

        f"Apply the SAME logic to {SOURCE_LANG}:\n"
        f"- Identify function words → keep them (romanized)\n"
        f"- Identify content words → translate some into English\n"
        f"- Preserve structure and word order exactly\n\n"

        f"HARD CONSTRAINTS (MUST FOLLOW):\n"
        f"- The output MUST visibly resemble the original sentence structure\n"
        f"- The output MUST NOT be a fully fluent English sentence\n"
        f"- The output MUST be a partial transformation, not a full translation\n\n"

        f"STRICT BEHAVIOR RULES:\n"
        f"- This is NOT a question-answering task\n"
        f"- DO NOT answer the question\n"
        f"- DO NOT add any new information\n"
        f"- DO NOT introduce any entities not present in the input\n"
        f"- The output MUST remain a question\n"
        f"- Do NOT rewrite the sentence into natural English\n\n"

        f"ANTI-TEMPLATE RULE (CRITICAL):\n"
        f"- Do NOT convert the sentence into known English question patterns\n"
        f"- Do NOT rewrite into forms like 'Which country/state is X ...'\n"
        f"- Even if the meaning is clear, DO NOT restructure the sentence\n"
        f"- Preserve original phrasing and word order strictly\n\n"

        f"SCRIPT RULES:\n"
        f"- Output MUST be in Roman/Latin script only\n"
        f"- Do NOT use {SOURCE_SCRIPT} script\n\n"

        f"VALIDITY CHECK (VERY IMPORTANT):\n"
        f"- If the output is a fluent English sentence, it is INVALID\n"
        f"- If the output does not resemble the original sentence structure, it is INVALID\n"
        f"- If invalid, modify it to keep parts of the original structure\n\n"

        f"CRITICAL ANCHORING RULE:\n"
        f"- The output MUST reuse the romanized form of the input sentence\n"
        f"- Do NOT generate a completely new sentence\n"
        f"- Modify only some words, do not rewrite everything\n"
        f"- At least half of the words should remain unchanged from the romanized input\n\n"

        f"LEXICAL PRESERVATION RULE (CRITICAL):\n"
        f"- You MUST keep most words from the input sentence after romanization\n"
        f"- Do NOT replace all words with English equivalents\n"
        f"- At least 50% of the words MUST remain from the original sentence (romanized)\n"
        f"- Only replace a FEW content words with English\n\n"

        f"OUTPUT FORMAT:\n"
        f"{{\"translation\": \"<code-mixed question>\"}}\n\n"

        f"{dict_hint_stage1}"
    )

    stage2_base_prompt = (
        f"You are answering a factual question written in {TARGET_LANG}.\n"
        f"This uses {SOURCE_LANG} grammar with English vocabulary (Roman script).\n"
        f"Output ONLY: {{\"answer\": \"...\"}} in {SOURCE_SCRIPT} script.\n\n"
        "Rules:\n"
        "- Answer MUST be from the candidate list\n"
        f"- Answer MUST be in {SOURCE_SCRIPT} script\n"
        "- No English in answer\n\n"
    )

    # print(f"\n[Evaluating {len(test_data)} examples with {n_shot}-shot prompts]")
    print(f"\n[Evaluating {len(test_data)} examples]")
    print(f"📁 Output directory: {output_subdir}")
    print(f"💡 Tip: Run 'tail -f {live_path}' in another terminal to watch progress\n")



    # ==== TWO-STAGE PIPELINE SETUP ====
    stage1_prompts         = []  # Hindi → Hinglish translation prompts
    stage1_metadata        = []  # (ex, lang, index, original_question)
    stage2_prompts         = []  # Hinglish → Hindi answer prompts  
    stage2_metadata        = []  # (ex, lang, index, object_candidates, hinglish_q, original_q, stage1_prompt, stage1_raw)
    
    batch_size      = 8

    live_data = {
        "progress":        "0/0",
        "percent":         "0%",
        "current_example": None,
        "results_so_far":  []
    }
    with open(live_path, "w", encoding="utf-8") as f:
        json.dump(live_data, f, indent=2, ensure_ascii=False)

    pbar = tqdm(total=len(test_data) * 2, desc="Processing (2-Stage)", unit="call")  # *2 for 2 calls per example
    
    # ============================================
    # STAGE 1: Build all translation prompts
    # ============================================
    print(f"\n[STAGE 1/2] Building {len(test_data)} translation prompts...")
    
    for idx, ex in enumerate(test_data):
        lang     = ex.get("language", "unknown")
        relation = ex.get("relation", "unknown")
        index    = ex.get("index", None)

        # Build Stage 1 prompt (Source Lang → Target Lang translation)
        test_question = ex["template"].replace("<subject>", ex["subject"]).replace("<mask>", "").strip()
        
        s1_prompt = stage1_base_prompt
        
        # # Add few-shot examples from dataset (source language questions, like Script 2)
        # key = (relation, lang)
        # candidates_pool = [c[1] for c in candidates_by_key[key] if c[0] != idx]
        # if candidates_pool:
        #     demonstrations = random.sample(candidates_pool, min(n_shot, len(candidates_pool)))
        #     for d in demonstrations:
        #         demo_question = d["template"].replace("<subject>", d["subject"]).replace("<mask>", "").strip()
        #         s1_prompt += f"Q: {demo_question}\nOutput:\n\n"
        
        # s1_prompt += f"Q: {test_question}\nOutput:"
        s1_prompt = stage1_base_prompt + f"\nQ: {test_question}\nOutput:"

        stage1_prompts.append(s1_prompt)
        stage1_metadata.append((ex, lang, index, test_question))

    # ============================================
    # STAGE 1: Execute all translations
    # ============================================
    print(f"[STAGE 1/2] Executing {len(stage1_prompts)} translation calls...")
    
    stage1_sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=60,  # Shorter for just translation
        structured_outputs=StructuredOutputsParams(json=json.dumps(TRANSLATION_SCHEMA))
    )
    
    hinglish_translations = []  # Store results for Stage 2
    
    for i in range(0, len(stage1_prompts), batch_size):
        batch_prompts = stage1_prompts[i:i+batch_size]
        batch_meta = stage1_metadata[i:i+batch_size]
        
        outputs = llm.generate(batch_prompts, stage1_sampling_params)
        
        for j, output in enumerate(outputs):
            ex, lang, index, original_question = batch_meta[j]
            raw_output = output.outputs[0].text
            
            # Parse translation
            try:
                parsed = json.loads(raw_output.strip())
                hinglish_q = parsed.get("translation", "").strip()
            except:
                hinglish_q = raw_output.strip()
            
            hinglish_translations.append({
                "ex": ex,
                "lang": lang,
                "index": index,
                "original_question": original_question,
                "hinglish_question": hinglish_q,
                "stage1_prompt": batch_prompts[j],
                "stage1_raw": raw_output
            })
            
            pbar.update(1)

    # ============================================
    # STAGE 2: Build all QA prompts using Hinglish
    # ============================================
    print(f"\n[STAGE 2/2] Building {len(hinglish_translations)} QA prompts...")
    
    for item in hinglish_translations:
        ex = item["ex"]
        lang = item["lang"]
        index = item["index"]
        hinglish_q = item["hinglish_question"]
        original_q = item["original_question"]
        
        relation = ex.get("relation", "unknown")
        object_candidates = parse_candidates(ex.get("object_candidates"))
        candidates_str = ", ".join(object_candidates) if object_candidates else ""
        # Build Stage 2 prompt (Hinglish → Hindi answer)
        s2_prompt = stage2_base_prompt
        
        # Add few-shot examples from dataset (source language, like Script 2)
        key = (relation, lang)
        candidates_pool = [c[1] for c in candidates_by_key[key] if c[0] != index]
        if candidates_pool:
            demonstrations = random.sample(candidates_pool, min(n_shot, len(candidates_pool)))
            for d in demonstrations:
                demo_question = d["template"].replace("<subject>", d["subject"]).replace("<mask>", "").strip()
                demo_candidates = parse_candidates(d.get("object_candidates"))
                demo_cands_str = ", ".join(demo_candidates) if demo_candidates else d["object"]
                s2_prompt += f"Q: {demo_question}\n"
                s2_prompt += f"Candidates: {demo_cands_str}\n"
                s2_prompt += f"Output: {{\"answer\": \"{d['object']}\"}}\n\n"
        
        s2_prompt += f"Q: {hinglish_q}\nCandidates: {candidates_str}\nOutput:"
        stage2_prompts.append(s2_prompt)

        stage2_metadata.append((
            ex,
            lang,
            index,
            object_candidates,
            hinglish_q,
            original_q,
            item["stage1_prompt"],
            item["stage1_raw"]
        ))

    # ============================================
    # STAGE 2: Execute all QA calls
    # ============================================
    print(f"[STAGE 2/2] Executing {len(stage2_prompts)} QA calls...")
    
    stage2_sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=60,
        structured_outputs=StructuredOutputsParams(json=json.dumps(ANSWER_SCHEMA))
    )
    
    for i in range(0, len(stage2_prompts), batch_size):
        batch_prompts = stage2_prompts[i:i+batch_size]
        batch_meta = stage2_metadata[i:i+batch_size]
        
        outputs = llm.generate(batch_prompts, stage2_sampling_params)
        
        for j, output in enumerate(outputs):
            ex, lang, index, object_candidates, hinglish_q, original_q, s1_prompt, s1_raw = batch_meta[j]
            raw_prediction = output.outputs[0].text
            
            target = ex["object"].strip()
            
            # Parse answer
            try:
                parsed = json.loads(raw_prediction.strip())
                prediction = parsed.get("answer", "").strip()
            except:
                prediction = raw_prediction.strip()

            # Match against candidates
            if object_candidates:
                match, matched_candidate = match_against_candidates(prediction, object_candidates, target)
            else:
                match = is_nontrivial_prefix(prediction, target) or is_nontrivial_prefix(target, prediction)
                matched_candidate = prediction

            # Update counters
            correct_total += int(match)
            total_total += 1
            per_lang_results[lang]["correct"] += int(match)
            per_lang_results[lang]["total"] += 1
            if match:
                per_lang_results[lang]["correct_indices"].append(index)

            # # Build result entry
            result_entry = {
                "index": index,
                "relation": ex.get("relation", "unknown"),
                "subject": ex["subject"],
                f"question_{src_key}": original_q,           # Hindi question
                f"{tgt_key}_question": hinglish_q,           # Hinglish translation (from Stage 1)
                "stage1_prompt": s1_prompt,                  # Full Stage 1 prompt
                "stage1_raw_output": s1_raw,                 # Raw Stage 1 model output
                "stage2_prompt": batch_prompts[j],           # Full Stage 2 prompt
                "stage2_raw_output": raw_prediction,         # Raw Stage 2 model output
                "model_prediction": prediction,              # Final parsed answer
                "matched_candidate": matched_candidate,
                "object_candidates": object_candidates if object_candidates else None,
                "used_candidates": bool(object_candidates),
                "ground_truth": target,
                "is_correct": bool(match),
            }
            
            detailed_results.append(result_entry)
            
            # Live update and print (same as before)
            progress_percent = (total_total / len(test_data)) * 100

            live_data = {
                "progress": f"{total_total}/{len(test_data)}",
                "percent": f"{progress_percent:.1f}%",
                "current_example": {
                    "index": index,
                    "subject": ex["subject"],
                    f"question_{src_key}": original_q,
                    f"{tgt_key}_question": hinglish_q,  # Now explicitly from Stage 1
                    "model_prediction": prediction,
                    "matched_candidate": matched_candidate,
                    "ground_truth": target,
                    "is_correct": bool(match),
                    # "final_prompt"  # REMOVE THIS - replaced with stage-specific prompts below
                    "stage1_prompt": s1_prompt,      # ADD
                    "stage2_prompt": batch_prompts[j],  # ADD
                },
                "running_accuracy": f"{(correct_total/total_total)*100:.2f}%" if total_total > 0 else "0%",
                "results_so_far": detailed_results[-50:]
            }
            
            with open(live_path, "w", encoding="utf-8") as f:
                json.dump(live_data, f, indent=2, ensure_ascii=False)
            
            status = "✅" if match else "❌"
            print(f"\n{status} [{total_total}/{len(test_data)}] Subject: {ex['subject']}")
            print(f"   {SOURCE_LANG} Q:    {original_q}")
            print(f"   {TARGET_LANG}:      {hinglish_q}")
            print(f"   Candidates:        {', '.join(object_candidates) if object_candidates else 'N/A'}")
            print(f"   Raw prediction:    {prediction}")
            print(f"   Matched candidate: {matched_candidate}")
            print(f"   Ground truth:      {target}")
            print(f"   Running Acc:       {(correct_total/total_total)*100:.2f}%")
            
            pbar.update(1)

    pbar.close()

    # pbar.close()

    overall_acc = correct_total / total_total if total_total > 0 else 0


    results = {
        "overall_acc": overall_acc,
        "overall_clc": None,
        "per_language_acc": {},
        "per_language_clc": {},
        "pipeline_type": "explicit_two_stage",  # ADD THIS
        "stage1_description": f"{SOURCE_LANG} -> {TARGET_LANG} translation",
        "stage2_description": f"{TARGET_LANG} question -> {SOURCE_SCRIPT} answer",
    }

    for lang, res in sorted(per_lang_results.items()):
        lang_acc = res["correct"] / res["total"] if res["total"] > 0 else 0
        results["per_language_acc"][lang] = lang_acc

    langs = sorted(list(per_lang_results.keys()))
    for lang in langs:
        others = [l for l in langs if l != lang]
        scores = [
            overlapping_ratio(
                per_lang_results[lang]["correct_indices"],
                per_lang_results[other]["correct_indices"]
            ) for other in others
        ]
        consistency = sum(scores) / len(scores) if scores else 0
        results["per_language_clc"][lang] = consistency

    results["overall_clc"] = (
        sum(results["per_language_clc"].values()) / len(results["per_language_clc"].values())
        if results["per_language_clc"] else 0
    )

    print(f"\n📊 Overall Accuracy: {overall_acc:.2%}")
    for lang, res in sorted(per_lang_results.items()):
        lang_acc = res["correct"] / res["total"] if res["total"] > 0 else 0
        print(f"  {lang}: {lang_acc:.2%} ({res['correct']} / {res['total']})")
    print(f'\n📊 Overall CLC: {results["overall_clc"]:.2%}')

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=4, ensure_ascii=False)

    live_data["status"]         = "COMPLETED"
    live_data["final_accuracy"] = f"{overall_acc:.2f}"
    with open(live_path, "w", encoding="utf-8") as f:
        json.dump(live_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved:")
    print(f"   Summary  → {summary_path}")
    print(f"   Detailed → {detailed_path}")
    print(f"   Live Log → {live_path}")

    return results


# Run evaluation
evaluate(llm, dataset, max_new_tokens=10, n_shot=3)



================================================
FILE: Evaluation-Scripts/2_call_en_placeholder.py
================================================
import os
import json
import glob
import random
import argparse
import numpy
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
from tqdm import tqdm
import time

# ==== CONFIGURATION - SET THIS ONCE ====
OUTPUT_DIR = "2_call_en_placeholder_corr_final"
# ======================================

# ==== Argument parser ====
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B",
                    help="Model name / HF hub path, e.g. meta-llama/Llama-3.1-8B")
parser.add_argument("--lang_file", type=str, default=None,
                    help="Path to a JSON file mapping model names to language lists, "
                         "e.g. {\"meta-llama/Llama-3.1-8B\": [\"hin\", \"ben\"]}")
parser.add_argument("--seed", type=int, default=12345)
# ==== Dataset control: which directory and which language-code subdirs to load ====
parser.add_argument("--data_dir", type=str, default="cm_klar",
                    help="Root data directory to glob for JSON files, e.g. cm_klar, ben_klar")
parser.add_argument("--lang_codes", type=str, default=None,
                    help="Comma-separated language-code subdirs to load, e.g. hin,ben,ori. "
                         "Overrides --lang_file / inline languages dict for this run.")
# ==== Prompt placeholders: e.g. --source_lang Bengali --source_script Bengali --target_lang English ====
parser.add_argument("--source_lang", type=str, default="Hindi",
                    help="Source language name, e.g. Hindi, Bengali, Odia")
parser.add_argument("--source_script", type=str, default="Devanagari",
                    help="Script name for the source language, e.g. Devanagari, Bengali, Odia")
parser.add_argument("--target_lang", type=str, default="English",
                    help="Target language for the translation field, e.g. English")
args = parser.parse_args()

model_name    = args.model_name
SOURCE_LANG   = args.source_lang    # e.g. "Hindi", "Bengali"
SOURCE_SCRIPT = args.source_script  # e.g. "Devanagari", "Bengali"
TARGET_LANG   = args.target_lang    # e.g. "English"

# ==== Key slugs for result dict fields (derived from language args) ====
# e.g. SOURCE_LANG="Bengali" → src_key="bengali" → field "question_bengali"
src_key = SOURCE_LANG.lower()   # used as: f"question_{src_key}"
tgt_key = TARGET_LANG.lower()   # used as: f"{tgt_key}_question"

# ==== Set random seed ====
def set_seed(seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)

# ==== Resolve which language codes to load ====
# Priority: --lang_codes > --lang_file > inline fallback dict
if args.lang_codes:
    valid_langs = set(c.strip() for c in args.lang_codes.split(",") if c.strip())

elif args.lang_file:
    with open(args.lang_file, "r", encoding="utf-8") as _lf:
        languages = json.load(_lf)
    valid_langs = set(languages.get(model_name, []))
    if not valid_langs:
        raise ValueError(f"No languages found for model: {model_name}")

else:
    raise ValueError("You must provide either --lang_codes or --lang_file")

relations = [
    "applies_to_jurisdiction", "capital", "capital_of", "continent",
    "country_of_citizenship", "developer", "field_of_work", "headquarters_location",
    "instrument", "language_of_work_or_name", "languages_spoken", "location_of_formation",
    "manufacturer", "native_language", "occupation", "official_language",
    "owned_by", "place_of_birth", "place_of_death", "religion"
]
valid_rels = set(relations)

print(f"[Dataset] data_dir={args.data_dir}  lang_codes={sorted(valid_langs)}")

# ==== Group file paths by relation ====
json_paths = glob.glob(f"{args.data_dir}/*/*.json")
path_map = defaultdict(dict)

for path in json_paths:
    lang = os.path.basename(os.path.dirname(path))
    rel = os.path.splitext(os.path.basename(path))[0]
    if lang in valid_langs and rel in valid_rels:
        path_map[rel][lang] = path

# ==== Load samples ====
samples = []

for rel, lang_paths in path_map.items():
    if not all(lang in lang_paths for lang in valid_langs):
        continue
    for lang in valid_langs:
        with open(lang_paths[lang], "r", encoding="utf-8") as f:
            content = json.load(f)
            loaded_samples = content["samples"]
            template = content["prompt_templates"][0]
            for sample in loaded_samples:
                new_sample = {
                    "subject": sample["subject"],
                    "object": sample["object"],
                    "object_candidates": sample.get("object_candidates", None),
                    "language": lang,
                    "relation": rel,
                    "template": template,
                    "index": sample["index"]
                }
                samples.append(new_sample)

dataset = Dataset.from_list(samples)

# ==== Model loading ====
llm = LLM(
    model=model_name,
    trust_remote_code=True,
    gpu_memory_utilization=0.20,
    max_model_len=4096,
    max_num_seqs=16
)

# ==== Guided JSON schemas ====
# Stage 1: Translation only (English → English, identity)
TRANSLATION_SCHEMA = {
    "type": "object",
    "properties": {
        "translation": {"type": "string"}
    },
    "required": ["translation"],
    "additionalProperties": False
}

# Stage 2: Answer only
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"}
    },
    "required": ["answer"],
    "additionalProperties": False
}



# guided_decoding = GuidedDecodingParams(json=GUIDED_JSON_SCHEMA)


# sampling_params = SamplingParams(
#     temperature=0.0,
#     max_tokens=120,
#     structured_outputs=StructuredOutputsParams(json=json.dumps(GUIDED_JSON_SCHEMA))
# )
stage1_sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=60,
    structured_outputs=StructuredOutputsParams(json=json.dumps(TRANSLATION_SCHEMA))
)

stage2_sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=60,
    structured_outputs=StructuredOutputsParams(json=json.dumps(ANSWER_SCHEMA))
)


# ==== Helpers ====
def parse_candidates(raw) -> list[str]:
    """
    Accept either:
      - a list   ["Boeing", "IBM", ...]
      - a string "Boeing, IBM, Google, ..."
    Returns a list of stripped candidate strings.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [c.strip() for c in raw if c.strip()]
    # string form
    return [c.strip() for c in raw.split(",") if c.strip()]


def is_nontrivial_prefix(prediction: str, target: str) -> bool:
    target = target.lower().strip()
    prediction = prediction.lower().strip()
    return len(prediction) > 0 and target.startswith(prediction)


def overlapping_ratio(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0


def match_against_candidates(prediction: str, candidates: list[str], target: str) -> tuple[bool, str]:
    """
    Try to find the best-matching candidate for the model's raw prediction.

    Strategy (in order):
      1. Exact match (case-insensitive)
      2. Candidate is a prefix of prediction  (e.g. pred="Microsoft Corp" cand="Microsoft")
      3. Prediction is a prefix of candidate  (existing nontrivial-prefix logic)
      4. Candidate appears as a substring in prediction

    Returns (is_correct, matched_candidate).
    If no candidate matches, falls back to the original prefix logic against target.
    """
    pred_low = prediction.lower().strip()

    best_cand = None
    for cand in candidates:
        cand_low = cand.lower().strip()
        if not cand_low:
            continue
        # exact
        if pred_low == cand_low:
            best_cand = cand
            break
        # candidate is prefix of prediction  ("Microsoft" in "Microsoft Corporation")
        if pred_low.startswith(cand_low):
            best_cand = cand
            break
        # prediction is prefix of candidate  ("Micro" for "Microsoft")
        if cand_low.startswith(pred_low) and len(pred_low) > 0:
            best_cand = cand
            break
        # substring
        if cand_low in pred_low:
            best_cand = cand
            break

    if best_cand is not None:
        is_correct = best_cand.lower().strip() == target.lower().strip()
        return is_correct, best_cand

    # No candidate matched — fall back to original logic against ground truth
    is_correct = is_nontrivial_prefix(prediction, target) or is_nontrivial_prefix(target, prediction)
    return is_correct, prediction


def parse_json_output(raw_text: str) -> tuple[str, str]:
    """
    Parse the guided-JSON output from the model.
    Returns (translation, answer). Falls back gracefully on malformed output.
    """
    try:
        parsed = json.loads(raw_text.strip())
        translation = parsed.get("translation", "").strip()
        answer      = parsed.get("answer", "").strip()
        return translation, answer
    except (json.JSONDecodeError, AttributeError):
        # Malformed output — return empty translation, raw text as answer
        return "", raw_text.strip()


# # ==== Evaluation ====
def evaluate(llm, dataset, max_new_tokens=10, n_shot=3):
    test_data = list(dataset)
    correct_total = 0
    total_total = 0
    per_lang_results = defaultdict(lambda: {"correct": 0, "total": 0, "correct_indices": []})
    detailed_results = []

    model_safe_name = model_name.replace("/", "_")
    lang_name = "_".join(sorted(valid_langs))
    output_subdir = os.path.join(OUTPUT_DIR, model_safe_name, lang_name)
    os.makedirs(output_subdir, exist_ok=True)

    summary_path  = os.path.join(output_subdir, "summary.json")
    detailed_path = os.path.join(output_subdir, "detailed.json")
    live_path     = os.path.join(output_subdir, "LIVE.json")

    candidates_by_key = defaultdict(list)
    for i, ex in enumerate(test_data):
        key = (ex.get("relation"), ex.get("language"))
        candidates_by_key[key].append((i, ex))

    print(f"\n[Evaluating {len(test_data)} examples with {n_shot}-shot prompts]")
    print(f"📁 Output directory: {output_subdir}")
    print(f"💡 Tip: Run 'tail -f {live_path}' in another terminal to watch progress\n")

    batch_size = 8

    live_data = {
        "progress": "0/0",
        "percent": "0%",
        "current_example": None,
        "results_so_far": []
    }
    with open(live_path, "w", encoding="utf-8") as f:
        json.dump(live_data, f, indent=2, ensure_ascii=False)

    pbar = tqdm(total=len(test_data) * 2, desc="Processing (2-Stage)", unit="call")

    stage1_base_prompt = (
        f"You are a translator.\n"
        f"Translate the following question from {SOURCE_LANG} ({SOURCE_SCRIPT}) into {TARGET_LANG}.\n\n"
        f"Output ONLY: {{\"translation\": \"...\"}}\n\n"
    )

    stage2_base_prompt = (
        f"You are answering a factual question.\n"
        f"The question is in {TARGET_LANG}.\n"
        f"Select the correct answer from the candidates and write it in {SOURCE_LANG} ({SOURCE_SCRIPT}).\n\n"
        f"Output ONLY: {{\"answer\": \"...\"}}\n\n"
        "Rules:\n"
        "- Answer MUST be from the candidate list\n"
        f"- Answer MUST be in {SOURCE_SCRIPT} script\n\n"
    )
    # ===== END =====

    # ============================================
    # STAGE 1: Build all translation prompts
    # ============================================
    print(f"\n[STAGE 1/2] Building {len(test_data)} translation prompts...")
    
    stage1_prompts = []
    stage1_metadata = []
    
    for idx, ex in enumerate(test_data):
        lang = ex.get("language", "unknown")
        relation = ex.get("relation", "unknown")
        index = ex.get("index", None)

        test_question = ex["template"].replace("<subject>", ex["subject"]).replace("<mask>", "").strip()
        
        # Stage 1: No few-shot needed for English→English
        s1_prompt = stage1_base_prompt
        s1_prompt += f"Q: {test_question}\nOutput:"

        stage1_prompts.append(s1_prompt)
        stage1_metadata.append((ex, lang, index, test_question))

    # ============================================
    # STAGE 1: Execute all translations
    # ============================================
    print(f"[STAGE 1/2] Executing {len(stage1_prompts)} translation calls...")
    
    english_translations = []
    
    for i in range(0, len(stage1_prompts), batch_size):
        batch_prompts = stage1_prompts[i:i+batch_size]
        batch_meta = stage1_metadata[i:i+batch_size]
        
        outputs = llm.generate(batch_prompts, stage1_sampling_params)
        
        for j, output in enumerate(outputs):
            ex, lang, index, original_question = batch_meta[j]
            raw_output = output.outputs[0].text
            
            try:
                parsed = json.loads(raw_output.strip())
                translated_q = parsed.get("translation", "").strip()
            except:
                translated_q = raw_output.strip()
            
            english_translations.append({
                "ex": ex,
                "lang": lang,
                "index": index,
                "original_question": original_question,
                "english_question": translated_q,
                "stage1_prompt": batch_prompts[j],
                "stage1_raw": raw_output
            })
            
            pbar.update(1)

    # ============================================
    # STAGE 2: Build all QA prompts
    # ============================================
    print(f"\n[STAGE 2/2] Building {len(english_translations)} QA prompts...")
    
    stage2_prompts = []
    stage2_metadata = []
    
    for idx, item in enumerate(english_translations):
        ex = item["ex"]
        lang = item["lang"]
        index = item["index"]
        english_q = item["english_question"]
        original_q = item["original_question"]
        
        relation = ex.get("relation", "unknown")
        object_candidates = parse_candidates(ex.get("object_candidates"))
        candidates_str = ", ".join(object_candidates) if object_candidates else ""

        # Stage 2: Few-shot from dataset (like Script 2)
        s2_prompt = stage2_base_prompt
        
        key = (relation, lang)
        candidates_pool = [c[1] for c in candidates_by_key[key] if c[0] != index]
        if candidates_pool:
            demonstrations = random.sample(candidates_pool, min(n_shot, len(candidates_pool)))
            for d in demonstrations:
                demo_question = d["template"].replace("<subject>", d["subject"]).replace("<mask>", "").strip()
                demo_candidates = parse_candidates(d.get("object_candidates"))
                demo_cands_str = ", ".join(demo_candidates) if demo_candidates else d["object"]
                s2_prompt += f"Q: {demo_question}\n"
                s2_prompt += f"Candidates: {demo_cands_str}\n"
                s2_prompt += f"Output: {{\"answer\": \"{d['object']}\"}}\n\n"
        
        s2_prompt += f"Q: {english_q}\nCandidates: {candidates_str}\nOutput:"

        stage2_prompts.append(s2_prompt)
        stage2_metadata.append((
            ex, lang, index, object_candidates, 
            english_q, original_q, item["stage1_prompt"], 
            item["stage1_raw"]
        ))

    # ============================================
    # STAGE 2: Execute all QA calls
    # ============================================
    print(f"[STAGE 2/2] Executing {len(stage2_prompts)} QA calls...")
    
    for i in range(0, len(stage2_prompts), batch_size):
        batch_prompts = stage2_prompts[i:i+batch_size]
        batch_meta = stage2_metadata[i:i+batch_size]
        
        outputs = llm.generate(batch_prompts, stage2_sampling_params)
        
        for j, output in enumerate(outputs):
            ex, lang, index, object_candidates, english_q, original_q, s1_prompt, s1_raw = batch_meta[j]
            raw_prediction = output.outputs[0].text
            
            target = ex["object"].strip()
            
            try:
                parsed = json.loads(raw_prediction.strip())
                prediction = parsed.get("answer", "").strip()
            except:
                prediction = raw_prediction.strip()

            if object_candidates:
                match, matched_candidate = match_against_candidates(prediction, object_candidates, target)
            else:
                match = is_nontrivial_prefix(prediction, target) or is_nontrivial_prefix(target, prediction)
                matched_candidate = prediction

            correct_total += int(match)
            total_total += 1
            per_lang_results[lang]["correct"] += int(match)
            per_lang_results[lang]["total"] += 1
            if match:
                per_lang_results[lang]["correct_indices"].append(index)

            result_entry = {
                "index": index,
                "relation": ex.get("relation", "unknown"),
                "subject": ex["subject"],
                f"question_{src_key}": original_q,
                f"{tgt_key}_question": english_q,
                "stage1_prompt": s1_prompt,
                "stage1_raw_output": s1_raw,
                "stage2_prompt": batch_prompts[j],
                "stage2_raw_output": raw_prediction,
                "model_prediction": prediction,
                "matched_candidate": matched_candidate,
                "object_candidates": object_candidates if object_candidates else None,
                "used_candidates": bool(object_candidates),
                "ground_truth": target,
                "is_correct": bool(match),
            }
            
            detailed_results.append(result_entry)
            
            progress_percent = (total_total / len(test_data)) * 100
            live_data = {
                "progress": f"{total_total}/{len(test_data)}",
                "percent": f"{progress_percent:.1f}%",
                "current_example": {
                    "index": index,
                    "subject": ex["subject"],
                    f"question_{src_key}": original_q,
                    f"{tgt_key}_question": english_q,
                    "model_prediction": prediction,
                    "matched_candidate": matched_candidate,
                    "ground_truth": target,
                    "is_correct": bool(match),
                    "stage1_prompt": s1_prompt,
                    "stage2_prompt": batch_prompts[j],
                },
                "running_accuracy": f"{(correct_total/total_total)*100:.2f}%" if total_total > 0 else "0%",
                "results_so_far": detailed_results[-50:]
            }
            
            with open(live_path, "w", encoding="utf-8") as f:
                json.dump(live_data, f, indent=2, ensure_ascii=False)
            
            status = "✅" if match else "❌"
            print(f"\n{status} [{total_total}/{len(test_data)}] Subject: {ex['subject']}")
            print(f"   {SOURCE_LANG} Q:    {original_q}")
            print(f"   {TARGET_LANG}:      {english_q}")
            print(f"   Candidates:        {', '.join(object_candidates) if object_candidates else 'N/A'}")
            print(f"   Raw prediction:    {prediction}")
            print(f"   Matched candidate: {matched_candidate}")
            print(f"   Ground truth:      {target}")
            print(f"   Running Acc:       {(correct_total/total_total)*100:.2f}%")
            
            pbar.update(1)

    pbar.close()

    overall_acc = correct_total / total_total if total_total > 0 else 0

    results = {
        "overall_acc": overall_acc,
        "overall_clc": None,
        "per_language_acc": {},
        "per_language_clc": {},
        "pipeline_type": "explicit_two_stage_en",
        "stage1_description": f"{SOURCE_LANG} -> {TARGET_LANG} translation",
        "stage2_description": f"{TARGET_LANG} question -> {SOURCE_SCRIPT} answer",
    }

    for lang, res in sorted(per_lang_results.items()):
        lang_acc = res["correct"] / res["total"] if res["total"] > 0 else 0
        results["per_language_acc"][lang] = lang_acc

    langs = sorted(list(per_lang_results.keys()))
    for lang in langs:
        others = [l for l in langs if l != lang]
        scores = [
            overlapping_ratio(
                per_lang_results[lang]["correct_indices"],
                per_lang_results[other]["correct_indices"]
            ) for other in others
        ]
        consistency = sum(scores) / len(scores) if scores else 0
        results["per_language_clc"][lang] = consistency

    results["overall_clc"] = (
        sum(results["per_language_clc"].values()) / len(results["per_language_clc"].values())
        if results["per_language_clc"] else 0
    )

    print(f"\n📊 Overall Accuracy: {overall_acc:.2%}")
    for lang, res in sorted(per_lang_results.items()):
        lang_acc = res["correct"] / res["total"] if res["total"] > 0 else 0
        print(f"  {lang}: {lang_acc:.2%} ({res['correct']} / {res['total']})")
    print(f'\n📊 Overall CLC: {results["overall_clc"]:.2%}')

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=4, ensure_ascii=False)

    live_data["status"] = "COMPLETED"
    live_data["final_accuracy"] = f"{overall_acc:.2f}"
    with open(live_path, "w", encoding="utf-8") as f:
        json.dump(live_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved:")
    print(f"   Summary  → {summary_path}")
    print(f"   Detailed → {detailed_path}")
    print(f"   Live Log → {live_path}")

    return results



# Run evaluation
evaluate(llm, dataset, max_new_tokens=10, n_shot=3)



================================================
FILE: Evaluation-Scripts/2_call_transliteration.py
================================================
import os
import json
import glob
import random
import argparse
import numpy
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
from tqdm import tqdm
import time


def load_language_config(lang_code: str) -> dict:
    config_path = f"configs/lang/{lang_code}.json"

    if not os.path.exists(config_path):
        raise ValueError(f"No config found for language: {lang_code}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ==== CONFIGURATION - SET THIS ONCE ====
OUTPUT_DIR      = "2_call_transliteration"
DATA_DIR        = "cm_klar"
DICTIONARY_PATH = None   # e.g. "dicts/hin_eng_dict.json" or None to disable
# ======================================

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, default="google/gemma-7b")
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--lang_code", type=str, default="hin")
parser.add_argument("--data_dir", type=str, default="cm_klar")
parser.add_argument("--output_dir", type=str, default="2_call_transliteration")

# ✅ MOVE HERE
parser.add_argument("--source_lang", type=str, required=True)
parser.add_argument("--source_script", type=str, required=True)
parser.add_argument("--target_lang", type=str, required=True)

args = parser.parse_args()

SOURCE_LANG   = args.source_lang
SOURCE_SCRIPT = args.source_script
TARGET_LANG   = args.target_lang

model_name = args.model_name

# ==== Set random seed ====
def set_seed(seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)



relations = [
    "applies_to_jurisdiction", "capital", "capital_of", "continent",
    "country_of_citizenship", "developer", "field_of_work", "headquarters_location",
    "instrument", "language_of_work_or_name", "languages_spoken", "location_of_formation",
    "manufacturer", "native_language", "occupation", "official_language",
    "owned_by", "place_of_birth", "place_of_death", "religion"
]

# valid_langs = set(languages[model_name])

valid_langs = set(c.strip() for c in args.lang_code.split(",") if c.strip())
valid_rels  = set(relations)

# ==== Language config lookup ====
# Maps lang code → source_lang, source_script, target_lang.
# To add a new language: add one entry here, nothing else changes.
lang_code     = list(valid_langs)[0]           # e.g. "hin", "ben", "mal"
src_key       = SOURCE_LANG.lower()            # → result field: "question_hindi"
tgt_key       = TARGET_LANG.lower()            # → result field: "hinglish_question"

print(f"[Config] model={model_name}  lang={lang_code}  {SOURCE_LANG} → {TARGET_LANG}")
print(f"[Config] DATA_DIR={DATA_DIR}  OUTPUT_DIR={OUTPUT_DIR}")


# ==== Load CM dictionary (optional lexicon hints) ====
cm_dictionary = {}
if DICTIONARY_PATH and os.path.exists(DICTIONARY_PATH):
    with open(DICTIONARY_PATH, "r", encoding="utf-8") as f:
        cm_dictionary = json.load(f)
    print(f"[Dictionary] Loaded {len(cm_dictionary)} entries from {DICTIONARY_PATH}")
else:
    print(f"[Dictionary] None loaded — proceeding without lexicon hints")

# ==== Group file paths by relation ====
json_paths = glob.glob(f"{DATA_DIR}/*/*.json")
path_map   = defaultdict(dict)

for path in json_paths:
    lang = os.path.basename(os.path.dirname(path))
    rel  = os.path.splitext(os.path.basename(path))[0]
    if lang in valid_langs and rel in valid_rels:
        path_map[rel][lang] = path

# ==== Load samples ====
samples = []

for rel, lang_paths in path_map.items():
    if not all(lang in lang_paths for lang in valid_langs):
        continue
    for lang in valid_langs:
        with open(lang_paths[lang], "r", encoding="utf-8") as f:
            content        = json.load(f)
            loaded_samples = content["samples"]
            template       = content["prompt_templates"][0]
            for sample in loaded_samples:
                new_sample = {
                    "subject":           sample["subject"],
                    "object":            sample["object"],
                    "object_candidates": sample.get("object_candidates", None),
                    "language":          lang,
                    "relation":          rel,
                    "template":          template,
                    "index":             sample["index"]
                }
                samples.append(new_sample)

dataset = Dataset.from_list(samples)
print(f"[Dataset] Loaded {len(samples)} samples across {len(path_map)} relations")

# ==== Model loading ====
llm = LLM(
    model=model_name,
    trust_remote_code=True,
    gpu_memory_utilization=0.45,
    max_model_len=4096,
    max_num_seqs= 16
)

# ==== Guided JSON schemas ====
# Stage 1: Translation only
TRANSLATION_SCHEMA = {
    "type": "object",
    "properties": {
        "translation": {"type": "string"}
    },
    "required": ["translation"],
    "additionalProperties": False
}

# Stage 2: Answer only (based on Hinglish question)
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"}
    },
    "required": ["answer"],
    "additionalProperties": False
}


# ==== Helpers ====
def parse_candidates(raw) -> list[str]:
    """
    Accept either:
      - a list   ["Boeing", "IBM", ...]
      - a string "Boeing, IBM, Google, ..."
    Returns a list of stripped candidate strings.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [c.strip() for c in raw if c.strip()]
    return [c.strip() for c in raw.split(",") if c.strip()]


def is_nontrivial_prefix(prediction: str, target: str) -> bool:
    target     = target.lower().strip()
    prediction = prediction.lower().strip()
    return len(prediction) > 0 and target.startswith(prediction)


def overlapping_ratio(list1, list2):
    set1         = set(list1)
    set2         = set(list2)
    intersection = set1.intersection(set2)
    union        = set1.union(set2)
    return len(intersection) / len(union) if union else 0


def match_against_candidates(prediction: str, candidates: list[str], target: str) -> tuple[bool, str]:
    """
    Try to find the best-matching candidate for the model's raw prediction.
    Strategy (in order):
      1. Exact match (case-insensitive)
      2. Candidate is a prefix of prediction  (e.g. pred="Microsoft Corp" cand="Microsoft")
      3. Prediction is a prefix of candidate  (existing nontrivial-prefix logic)
      4. Candidate appears as a substring in prediction
    Returns (is_correct, matched_candidate).
    If no candidate matches, falls back to the original prefix logic against target.
    """
    pred_low  = prediction.lower().strip()
    best_cand = None

    for cand in candidates:
        cand_low = cand.lower().strip()
        if not cand_low:
            continue
        if pred_low == cand_low:
            best_cand = cand; break
        if pred_low.startswith(cand_low):
            best_cand = cand; break
        if cand_low.startswith(pred_low) and len(pred_low) > 0:
            best_cand = cand; break
        if cand_low in pred_low:
            best_cand = cand; break

    if best_cand is not None:
        is_correct = best_cand.lower().strip() == target.lower().strip()
        return is_correct, best_cand

    is_correct = is_nontrivial_prefix(prediction, target) or is_nontrivial_prefix(target, prediction)
    return is_correct, prediction


def parse_translation_output(raw_text: str) -> str:
    """Parse Stage 1: Translation output."""
    try:
        parsed = json.loads(raw_text.strip())
        return parsed.get("translation", "").strip()
    except (json.JSONDecodeError, AttributeError):
        return raw_text.strip()

def parse_answer_output(raw_text: str) -> str:
    """Parse Stage 2: Answer output."""
    try:
        parsed = json.loads(raw_text.strip())
        return parsed.get("answer", "").strip()
    except (json.JSONDecodeError, AttributeError):
        return raw_text.strip()


# ==== Evaluation ====
def evaluate(llm, dataset, max_new_tokens=10, n_shot=3):
    test_data        = list(dataset)
    correct_total    = 0
    total_total      = 0
    per_lang_results = defaultdict(lambda: {"correct": 0, "total": 0, "correct_indices": []})
    detailed_results = []

    model_safe_name = model_name.replace("/", "_")
    lang_name       = list(valid_langs)[0]
    output_subdir   = os.path.join(OUTPUT_DIR, model_safe_name, lang_name)
    os.makedirs(output_subdir, exist_ok=True)

    summary_path  = os.path.join(output_subdir, "summary.json")
    detailed_path = os.path.join(output_subdir, "detailed.json")
    live_path     = os.path.join(output_subdir, "LIVE.json")

    candidates_by_key = defaultdict(list)
    for i, ex in enumerate(test_data):
        key = (ex.get("relation"), ex.get("language"))
        candidates_by_key[key].append((i, ex))


    # ==== Stage 1: Translation Prompt ====
    dict_hint_stage1 = ""
    if cm_dictionary:
        sample_entries = list(cm_dictionary.items())[:10]
        dict_hint_stage1 = (
            f"\nDictionary hint — use these {SOURCE_LANG}→English mappings:\n"
            + "\n".join(f"  {k} → {v}" for k, v in sample_entries)
            + "\n"
        )

    # stage1_base_prompt = (
    #     f"You are a translator. Convert this {SOURCE_LANG} ({SOURCE_SCRIPT} script) question "
    #     f"into {TARGET_LANG} ({SOURCE_LANG} grammar + English content words in Roman/Latin script).\n\n"
    #     f"Rules:\n"
    #     f"- Preserve {SOURCE_LANG} sentence structure and grammar markers\n"
    #     f"- Replace only content words (nouns, verbs, adjectives) with English equivalents\n"
    #     f"- Write output in Roman/Latin script only\n"
    #     f"- Output ONLY: {{\"translation\": \"...\"}}\n\n"
    #     f"{dict_hint_stage1}"
    # )
    # stage1_base_prompt = (
    #     f"You are a translator. Convert this {SOURCE_LANG} ({SOURCE_SCRIPT} script) question "
    #     f"into {TARGET_LANG} ({SOURCE_LANG} grammar + English content words in Roman/Latin script).\n\n"
    #     f"CRITICAL INSTRUCTIONS:\n"
    #     f"1. Keep ALL {SOURCE_LANG} function words (question words, postpositions, auxiliaries, particles) in Romanized form\n"
    #     f"2. Replace ONLY content words (nouns, main verbs, adjectives) with English equivalents\n"
    #     f"3. Maintain {SOURCE_LANG} word order exactly - do NOT rearrange to English word order\n"
    #     f"4. The output should read like {SOURCE_LANG} sentence structure with English vocabulary mixed in\n"
    #     f"5. Write output in Roman/Latin script only\n"
    #     f"6. Output ONLY: {{\"translation\": \"...\"}}\n\n"
    #     f"{dict_hint_stage1}"
    # )
#     stage1_base_prompt = (
#     f"You are a STRICT translator performing code-mixing.\n\n"

#     f"Convert this {SOURCE_LANG} ({SOURCE_SCRIPT} script) question into {TARGET_LANG} "
#     f"({SOURCE_LANG} grammar + English content words in Roman/Latin script).\n\n"

#     f"DEFINITION OF CODE-MIXING (FOLLOW EXACTLY):\n"
#     f"1. Keep ALL {SOURCE_LANG} function words (question words, postpositions, auxiliaries, particles) in Romanized form\n"
#     f"2. Replace ONLY content words (nouns, main verbs, adjectives) with English equivalents\n"
#     f"3. Maintain {SOURCE_LANG} word order exactly - do NOT rearrange to English word order\n"
#     f"4. The output MUST be a MIX of {SOURCE_LANG} grammar and English words (not fully English, not fully {SOURCE_LANG})\n\n"

#     f"STRICT CONSTRAINTS:\n"
#     f"- This is NOT a question-answering task\n"
#     f"- DO NOT answer the question\n"
#     f"- DO NOT add any new information\n"
#     f"- DO NOT introduce any entities (places, names, answers) not present in the input\n"
#     f"- The output MUST remain a QUESTION\n"
#     f"- Do NOT convert the question into a statement\n\n"

#     f"SCRIPT RULES:\n"
#     f"- Write output in Roman/Latin script only\n"
#     f"- Do NOT use {SOURCE_SCRIPT} script\n\n"

#     f"OUTPUT FORMAT:\n"
#     f"{{\"translation\": \"<code-mixed question>\"}}\n\n"

#     f"{dict_hint_stage1}"
# )


    # stage1_base_prompt = (
    #     f"You are a STRICT translator performing code-mixing.\n\n"

    #     f"Convert this {SOURCE_LANG} ({SOURCE_SCRIPT} script) question into {TARGET_LANG} "
    #     f"({SOURCE_LANG} grammar + English content words in Roman/Latin script).\n\n"

    #     f"DEFINITION OF CODE-MIXING (MANDATORY):\n"
    #     f"- Keep function words from {SOURCE_LANG} (romanized)\n"
    #     f"- Replace ONLY content words with English equivalents\n"
    #     f"- Preserve original word order\n\n"

    #     f"HARD CONSTRAINTS (MUST FOLLOW):\n"
    #     f"- The output MUST contain at least ONE romanized {SOURCE_LANG} word\n"
    #     f"- The output MUST contain at least ONE English word\n"
    #     f"- The output MUST NOT be fully English\n"
    #     f"- The output MUST NOT be fully {SOURCE_LANG}\n\n"

    #     f"STRICT BEHAVIOR RULES:\n"
    #     f"- This is NOT a question-answering task\n"
    #     f"- DO NOT answer the question\n"
    #     f"- DO NOT add any new information\n"
    #     f"- DO NOT introduce any entities not present in the input\n"
    #     f"- The output MUST remain a question\n\n"

    #     f"SCRIPT RULES:\n"
    #     f"- Output MUST be in Roman/Latin script only\n"
    #     f"- Do NOT use {SOURCE_SCRIPT} script\n\n"

    #     f"VALIDITY CHECK (VERY IMPORTANT):\n"
    #     f"- If the output is fully English, it is WRONG\n"
    #     f"- If the output does not contain {SOURCE_LANG} words, it is WRONG\n"
    #     f"- If unsure, modify the output to include {SOURCE_LANG} words\n\n"

    #     f"OUTPUT FORMAT:\n"
    #     f"{{\"translation\": \"<code-mixed question>\"}}\n\n"

    #     f"{dict_hint_stage1}"
    # )

    # stage1_base_prompt = (
    #     f"You are a STRICT translator performing code-mixing.\n\n"

    #     f"Convert this {SOURCE_LANG} ({SOURCE_SCRIPT} script) question into {TARGET_LANG} "
    #     f"({SOURCE_LANG} grammar + English content words in Roman/Latin script).\n\n"

    #     f"DEFINITION OF CODE-MIXING (MANDATORY):\n"
    #     f"- Keep function words from {SOURCE_LANG} (romanized)\n"
    #     f"- Replace ONLY content words with English equivalents\n"
    #     f"- Preserve original word order\n\n"

    #     f"HARD CONSTRAINTS (MUST FOLLOW):\n"
    #     f"- The output MUST contain at least ONE romanized {SOURCE_LANG} function word\n"
    #     f"- Example function words include: ki, kothay, kahan, kab, ke, er, ka (use equivalents appropriate to the language)\n"
    #     f"- The output MUST contain at least ONE English word\n"
    #     f"- The output MUST NOT be fully English\n"
    #     f"- The output MUST NOT be fully {SOURCE_LANG}\n\n"

    #     f"STRICT BEHAVIOR RULES:\n"
    #     f"- This is NOT a question-answering task\n"
    #     f"- DO NOT answer the question\n"
    #     f"- DO NOT add any new information\n"
    #     f"- DO NOT introduce any entities not present in the input\n"
    #     f"- The output MUST remain a question\n"
    #     f"- The output should look like a partial word-by-word transformation, NOT a fluent English sentence\n\n"

    #     f"SCRIPT RULES:\n"
    #     f"- Output MUST be in Roman/Latin script only\n"
    #     f"- Do NOT use {SOURCE_SCRIPT} script\n\n"

    #     f"VALIDITY CHECK (VERY IMPORTANT):\n"
    #     f"- If the output is a fully fluent English sentence, it is INVALID\n"
    #     f"- If the output does not contain any romanized {SOURCE_LANG} function word, it is INVALID\n"
    #     f"- If invalid, modify the output to include {SOURCE_LANG} function words\n\n"

    #     f"OUTPUT FORMAT:\n"
    #     f"{{\"translation\": \"<code-mixed question>\"}}\n\n"

    #     f"{dict_hint_stage1}"
    # )

    # stage1_base_prompt = (
    #     f"You are a STRICT translator performing code-mixing.\n\n"

    #     f"Convert this {SOURCE_LANG} ({SOURCE_SCRIPT} script) question into {TARGET_LANG} "
    #     f"(Roman script) while preserving the original structure.\n\n"

    #     f"DEFINITION OF CODE-MIXING (MANDATORY):\n"
    #     f"- Do NOT fully translate the sentence\n"
    #     f"- Keep parts of the original sentence structure unchanged (in romanized form)\n"
    #     f"- Replace ONLY some content words with English equivalents\n"
    #     f"- Preserve word order exactly\n\n"

    #     f"HARD CONSTRAINTS (MUST FOLLOW):\n"
    #     f"- The output MUST visibly resemble the original sentence structure\n"
    #     f"- The output MUST NOT be a fully fluent English sentence\n"
    #     f"- The output MUST be a partial transformation, not a full translation\n\n"

    #     f"STRICT BEHAVIOR RULES:\n"
    #     f"- This is NOT a question-answering task\n"
    #     f"- DO NOT answer the question\n"
    #     f"- DO NOT add any new information\n"
    #     f"- DO NOT introduce any entities not present in the input\n"
    #     f"- The output MUST remain a question\n"
    #     f"- Do NOT rewrite the sentence into natural English\n\n"

    #     f"SCRIPT RULES:\n"
    #     f"- Output MUST be in Roman/Latin script only\n"
    #     f"- Do NOT use {SOURCE_SCRIPT} script\n\n"

    #     f"VALIDITY CHECK (VERY IMPORTANT):\n"
    #     f"- If the output is a fluent English sentence, it is INVALID\n"
    #     f"- If the output does not resemble the original sentence structure, it is INVALID\n"
    #     f"- If invalid, modify it to keep parts of the original structure\n\n"
    #     f"- The output MUST reuse parts of the input question (after romanization), not generate a completely new sentence\n"
    #     f"CRITICAL STEP:\n"
    #     f"- First convert the input into a romanized version preserving the original words\n"
    #     f"- Then modify that romanized sentence by replacing some content words with English\n\n"

    #     f"OUTPUT FORMAT:\n"
    #     f"{{\"translation\": \"<code-mixed question>\"}}\n\n"

    #     f"{dict_hint_stage1}"
    # )

    # stage1_base_prompt = (
    #     f"You are a STRICT translator performing code-mixing.\n\n"

    #     f"Convert this {SOURCE_LANG} ({SOURCE_SCRIPT} script) question into {TARGET_LANG} "
    #     f"(Roman script) while preserving the original structure.\n\n"

    #     f"DEFINITION OF CODE-MIXING (MANDATORY):\n"
    #     f"- Do NOT fully translate the sentence\n"
    #     f"- Keep parts of the original sentence structure unchanged (in romanized form)\n"
    #     f"- Replace ONLY some content words with English equivalents\n"
    #     f"- Preserve word order exactly\n\n"

    #     f"CRITICAL TRANSFORMATION PROCESS:\n"
    #     f"- First convert the input into a romanized version preserving original words\n"
    #     f"- Then replace some content words (nouns, main verbs, adjectives) with English\n"
    #     f"- Keep function words (question words, particles, auxiliaries, postpositions) unchanged\n\n"

    #     f"ILLUSTRATION (Hinglish example):\n"
    #     f"Input: Bharat ki rajdhani kya hai?\n"
    #     f"Romanized: Bharat ki rajdhani kya hai?\n"
    #     f"Code-mixed: Bharat ki capital kya hai?\n\n"

    #     f"Explanation:\n"
    #     f"- 'Bharat', 'ki', 'kya', 'hai' are function words or structural elements → kept unchanged\n"
    #     f"- 'rajdhani' (a noun / content word) → replaced with 'capital'\n"
    #     f"- Word order is unchanged\n"
    #     f"- Sentence remains a question\n\n"

    #     f"Apply the SAME logic to {SOURCE_LANG}:\n"
    #     f"- Identify function words (question words, particles, auxiliaries, postpositions) → keep them (romanized)\n"
    #     f"- Identify content words (nouns, main verbs, adjectives) → translate some into English\n"
    #     f"- Preserve structure and word order exactly\n\n"

    #     f"HARD CONSTRAINTS (MUST FOLLOW):\n"
    #     f"- The output MUST visibly resemble the original sentence structure\n"
    #     f"- The output MUST NOT be a fully fluent English sentence\n"
    #     f"- The output MUST be a partial transformation, not a full translation\n\n"

    #     f"STRICT BEHAVIOR RULES:\n"
    #     f"- This is NOT a question-answering task\n"
    #     f"- DO NOT answer the question\n"
    #     f"- DO NOT add any new information\n"
    #     f"- DO NOT introduce any entities not present in the input\n"
    #     f"- The output MUST remain a question\n"
    #     f"- Do NOT rewrite the sentence into natural English\n\n"

    #     f"SCRIPT RULES:\n"
    #     f"- Output MUST be in Roman/Latin script only\n"
    #     f"- Do NOT use {SOURCE_SCRIPT} script\n\n"

    #     f"VALIDITY CHECK (VERY IMPORTANT):\n"
    #     f"- If the output is a fluent English sentence, it is INVALID\n"
    #     f"- If the output does not resemble the original sentence structure, it is INVALID\n"
    #     f"- If invalid, modify it to keep parts of the original structure\n\n"

    #     f"CRITICAL ANCHORING RULE:\n"
    #     f"- The output MUST reuse the romanized form of the input sentence\n"
    #     f"- Do NOT generate a completely new sentence\n"
    #     f"- Modify only some words, do not rewrite everything\n\n"

    #     f"OUTPUT FORMAT:\n"
    #     f"{{\"translation\": \"<code-mixed question>\"}}\n\n"

    #     f"{dict_hint_stage1}"
    # )


    stage1_base_prompt = (
        f"You are a transliterator. Your task is to convert a {SOURCE_LANG} question "
        f"written in {SOURCE_SCRIPT} script into Roman/Latin script.\n\n"

        f"TASK: DIRECT TRANSLITERATION ONLY\n"
        f"- Rewrite every word of the input in Roman/Latin script\n"
        f"- Do NOT translate any word into English\n"
        f"- Do NOT replace any word with its English meaning\n"
        f"- Preserve ALL words and their order exactly\n"
        f"- The output should be a phonetic rendering of the input\n\n"

        f"STRICT RULES:\n"
        f"- This is NOT a translation task\n"
        f"- This is NOT a question-answering task\n"
        f"- DO NOT answer the question\n"
        f"- DO NOT change word meanings\n"
        f"- DO NOT omit any words\n"
        f"- Output MUST be in Roman/Latin script only\n"
        f"- Do NOT use {SOURCE_SCRIPT} script\n\n"

        f"EXAMPLE ({SOURCE_LANG}):\n"
        f"Input ({SOURCE_SCRIPT}): [a sentence in {SOURCE_SCRIPT} script]\n"
        f"Output (Roman): [the same sentence written phonetically in Roman script]\n\n"

        f"OUTPUT FORMAT:\n"
        f"{{\"translation\": \"<romanized question>\"}}\n\n"
    )

    stage2_base_prompt = (
        f"You are answering a factual question written in {TARGET_LANG}.\n"
        f"This uses {SOURCE_LANG} grammar with English vocabulary (Roman script).\n"
        f"Output ONLY: {{\"answer\": \"...\"}} in {SOURCE_SCRIPT} script.\n\n"
        "Rules:\n"
        "- Answer MUST be from the candidate list\n"
        f"- Answer MUST be in {SOURCE_SCRIPT} script\n"
        "- No English in answer\n\n"
    )

    # print(f"\n[Evaluating {len(test_data)} examples with {n_shot}-shot prompts]")
    print(f"\n[Evaluating {len(test_data)} examples]")
    print(f"📁 Output directory: {output_subdir}")
    print(f"💡 Tip: Run 'tail -f {live_path}' in another terminal to watch progress\n")



    # ==== TWO-STAGE PIPELINE SETUP ====
    stage1_prompts         = []  # Hindi → Hinglish translation prompts
    stage1_metadata        = []  # (ex, lang, index, original_question)
    stage2_prompts         = []  # Hinglish → Hindi answer prompts  
    stage2_metadata        = []  # (ex, lang, index, object_candidates, hinglish_q, original_q, stage1_prompt, stage1_raw)
    
    batch_size      = 8

    live_data = {
        "progress":        "0/0",
        "percent":         "0%",
        "current_example": None,
        "results_so_far":  []
    }
    with open(live_path, "w", encoding="utf-8") as f:
        json.dump(live_data, f, indent=2, ensure_ascii=False)

    pbar = tqdm(total=len(test_data) * 2, desc="Processing (2-Stage)", unit="call")  # *2 for 2 calls per example
    
    # ============================================
    # STAGE 1: Build all translation prompts
    # ============================================
    print(f"\n[STAGE 1/2] Building {len(test_data)} translation prompts...")
    
    for idx, ex in enumerate(test_data):
        lang     = ex.get("language", "unknown")
        relation = ex.get("relation", "unknown")
        index    = ex.get("index", None)

        # Build Stage 1 prompt (Source Lang → Target Lang translation)
        test_question = ex["template"].replace("<subject>", ex["subject"]).replace("<mask>", "").strip()
        
        s1_prompt = stage1_base_prompt
        
        # # Add few-shot examples from dataset (source language questions, like Script 2)
        # key = (relation, lang)
        # candidates_pool = [c[1] for c in candidates_by_key[key] if c[0] != idx]
        # if candidates_pool:
        #     demonstrations = random.sample(candidates_pool, min(n_shot, len(candidates_pool)))
        #     for d in demonstrations:
        #         demo_question = d["template"].replace("<subject>", d["subject"]).replace("<mask>", "").strip()
        #         s1_prompt += f"Q: {demo_question}\nOutput:\n\n"
        
        # s1_prompt += f"Q: {test_question}\nOutput:"
        s1_prompt = stage1_base_prompt + f"\nQ: {test_question}\nOutput:"

        stage1_prompts.append(s1_prompt)
        stage1_metadata.append((ex, lang, index, test_question))

    # ============================================
    # STAGE 1: Execute all translations
    # ============================================
    print(f"[STAGE 1/2] Executing {len(stage1_prompts)} translation calls...")
    
    stage1_sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=60,  # Shorter for just translation
        structured_outputs=StructuredOutputsParams(json=json.dumps(TRANSLATION_SCHEMA))
    )
    
    hinglish_translations = []  # Store results for Stage 2
    
    for i in range(0, len(stage1_prompts), batch_size):
        batch_prompts = stage1_prompts[i:i+batch_size]
        batch_meta = stage1_metadata[i:i+batch_size]
        
        outputs = llm.generate(batch_prompts, stage1_sampling_params)
        
        for j, output in enumerate(outputs):
            ex, lang, index, original_question = batch_meta[j]
            raw_output = output.outputs[0].text
            
            # Parse translation
            try:
                parsed = json.loads(raw_output.strip())
                hinglish_q = parsed.get("translation", "").strip()
            except:
                hinglish_q = raw_output.strip()
            
            hinglish_translations.append({
                "ex": ex,
                "lang": lang,
                "index": index,
                "original_question": original_question,
                "hinglish_question": hinglish_q,
                "stage1_prompt": batch_prompts[j],
                "stage1_raw": raw_output
            })
            
            pbar.update(1)

    # ============================================
    # STAGE 2: Build all QA prompts using Hinglish
    # ============================================
    print(f"\n[STAGE 2/2] Building {len(hinglish_translations)} QA prompts...")
    
    for item in hinglish_translations:
        ex = item["ex"]
        lang = item["lang"]
        index = item["index"]
        hinglish_q = item["hinglish_question"]
        original_q = item["original_question"]
        
        relation = ex.get("relation", "unknown")
        object_candidates = parse_candidates(ex.get("object_candidates"))
        candidates_str = ", ".join(object_candidates) if object_candidates else ""
        # Build Stage 2 prompt (Hinglish → Hindi answer)
        s2_prompt = stage2_base_prompt
        
        # Add few-shot examples from dataset (source language, like Script 2)
        key = (relation, lang)
        candidates_pool = [c[1] for c in candidates_by_key[key] if c[0] != index]
        if candidates_pool:
            demonstrations = random.sample(candidates_pool, min(n_shot, len(candidates_pool)))
            for d in demonstrations:
                demo_question = d["template"].replace("<subject>", d["subject"]).replace("<mask>", "").strip()
                demo_candidates = parse_candidates(d.get("object_candidates"))
                demo_cands_str = ", ".join(demo_candidates) if demo_candidates else d["object"]
                s2_prompt += f"Q: {demo_question}\n"
                s2_prompt += f"Candidates: {demo_cands_str}\n"
                s2_prompt += f"Output: {{\"answer\": \"{d['object']}\"}}\n\n"
        
        s2_prompt += f"Q: {hinglish_q}\nCandidates: {candidates_str}\nOutput:"
        stage2_prompts.append(s2_prompt)

        stage2_metadata.append((
            ex,
            lang,
            index,
            object_candidates,
            hinglish_q,
            original_q,
            item["stage1_prompt"],
            item["stage1_raw"]
        ))

    # ============================================
    # STAGE 2: Execute all QA calls
    # ============================================
    print(f"[STAGE 2/2] Executing {len(stage2_prompts)} QA calls...")
    
    stage2_sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=60,
        structured_outputs=StructuredOutputsParams(json=json.dumps(ANSWER_SCHEMA))
    )
    
    for i in range(0, len(stage2_prompts), batch_size):
        batch_prompts = stage2_prompts[i:i+batch_size]
        batch_meta = stage2_metadata[i:i+batch_size]
        
        outputs = llm.generate(batch_prompts, stage2_sampling_params)
        
        for j, output in enumerate(outputs):
            ex, lang, index, object_candidates, hinglish_q, original_q, s1_prompt, s1_raw = batch_meta[j]
            raw_prediction = output.outputs[0].text
            
            target = ex["object"].strip()
            
            # Parse answer
            try:
                parsed = json.loads(raw_prediction.strip())
                prediction = parsed.get("answer", "").strip()
            except:
                prediction = raw_prediction.strip()

            # Match against candidates
            if object_candidates:
                match, matched_candidate = match_against_candidates(prediction, object_candidates, target)
            else:
                match = is_nontrivial_prefix(prediction, target) or is_nontrivial_prefix(target, prediction)
                matched_candidate = prediction

            # Update counters
            correct_total += int(match)
            total_total += 1
            per_lang_results[lang]["correct"] += int(match)
            per_lang_results[lang]["total"] += 1
            if match:
                per_lang_results[lang]["correct_indices"].append(index)

            # # Build result entry
            result_entry = {
                "index": index,
                "relation": ex.get("relation", "unknown"),
                "subject": ex["subject"],
                f"question_{src_key}": original_q,           # Hindi question
                f"{tgt_key}_question": hinglish_q,           # Hinglish translation (from Stage 1)
                "stage1_prompt": s1_prompt,                  # Full Stage 1 prompt
                "stage1_raw_output": s1_raw,                 # Raw Stage 1 model output
                "stage2_prompt": batch_prompts[j],           # Full Stage 2 prompt
                "stage2_raw_output": raw_prediction,         # Raw Stage 2 model output
                "model_prediction": prediction,              # Final parsed answer
                "matched_candidate": matched_candidate,
                "object_candidates": object_candidates if object_candidates else None,
                "used_candidates": bool(object_candidates),
                "ground_truth": target,
                "is_correct": bool(match),
            }
            
            detailed_results.append(result_entry)
            
            # Live update and print (same as before)
            progress_percent = (total_total / len(test_data)) * 100

            live_data = {
                "progress": f"{total_total}/{len(test_data)}",
                "percent": f"{progress_percent:.1f}%",
                "current_example": {
                    "index": index,
                    "subject": ex["subject"],
                    f"question_{src_key}": original_q,
                    f"{tgt_key}_question": hinglish_q,  # Now explicitly from Stage 1
                    "model_prediction": prediction,
                    "matched_candidate": matched_candidate,
                    "ground_truth": target,
                    "is_correct": bool(match),
                    # "final_prompt"  # REMOVE THIS - replaced with stage-specific prompts below
                    "stage1_prompt": s1_prompt,      # ADD
                    "stage2_prompt": batch_prompts[j],  # ADD
                },
                "running_accuracy": f"{(correct_total/total_total)*100:.2f}%" if total_total > 0 else "0%",
                "results_so_far": detailed_results[-50:]
            }
            
            with open(live_path, "w", encoding="utf-8") as f:
                json.dump(live_data, f, indent=2, ensure_ascii=False)
            
            status = "✅" if match else "❌"
            print(f"\n{status} [{total_total}/{len(test_data)}] Subject: {ex['subject']}")
            print(f"   {SOURCE_LANG} Q:    {original_q}")
            print(f"   {TARGET_LANG}:      {hinglish_q}")
            print(f"   Candidates:        {', '.join(object_candidates) if object_candidates else 'N/A'}")
            print(f"   Raw prediction:    {prediction}")
            print(f"   Matched candidate: {matched_candidate}")
            print(f"   Ground truth:      {target}")
            print(f"   Running Acc:       {(correct_total/total_total)*100:.2f}%")
            
            pbar.update(1)

    pbar.close()

    # pbar.close()

    overall_acc = correct_total / total_total if total_total > 0 else 0


    results = {
        "overall_acc": overall_acc,
        "overall_clc": None,
        "per_language_acc": {},
        "per_language_clc": {},
        "pipeline_type": "explicit_two_stage",  # ADD THIS
        "stage1_description": f"{SOURCE_LANG} -> {TARGET_LANG} translation",
        "stage2_description": f"{TARGET_LANG} question -> {SOURCE_SCRIPT} answer",
    }

    for lang, res in sorted(per_lang_results.items()):
        lang_acc = res["correct"] / res["total"] if res["total"] > 0 else 0
        results["per_language_acc"][lang] = lang_acc

    langs = sorted(list(per_lang_results.keys()))
    for lang in langs:
        others = [l for l in langs if l != lang]
        scores = [
            overlapping_ratio(
                per_lang_results[lang]["correct_indices"],
                per_lang_results[other]["correct_indices"]
            ) for other in others
        ]
        consistency = sum(scores) / len(scores) if scores else 0
        results["per_language_clc"][lang] = consistency

    results["overall_clc"] = (
        sum(results["per_language_clc"].values()) / len(results["per_language_clc"].values())
        if results["per_language_clc"] else 0
    )

    print(f"\n📊 Overall Accuracy: {overall_acc:.2%}")
    for lang, res in sorted(per_lang_results.items()):
        lang_acc = res["correct"] / res["total"] if res["total"] > 0 else 0
        print(f"  {lang}: {lang_acc:.2%} ({res['correct']} / {res['total']})")
    print(f'\n📊 Overall CLC: {results["overall_clc"]:.2%}')

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=4, ensure_ascii=False)

    live_data["status"]         = "COMPLETED"
    live_data["final_accuracy"] = f"{overall_acc:.2f}"
    with open(live_path, "w", encoding="utf-8") as f:
        json.dump(live_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved:")
    print(f"   Summary  → {summary_path}")
    print(f"   Detailed → {detailed_path}")
    print(f"   Live Log → {live_path}")

    return results


# Run evaluation
evaluate(llm, dataset, max_new_tokens=10, n_shot=3)



================================================
FILE: Evaluation-Scripts/automation.sh
================================================
#!/bin/bash
# ============================================================
#  automation.sh
#  Runs all evaluation scripts for multiple languages.
#
#  Scripts run per language:
#    1.  Baseline_filter_knowns           (filter_knowns_live.py)
#    2.  Baseline_filter_knowns_with_obj  (filter_knowns_live_obj.py)
#    3.  Implicit_CM                      (1_call_pure_implicit_cm.py)
#    4.  Implicit_EN                      (1_call_pure_implicit_en.py)
#    5.  1_Call_CM                        (1_call_cm_placeholder.py)
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
#  MODELS — comment/uncomment as needed
#  All active models run for every language.
# ──────────────────────────────────────────
MODELS=(
    # ── Small / tiny models ────────────────
    # "Qwen/Qwen2.5-0.5B"
    # "Qwen/Qwen2.5-1.5B"
    # "Qwen/Qwen2.5-3B"
    # "meta-llama/Llama-3.2-1B"
    # "meta-llama/Llama-3.2-3B"
    "google/gemma-3-270m"
    # "google/gemma-3-270m-it"
    # "google/gemma-3-1b-pt"
    # "google/gemma-3-1b-it"

    # ── Mid-size models ────────────────────
    # "meta-llama/Llama-3.1-8B"
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "Qwen/Qwen2.5-7B"
    # "google/gemma-7b"
    # "google/gemma-3-4b-it"
    # "google/gemma-3-4b"
    # "Qwen/Qwen3-8B"
    # "google/gemma-3-12b-it"
    # "google/gemma-3-12b-pt"

    # ── Large models ───────────────────────
    # "Qwen/Qwen2.5-14B"
    # "Qwen/Qwen2.5-14B-Instruct"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

    # ── Qwen3 Instruct / Thinking ──────────
    # "Qwen/Qwen3-4B-Instruct-2507"
    # "Qwen/Qwen3-4B-Thinking-2507"
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

    # ── Baseline lang_codes: native dir + english dir joined with comma ──
    # e.g.  ben → "ben,ben-en"   |   hin-en → "hin-en,hin-en-en"  (handled by filter scripts)
    local LANG_CODE_WITH_EN="${LANG_CODE},${LANG_CODE}-en"

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
                --model_name    "$MODEL" \
                --seed          "$SEED" \
                --lang_codes    "$LANG_CODE_WITH_EN"

        # ── Script 2: Baseline_filter_knowns_with_obj  (native + en) ──────
        run_script \
            "Baseline_filter_knowns_with_obj | ${SOURCE_LANG} | ${MODEL}" \
            "$(logfile "$LANG_CODE" "Baseline_filter_knowns_with_obj" "$MODEL")" \
            ${SCRIPT_DIR}/filter_knowns_live_obj.py \
                --model_name    "$MODEL" \
                --seed          "$SEED" \
                --lang_codes    "$LANG_CODE_WITH_EN"

        # ── Script 3: Implicit_CM  (native only) ──────────────────────────
        run_script \
            "Implicit_CM | ${SOURCE_LANG} | ${MODEL}" \
            "$(logfile "$LANG_CODE" "Implicit_CM" "$MODEL")" \
            ${SCRIPT_DIR}/1_call_pure_implicit_cm.py \
                --model_name    "$MODEL" \
                --seed          "$SEED" \
                --lang_code     "$LANG_CODE" \
                --data_dir      "$DATA_DIR" \
                --source_lang   "$SOURCE_LANG" \
                --source_script "$SOURCE_SCRIPT" \
                --target_lang   "$TARGET_LANG_CM"

        # ── Script 4: Implicit_EN  (native only) ──────────────────────────
        run_script \
            "Implicit_EN | ${SOURCE_LANG} | ${MODEL}" \
            "$(logfile "$LANG_CODE" "Implicit_EN" "$MODEL")" \
            ${SCRIPT_DIR}/1_call_pure_implicit_en.py \
                --model_name    "$MODEL" \
                --seed          "$SEED" \
                --lang_code     "$LANG_CODE" \
                --data_dir      "$DATA_DIR" \
                --source_lang   "$SOURCE_LANG" \
                --source_script "$SOURCE_SCRIPT" \
                --target_lang   "$TARGET_LANG_EN"

        # ── Script 5: 1_Call_CM  (native only) ────────────────────────────
        run_script \
            "1_Call_CM | ${SOURCE_LANG} | ${MODEL}" \
            "$(logfile "$LANG_CODE" "1_Call_CM" "$MODEL")" \
            ${SCRIPT_DIR}/1_call_cm_placeholder.py \
                --model_name    "$MODEL" \
                --seed          "$SEED" \
                --lang_code     "$LANG_CODE" \
                --data_dir      "$DATA_DIR" \
                --source_lang   "$SOURCE_LANG" \
                --source_script "$SOURCE_SCRIPT" \
                --target_lang   "$TARGET_LANG_CM"

        # ── Script 6: 1_Call_EN  (native only) ────────────────────────────
        run_script \
            "1_Call_EN | ${SOURCE_LANG} | ${MODEL}" \
            "$(logfile "$LANG_CODE" "1_Call_EN" "$MODEL")" \
            ${SCRIPT_DIR}/1_call_en_placeholder.py \
                --model_name    "$MODEL" \
                --seed          "$SEED" \
                --data_dir      "$DATA_DIR" \
                --lang_codes    "$LANG_CODE" \
                --source_lang   "$SOURCE_LANG" \
                --source_script "$SOURCE_SCRIPT" \
                --target_lang   "$TARGET_LANG_EN"

        # ── Script 7: 2_Call_CM  (native only) ────────────────────────────
        run_script \
            "2_Call_CM | ${SOURCE_LANG} | ${MODEL}" \
            "$(logfile "$LANG_CODE" "2_Call_CM" "$MODEL")" \
            ${SCRIPT_DIR}/2_call_cm_placeholder_correct.py \
                --model_name    "$MODEL" \
                --seed          "$SEED" \
                --lang_code     "$LANG_CODE" \
                --data_dir      "$DATA_DIR" \
                --source_lang   "$SOURCE_LANG" \
                --source_script "$SOURCE_SCRIPT" \
                --target_lang   "$TARGET_LANG_CM"

        # ── Script 8: 2_Call_EN  (native only) ────────────────────────────
        run_script \
            "2_Call_EN | ${SOURCE_LANG} | ${MODEL}" \
            "$(logfile "$LANG_CODE" "2_Call_EN" "$MODEL")" \
            ${SCRIPT_DIR}/2_call_en_placeholder.py \
                --model_name    "$MODEL" \
                --seed          "$SEED" \
                --data_dir      "$DATA_DIR" \
                --lang_codes    "$LANG_CODE" \
                --source_lang   "$SOURCE_LANG" \
                --source_script "$SOURCE_SCRIPT" \
                --target_lang   "$TARGET_LANG_EN"

        # ── Script 9: 2_Call_Transliteration  (native only) ───────────────
        run_script \
            "2_Call_Transliteration | ${SOURCE_LANG} | ${MODEL}" \
            "$(logfile "$LANG_CODE" "2_Call_Transliteration" "$MODEL")" \
            ${SCRIPT_DIR}/2_call_transliteration.py \
                --model_name    "$MODEL" \
                --seed          "$SEED" \
                --lang_code     "$LANG_CODE" \
                --data_dir      "$DATA_DIR" \
                --source_lang   "$SOURCE_LANG" \
                --source_script "$SOURCE_SCRIPT" \
                --target_lang   "Transliterated"

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
                --model_name    "$MODEL" \
                --seed          "$SEED" \
                --lang_codes    "$LANG_CODE"

        # -- Script 2: Baseline_filter_knowns_with_obj  (en only) ---------
        run_script \
            "Baseline_filter_knowns_with_obj | English | ${MODEL}" \
            "$(logfile "$LANG_CODE" "Baseline_filter_knowns_with_obj" "$MODEL")" \
            ${SCRIPT_DIR}/filter_knowns_live_obj.py \
                --model_name    "$MODEL" \
                --seed          "$SEED" \
                --lang_codes    "$LANG_CODE"

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
# run_language_en

# ── Indic languages ───────────────────────────────────────────────────────────
# run_language  "hin"    "Hindi"      "Hindi"      "Hinglish"
# run_language  "hin-en" "Hindi"      "Hindi"      "Hinglish"
# run_language  "ben"    "Bengali"    "Bengali"    "Banglish"
# run_language  "asm"    "Assamese"   "Assamese"   "Assamglish"
# run_language  "ori"    "Odia"       "Odia"       "Odiglish"
# run_language  "guj"    "Gujarati"   "Gujarati"   "Gujlish"
run_language  "tel"    "Telugu"     "Telugu"     "Teluglish"
# run_language  "mal"    "Malayalam"  "Malayalam"  "Malyalamglish"
# run_language  "mai"    "Maithili"   "Maithili"   "Maithilish"
# run_language  "nep"    "Nepali"     "Nepali"     "Nepglish"

# ============================================================
#  DONE
# ============================================================
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  🎉  All evaluations complete!                       ║"
echo "║  Logs → logs/<lang_code>/                            ║"
echo "╚══════════════════════════════════════════════════════╝"



================================================
FILE: Evaluation-Scripts/filter_knowns_live.py
================================================
import os
import json
import glob
import random
import argparse
import numpy
from vllm import LLM, SamplingParams
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
from tqdm import tqdm

# ==== Argument parser ====
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B")
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument(
    "--lang_codes",
    type=str,
    required=True,
    help="Comma-separated language codes, e.g. asm,ben,guj"
)
args = parser.parse_args()

model_name = args.model_name

# ==== Set random seed ====
def set_seed(seed: int) -> None:
    """Globally set random seed."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)

# ==== Define valid languages and relations ====
# languages = {
#     # "meta-llama/Llama-3.2-1B": [ "en","hin","hinglish_dev"],
#     # "meta-llama/Llama-3.1-8B": ["en","hinglish_dev"],
#     # "Qwen/Qwen2.5-7B": ["en","hinglish_dev"],
#     # "meta-llama/Llama-3.1-8B": ["asm","asm-en","ben","bn-en","guj","guj-en","mal","mal-en","ori","ori-en","mar"],
#     # "Qwen/Qwen2.5-7B": ["asm","asm-en","ben","bn-en","guj","guj-en","mal","mal-en","ori","ori-en","mar"],
#     "google/gemma-7b": ["asm","asm-en","ben","bn-en","guj","guj-en","mal","mal-en","ori","ori-en","mar"],


#     # "meta-llama/Llama-2-7b-hf": ["hi","hinglish", "en", "bn-eng"], #["ca", "en", "es", "fr", "hu", "ja", "ko", "nl", "ru", "uk", "vi", "zh"],
#     # "bigscience/bloom-560m": ["hinglish"] #, "ca", "en", "es", "fr", "vi", "zh"]
# }
relations = [
    "applies_to_jurisdiction", "capital", "capital_of", "continent", \
    "country_of_citizenship", "developer", "field_of_work", "headquarters_location", \
    "instrument", "language_of_work_or_name", "languages_spoken", "location_of_formation", \
    "manufacturer", "native_language", "occupation", "official_language", \
    "owned_by", "place_of_birth", "place_of_death","religion"
]
# valid_langs = set(languages[model_name])
valid_langs = set(c.strip() for c in args.lang_codes.split(",") if c.strip())
valid_rels = set(relations)

# ====  Group file paths by relation ====
json_paths = glob.glob("cm_klar/*/*.json")
path_map = defaultdict(dict)  # (relation -> lang -> path)

for path in json_paths:
    lang = os.path.basename(os.path.dirname(path))
    rel = os.path.splitext(os.path.basename(path))[0]
    if lang in valid_langs and rel in valid_rels:
        path_map[rel][lang] = path

# ==== Load samples ====
samples = []

for rel, lang_paths in path_map.items():
    if not all(lang in lang_paths for lang in valid_langs):
        continue

    for lang in valid_langs:
        with open(lang_paths[lang], "r", encoding="utf-8") as f:
            content = json.load(f)
            loaded_samples = content["samples"]
            template = content["prompt_templates"][0]

            for sample in loaded_samples:
                new_sample = {
                    "subject": sample["subject"],
                    "object": sample["object"],
                    "language": lang,
                    "relation": rel,
                    "template": template,
                    "index": sample["index"]
                }
                samples.append(new_sample)

# ==== Apply prompt formatting ====
def apply_prompt(example):
    prompt = example["template"].replace("<subject>", example["subject"]).replace("<mask>", "")
    example["input"] = prompt.strip()
    example["target"] = " " + example["object"]
    return example

dataset = Dataset.from_list([apply_prompt(ex) for ex in samples])

# ==== Replace model loading ====
# Instead of:
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")

# Use:
# llm = LLM(model=model_name, trust_remote_code=True, gpu_memory_utilization=0.9)
llm = LLM(
    model=model_name,
    trust_remote_code=True,
    gpu_memory_utilization=0.20,   # you already bumped this to 0.9
    max_model_len=4096,            # cap sequence length to save memory
    max_num_seqs=16                # max parallel sequences vLLM processes at once
)
sampling_params = SamplingParams(temperature=0.0, max_tokens=10)

# ==== Evaluation ====
def is_nontrivial_prefix(prediction: str, target: str) -> bool:
    """Return true if prediction is (case insensitive) prefix of the target."""
    target = target.lower().strip()
    prediction = prediction.lower().strip()
    return len(prediction) > 0 and target.startswith(prediction)


def overlapping_ratio(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0


OUTPUT_DIR = "filter_knowns_live_baseline"  # ← add this near the top of the file

def evaluate(llm, dataset, max_new_tokens=10, n_shot=3, save_path="filter_knowns_live_baseline/results.json"):
    test_data = list(dataset)
    correct_total = 0
    total_total = 0
    per_lang_results = defaultdict(lambda: {"correct": 0, "total": 0, "correct_indices": []})
    detailed_results = []

    model_safe_name = model_name.replace("/", "_")
    live_dir = os.path.join(OUTPUT_DIR, model_safe_name)
    os.makedirs(live_dir, exist_ok=True)
    live_path = os.path.join(live_dir, "LIVE.json")

    # Per-language detailed output dirs (created on demand)
    lang_detailed_results = defaultdict(list)

    live_data = {"progress": "0/0", "percent": "0%", "current_example": None, "results_so_far": []}
    with open(live_path, "w", encoding="utf-8") as f:
        json.dump(live_data, f, indent=2, ensure_ascii=False)

    print(f"📁 Live log: {live_path}")
    print(f"💡 Tip: Run 'tail -f {live_path}' in another terminal to watch progress\n")

    candidates_by_key = defaultdict(list)
    for i, ex in enumerate(test_data):
        key = (ex.get("relation"), ex.get("language"))
        candidates_by_key[key].append((i, ex))

    print(f"\n[Evaluating {len(test_data)} examples with {n_shot}-shot prompts]")

    prompts = []
    prompt_metadata = []
    batch_size = 8

    for idx, ex in enumerate(tqdm(test_data)):
        lang = ex.get("language", "unknown")
        relation = ex.get("relation", "unknown")
        index = ex.get("index", None)

        key = (relation, lang)
        candidates = [c[1] for c in candidates_by_key[key] if c[0] != idx]
        demonstrations = random.sample(candidates, min(n_shot, len(candidates)))

        few_shot_prompt = "".join([f"{d['input']}{d['target']}\n" for d in demonstrations]) + ex["input"]
        prompts.append(few_shot_prompt)
        prompt_metadata.append((ex, few_shot_prompt, lang, index))

        if len(prompts) == batch_size or idx == len(test_data) - 1:
            outputs = llm.generate(prompts, sampling_params)

            for i, output in enumerate(outputs):
                ex, few_shot_prompt, lang, index = prompt_metadata[i]
                prediction = output.outputs[0].text.split('\n')[0].strip()
                target = ex["target"]

                match = is_nontrivial_prefix(prediction, target) or is_nontrivial_prefix(target, prediction)
                correct_total += int(match)
                total_total += 1
                per_lang_results[lang]["correct"] += int(match)
                per_lang_results[lang]["total"] += 1
                if match:
                    per_lang_results[lang]["correct_indices"].append(index)

                result_entry = {
                    "index": index,
                    "relation": ex.get("relation"),
                    "subject": ex.get("subject"),
                    "language": lang,
                    "prompt": few_shot_prompt,
                    "model_prediction": prediction,
                    "ground_truth": target.strip(),
                    "is_correct": bool(match),
                    "final_prompt": few_shot_prompt
                }
                detailed_results.append(result_entry)
                lang_detailed_results[lang].append(result_entry)

                progress_pct = (total_total / len(test_data)) * 100
                live_data = {
                    "progress": f"{total_total}/{len(test_data)}",
                    "percent": f"{progress_pct:.1f}%",
                    "running_accuracy": f"{(correct_total / total_total) * 100:.2f}%",
                    "current_example": result_entry,
                    "results_so_far": detailed_results[-50:]
                }
                with open(live_path, "w", encoding="utf-8") as f:
                    json.dump(live_data, f, indent=2, ensure_ascii=False)

            prompts = []
            prompt_metadata = []

    # Save per-language detailed JSONs
    for lang, lang_results in lang_detailed_results.items():
        lang_dir = os.path.join(live_dir, lang)
        os.makedirs(lang_dir, exist_ok=True)
        lang_correct = sum(1 for r in lang_results if r["is_correct"])
        lang_total = len(lang_results)
        lang_acc = lang_correct / lang_total if lang_total > 0 else 0
        lang_output = {
            "language": lang,
            "accuracy": f"{lang_acc:.2%}",
            "correct": lang_correct,
            "total": lang_total,
            "results": lang_results
        }
        lang_detailed_path = os.path.join(lang_dir, "detailed.json")
        with open(lang_detailed_path, "w", encoding="utf-8") as f:
            json.dump(lang_output, f, indent=4, ensure_ascii=False)
        print(f"💾 Saved {lang} detailed results → {lang_detailed_path}")

    live_data["status"] = "COMPLETED"
    with open(live_path, "w", encoding="utf-8") as f:
        json.dump(live_data, f, indent=2, ensure_ascii=False)

    overall_acc = correct_total / total_total if total_total > 0 else 0
    print(f"\n📊 Overall Accuracy: {overall_acc:.2%}")

    results = {"overall_acc": overall_acc, "overall_clc": None, "per_language_acc": {}, "per_language_clc": {}}

    for lang, res in sorted(per_lang_results.items()):
        lang_acc = res["correct"] / res["total"] if res["total"] > 0 else 0
        results["per_language_acc"][lang] = lang_acc
        print(f"  {lang}: {lang_acc:.2%} ({res['correct']} / {res['total']})")

    langs = sorted(list(per_lang_results.keys()))
    for lang in langs:
        others = [l for l in langs if l != lang]
        scores = [overlapping_ratio(per_lang_results[lang]["correct_indices"], per_lang_results[other]["correct_indices"]) for other in others]
        consistency = sum(scores) / len(scores) if scores else 0
        results["per_language_clc"][lang] = consistency

    results["overall_clc"] = sum(results["per_language_clc"].values()) / len(results["per_language_clc"].values())

    print(f'\n📊 Overall CLC: {results["overall_clc"]:.2%}')
    for lang in langs:
        print(f'  {lang} cross-lingual consistency: {results["per_language_clc"][lang]:.2%}')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    return results

safe_save_path = f"filter_knowns_live/{model_name.replace('/', '_')}_results.json"
os.makedirs(os.path.dirname(safe_save_path), exist_ok=True)
evaluate(llm, dataset, max_new_tokens=10, n_shot=3, save_path=safe_save_path)



================================================
FILE: Evaluation-Scripts/filter_knowns_live_obj.py
================================================
import os
import json
import glob
import random
import argparse
import numpy
from vllm import LLM, SamplingParams
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
from tqdm import tqdm

# ==== Argument parser ====
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument(
    "--lang_codes",
    type=str,
    required=True,
    help="Comma-separated language codes, e.g. asm,ben,guj"
)
args = parser.parse_args()

model_name = args.model_name

# ==== Set random seed ====
def set_seed(seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)

# # ==== Define valid languages and relations ====F
# languages = {
#     # "meta-llama/Llama-3.2-1B": ["hin"],
#     # "meta-llama/Llama-3.1-8B": ["hin","en","hinglish_dev"],
#     # "Qwen/Qwen2.5-7B": ["hin","en","hinglish_dev"],
#     # "meta-llama/Llama-3.1-8B": ["asm","asm-en","ben","bn-en","guj","guj-en","mal","mal-en","ori","ori-en","mar"],
#     "meta-llama/Llama-3.1-8B-Instruct": ["asm","asm-en","ben","bn-en","guj","guj-en","mal","mal-en","ori","ori-en","mar"],
#     # "Qwen/Qwen2.5-7B": ["asm","asm-en","ben","bn-en","guj","guj-en","mal","mal-en","ori","ori-en","mar"],
#     # "google/gemma-7b": ["asm","asm-en","ben","bn-en","guj","guj-en","mal","mal-en","ori","ori-en","mar"],

# }
relations = [
    "applies_to_jurisdiction", "capital", "capital_of", "continent",
    "country_of_citizenship", "developer", "field_of_work", "headquarters_location",
    "instrument", "language_of_work_or_name", "languages_spoken", "location_of_formation",
    "manufacturer", "native_language", "occupation", "official_language",
    "owned_by", "place_of_birth", "place_of_death", "religion"
]
# valid_langs = set(languages[model_name])
valid_langs = set(c.strip() for c in args.lang_codes.split(",") if c.strip())
valid_rels = set(relations)

# ==== Group file paths by relation ====
json_paths = glob.glob("cm_klar/*/*.json")
path_map = defaultdict(dict)

for path in json_paths:
    lang = os.path.basename(os.path.dirname(path))
    rel = os.path.splitext(os.path.basename(path))[0]
    if lang in valid_langs and rel in valid_rels:
        path_map[rel][lang] = path

# ==== Load samples ====
samples = []

for rel, lang_paths in path_map.items():
    if not all(lang in lang_paths for lang in valid_langs):
        continue
    for lang in valid_langs:
        with open(lang_paths[lang], "r", encoding="utf-8") as f:
            content = json.load(f)
            loaded_samples = content["samples"]
            template = content["prompt_templates"][0]
            for sample in loaded_samples:
                new_sample = {
                    "subject": sample["subject"],
                    "object": sample["object"],
                    "object_candidates": sample.get("object_candidates", None),
                    "language": lang,
                    "relation": rel,
                    "template": template,
                    "index": sample["index"]
                }
                samples.append(new_sample)

# ==== Apply prompt formatting ====
def apply_prompt(example):
    prompt = example["template"].replace("<subject>", example["subject"]).replace("<mask>", "")
    example["input"] = prompt.strip()
    example["target"] = " " + example["object"]
    return example

dataset = Dataset.from_list([apply_prompt(ex) for ex in samples])

# ==== Model loading ====
llm = LLM(
    model=model_name,
    trust_remote_code=True,
    gpu_memory_utilization=0.20,
    max_model_len=4096,
    max_num_seqs=16
)
sampling_params = SamplingParams(temperature=0.0, max_tokens=10, stop=["\n", "Answer"])

# ==== Evaluation helpers ====
def is_nontrivial_prefix(prediction: str, target: str) -> bool:
    target = target.lower().strip()
    prediction = prediction.lower().strip()
    return len(prediction) > 0 and target.startswith(prediction)

def overlapping_ratio(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0

def parse_candidates(raw):
    if raw is None:
        return []
    if isinstance(raw, list):
        return [c.strip() for c in raw if c.strip()]
    return [c.strip() for c in raw.split(",") if c.strip()]

def match_against_candidates(prediction, candidates, target):
    pred_low = prediction.lower().strip()
    best_cand = None
    for cand in candidates:
        cand_low = cand.lower().strip()
        if not cand_low:
            continue
        if pred_low == cand_low:
            best_cand = cand
            break
        if pred_low.startswith(cand_low):
            best_cand = cand
            break
        if cand_low.startswith(pred_low) and len(pred_low) > 0:
            best_cand = cand
            break
        if cand_low in pred_low:
            best_cand = cand
            break
    if best_cand is not None:
        is_correct = best_cand.lower().strip() == target.lower().strip()
        return is_correct, best_cand
    is_correct = is_nontrivial_prefix(prediction, target) or is_nontrivial_prefix(target, prediction)
    return is_correct, prediction

# ==== Output directory ====
OUTPUT_DIR = "filter_knowns_live_with_obj_baseline"

def evaluate(llm, dataset, max_new_tokens=10, n_shot=3, save_path="filter_knowns_live_with_obj_baseline/results.json"):
    test_data = list(dataset)
    correct_total = 0
    total_total = 0
    per_lang_results = defaultdict(lambda: {"correct": 0, "total": 0, "correct_indices": []})
    detailed_results = []

    model_safe_name = model_name.replace("/", "_")
    live_dir = os.path.join(OUTPUT_DIR, model_safe_name)
    os.makedirs(live_dir, exist_ok=True)
    live_path = os.path.join(live_dir, "LIVE.json")

    # Per-language detailed output dirs (created on demand)
    lang_detailed_results = defaultdict(list)

    live_data = {"progress": "0/0", "percent": "0%", "current_example": None, "results_so_far": []}
    with open(live_path, "w", encoding="utf-8") as f:
        json.dump(live_data, f, indent=2, ensure_ascii=False)

    print(f"📁 Live log: {live_path}")
    print(f"💡 Tip: Run 'tail -f {live_path}' in another terminal to watch progress\n")

    candidates_by_key = defaultdict(list)
    for i, ex in enumerate(test_data):
        key = (ex.get("relation"), ex.get("language"))
        candidates_by_key[key].append((i, ex))

    print(f"\n[Evaluating {len(test_data)} examples with {n_shot}-shot prompts]")

    prompts = []
    prompt_metadata = []
    batch_size = 8

    for idx, ex in enumerate(tqdm(test_data)):
        lang = ex.get("language", "unknown")
        relation = ex.get("relation", "unknown")
        index = ex.get("index", None)

        key = (relation, lang)
        candidates = [c[1] for c in candidates_by_key[key] if c[0] != idx]
        demonstrations = random.sample(candidates, min(n_shot, len(candidates)))

        # Few-shot demos (no candidates shown)
        few_shot_prompt = "".join([f"{d['input']}{d['target']}\n" for d in demonstrations])

        # Test question with candidates shown to model
        object_candidates = parse_candidates(ex.get("object_candidates", None))
        if object_candidates:
            candidates_str = ", ".join(object_candidates)
            few_shot_prompt += f"{ex['input']}\nCandidates: {candidates_str}\nAnswer:"
        else:
            few_shot_prompt += ex["input"]

        prompts.append(few_shot_prompt)
        prompt_metadata.append((ex, few_shot_prompt, lang, index))

        if len(prompts) == batch_size or idx == len(test_data) - 1:
            outputs = llm.generate(prompts, sampling_params)

            for i, output in enumerate(outputs):
                ex, few_shot_prompt, lang, index = prompt_metadata[i]
                prediction = output.outputs[0].text.split('\n')[0].strip()
                target = ex["target"].strip()

                # Parse candidates for matching
                candidates_list = parse_candidates(ex.get("object_candidates", None))

                if candidates_list:
                    match, matched_candidate = match_against_candidates(prediction, candidates_list, target)
                else:
                    match = is_nontrivial_prefix(prediction, target) or is_nontrivial_prefix(target, prediction)
                    matched_candidate = prediction

                correct_total += int(match)
                total_total += 1
                per_lang_results[lang]["correct"] += int(match)
                per_lang_results[lang]["total"] += 1
                if match:
                    per_lang_results[lang]["correct_indices"].append(index)

                result_entry = {
                    "index": index,
                    "relation": ex.get("relation"),
                    "subject": ex.get("subject"),
                    "language": lang,
                    "prompt": few_shot_prompt,
                    "model_prediction": prediction,
                    "matched_candidate": matched_candidate,
                    "ground_truth": target.strip(),
                    "is_correct": bool(match),
                    "final_prompt": few_shot_prompt,
                    "object_candidates": candidates_list if candidates_list else None,
                    "used_candidates": bool(candidates_list)
                }
                detailed_results.append(result_entry)
                lang_detailed_results[lang].append(result_entry)

                progress_pct = (total_total / len(test_data)) * 100
                live_data = {
                    "progress": f"{total_total}/{len(test_data)}",
                    "percent": f"{progress_pct:.1f}%",
                    "running_accuracy": f"{(correct_total / total_total) * 100:.2f}%" if total_total > 0 else "0%",
                    "current_example": result_entry,
                    "results_so_far": detailed_results[-50:]
                }
                with open(live_path, "w", encoding="utf-8") as f:
                    json.dump(live_data, f, indent=2, ensure_ascii=False)

            prompts = []
            prompt_metadata = []

    # Save per-language detailed JSONs
    for lang, lang_results in lang_detailed_results.items():
        lang_dir = os.path.join(live_dir, lang)
        os.makedirs(lang_dir, exist_ok=True)
        lang_correct = sum(1 for r in lang_results if r["is_correct"])
        lang_total = len(lang_results)
        lang_acc = lang_correct / lang_total if lang_total > 0 else 0
        lang_output = {
            "language": lang,
            "accuracy": f"{lang_acc:.2%}",
            "correct": lang_correct,
            "total": lang_total,
            "results": lang_results
        }
        lang_detailed_path = os.path.join(lang_dir, "detailed.json")
        with open(lang_detailed_path, "w", encoding="utf-8") as f:
            json.dump(lang_output, f, indent=4, ensure_ascii=False)
        print(f"💾 Saved {lang} detailed results → {lang_detailed_path}")

    live_data["status"] = "COMPLETED"
    with open(live_path, "w", encoding="utf-8") as f:
        json.dump(live_data, f, indent=2, ensure_ascii=False)

    overall_acc = correct_total / total_total if total_total > 0 else 0
    print(f"\n📊 Overall Accuracy: {overall_acc:.2%}")

    results = {"overall_acc": overall_acc, "overall_clc": None, "per_language_acc": {}, "per_language_clc": {}}

    for lang, res in sorted(per_lang_results.items()):
        lang_acc = res["correct"] / res["total"] if res["total"] > 0 else 0
        results["per_language_acc"][lang] = lang_acc
        print(f"  {lang}: {lang_acc:.2%} ({res['correct']} / {res['total']})")

    langs = sorted(list(per_lang_results.keys()))
    for lang in langs:
        others = [l for l in langs if l != lang]
        scores = [overlapping_ratio(per_lang_results[lang]["correct_indices"], per_lang_results[other]["correct_indices"]) for other in others]
        consistency = sum(scores) / len(scores) if scores else 0
        results["per_language_clc"][lang] = consistency

    results["overall_clc"] = sum(results["per_language_clc"].values()) / len(results["per_language_clc"].values())

    print(f'\n📊 Overall CLC: {results["overall_clc"]:.2%}')
    for lang in langs:
        print(f'  {lang} cross-lingual consistency: {results["per_language_clc"][lang]:.2%}')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    return results

safe_save_path = f"filter_knowns_live-obj/{model_name.replace('/', '_')}_results.json"
os.makedirs(os.path.dirname(safe_save_path), exist_ok=True)
evaluate(llm, dataset, max_new_tokens=10, n_shot=3, save_path=safe_save_path)



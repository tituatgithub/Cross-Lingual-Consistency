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
OUTPUT_DIR = "filter_knowns_implicit-trans-en"
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
    gpu_memory_utilization=0.25,
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

    base_prompt = (
        f"You are answering factual questions written in {SOURCE_LANG} ({SOURCE_SCRIPT} script).\n"
        f"Step 1 (internal only): Mentally translate the {SOURCE_LANG} question into {TARGET_LANG} "
        f"to understand it clearly.\n"
        "Step 2: Based on your understanding, output a JSON object with exactly two fields:\n"
        f"  \"translation\": the {TARGET_LANG} translation of the question,\n"
        f"  \"answer\": the correct answer in {SOURCE_LANG} {SOURCE_SCRIPT} script, chosen from the provided candidates.\n\n"
        "Rules:\n"
        "- The answer MUST be one of the listed candidates.\n"
        f"- The answer MUST be written in {SOURCE_LANG} {SOURCE_SCRIPT} script only.\n"
        f"- Do NOT use {TARGET_LANG} in the answer field.\n\n"
    )

    print(f"\n[Evaluating {len(test_data)} examples with {n_shot}-shot prompts]")
    print(f"📁 Output directory: {output_subdir}")
    print(f"💡 Tip: Run 'tail -f {live_path}' in another terminal to watch progress\n")

    prompts = []
    prompt_metadata = []
    batch_size = 1

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

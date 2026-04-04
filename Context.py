================================================
FILE: README.md
================================================
# Cross-Lingual-Consistency


================================================
FILE: Evaluation-Scripts/Hindi_preds/Explicit_CM.py
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
import time

# ==== CONFIGURATION - SET THIS ONCE ====
OUTPUT_DIR = "filter_knowns_iExplicit_CM"
# ======================================

# ==== Argument parser ====
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
parser.add_argument("--seed", type=int, default=12345)
args = parser.parse_args()

model_name = args.model_name

# ==== Set random seed ====
def set_seed(seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)

# ==== Define valid languages and relations ====
languages = {
    "Qwen/Qwen2.5-0.5B":  ["hin"],
    "Qwen/Qwen2.5-1.5B":  ["hin"],
    "Qwen/Qwen2.5-3B":    ["hin"],
    "meta-llama/Llama-3.2-1B": ["hin"],
    "meta-llama/Llama-3.2-3B": ["hin"],
}

relations = [
    "applies_to_jurisdiction", "capital", "capital_of", "continent",
    "country_of_citizenship", "developer", "field_of_work", "headquarters_location",
    "instrument", "language_of_work_or_name", "languages_spoken", "location_of_formation",
    "manufacturer", "native_language", "occupation", "official_language",
    "owned_by", "place_of_birth", "place_of_death", "religion"
]
valid_langs = set(languages[model_name])
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
                    # ---- CHANGE: load object_candidates if present ----
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
    gpu_memory_utilization=0.7,
    max_model_len=4096,
    max_num_seqs=16
)
sampling_params = SamplingParams(temperature=0.0, max_tokens=30, stop=["\n", "Answer"])
translation_params = SamplingParams(temperature=0.0, max_tokens=100, stop=["\n", "Hindi:", "Example:"])


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


# ---- NEW: match prediction against a candidate list ----
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

    base_prompt = "Answer with a short factual phrase in Hindi using Devanagari script only. Do not use English.\n\n"

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

        # ---- Call 1: Translate Hindi question to Hinglish ----
        test_question = ex["template"].replace("<subject>", ex["subject"]).replace("<mask>", "").strip()
        translation_prompt = (
            "Translate this Hindi question to Hinglish. Hinglish means writing Hindi words using English (Roman) letters, "
            "while keeping some English words too.\n\n"
            "Rules:\n"
            "- Output MUST use only English letters (a-z, A-Z)\n"
            "- Replace Hindi words with their Roman spelling\n"
            "- Keep question words like 'what', 'where', 'who' in English\n\n"
            "Examples:\n"
            "Hindi: दिल्ली भारत की राजधानी क्या है?\n"
            "Hinglish: Dilli Bharat ki capital kya hai?\n\n"
            "Hindi: आइंस्टीन का पेशा क्या था?\n"
            "Hinglish: Einstein ka occupation kya tha?\n\n"
            "Hindi: मासूम XI की मृत्यु कहाँ हुई थी?\n"
            "Hinglish: Masoom XI ki death kahan hui thi?\n\n"
            f"Hindi: {test_question}\n"
            "Hinglish:"
        )
        translation_output  = llm.generate([translation_prompt], translation_params)
        hinglish_question   = translation_output[0].outputs[0].text.strip().split('\n')[0].strip()

        # ---- Build few-shot prompt ----
        # ---- CHANGE: if object_candidates exist, append them to the prompt ----
        object_candidates = parse_candidates(ex.get("object_candidates"))

        few_shot_prompt = base_prompt
        for d in demonstrations:
            demo_question = d["template"].replace("<subject>", d["subject"]).replace("<mask>", "").strip()
            demo_question = demo_question.replace("उत्तर है:", "").replace("?", "").strip()
            few_shot_prompt += f"Q: {demo_question}\nA: {d['object']}\n\n"

        hinglish_clean = hinglish_question.replace("Answer is:", "").replace("Uttar hai:", "").replace("?", "").strip()

        if object_candidates:
            # Present a constrained multiple-choice style prompt
            candidates_str = ", ".join(object_candidates)
            few_shot_prompt += (
                f"Q: {hinglish_clean}\n"
                f"Candidates: {candidates_str}\n"
                f"A:"
            )
        else:
            # No candidates — open generation as before
            few_shot_prompt += f"Q: {hinglish_clean}\nA:"

        prompts.append(few_shot_prompt)
        prompt_metadata.append((ex, lang, index, hinglish_question, object_candidates))

        if len(prompts) == batch_size or idx == len(test_data) - 1:
            outputs = llm.generate(prompts, sampling_params)

            for i, output in enumerate(outputs):
                ex, lang, index, hinglish_question, object_candidates = prompt_metadata[i]
                raw_prediction = output.outputs[0].text

                target   = ex["object"].strip()
                question = ex["template"].replace("<subject>", ex["subject"]).replace("<mask>", "").strip()

                lines      = raw_prediction.strip().split('\n')
                prediction = lines[0].strip()

                # ---- CHANGE: use candidate-aware matching when candidates are available ----
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
                    "question_hindi":     question,
                    "codemixed_question": hinglish_question,
                    "model_prediction":   prediction,
                    # ---- CHANGE: extra fields for candidate-based evaluation ----
                    "matched_candidate":  matched_candidate,
                    "object_candidates":  object_candidates if object_candidates else None,
                    "used_candidates":    bool(object_candidates),
                    "ground_truth":       target,
                    "is_correct":         bool(match),
                    "raw_output":         raw_prediction
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
                        "question_hindi":     question,
                        "codemixed_question": hinglish_question,
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
                print(f"   Hindi Q:         {question}")
                print(f"   Hinglish Q:      {hinglish_question}")
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
FILE: Evaluation-Scripts/Hindi_preds/Explicit_EN.py
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
import time

# ==== CONFIGURATION - SET THIS ONCE ====
OUTPUT_DIR = "filter_knowns_explicit_en"
# ======================================

# ==== Argument parser ====
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
parser.add_argument("--seed", type=int, default=12345)
args = parser.parse_args()

model_name = args.model_name

# ==== Set random seed ====
def set_seed(seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)

# ==== Define valid languages and relations ====
languages = {
    "Qwen/Qwen2.5-0.5B":  ["hin"],
    "Qwen/Qwen2.5-1.5B":  ["hin"],
    "Qwen/Qwen2.5-3B":    ["hin"],
    "meta-llama/Llama-3.2-1B": ["hin"],
    "meta-llama/Llama-3.2-3B": ["hin"],
}

relations = [
    "applies_to_jurisdiction", "capital", "capital_of", "continent",
    "country_of_citizenship", "developer", "field_of_work", "headquarters_location",
    "instrument", "language_of_work_or_name", "languages_spoken", "location_of_formation",
    "manufacturer", "native_language", "occupation", "official_language",
    "owned_by", "place_of_birth", "place_of_death", "religion"
]
valid_langs = set(languages[model_name])
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
                    # ---- CHANGE: load object_candidates if present ----
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
    gpu_memory_utilization=0.7,
    max_model_len=4096,
    max_num_seqs=16
)
sampling_params = SamplingParams(temperature=0.0, max_tokens=30, stop=["\n", "Answer"])
translation_params = SamplingParams(temperature=0.0, max_tokens=100, stop=["\n", "Hindi:", "English:", "Example:"])


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


# ---- NEW: match prediction against a candidate list ----
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

    base_prompt = "Answer with a short factual phrase in Hindi using Devanagari script only. Do not use English.\n\n"

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

        # ---- Call 1: Translate Hindi question to English ----
        test_question = ex["template"].replace("<subject>", ex["subject"]).replace("<mask>", "").strip()
        translation_prompt = (
            "Translate the following Hindi question into fluent English.\n\n"
            "Rules:\n"
            "- Output MUST be in English only\n"
            "- Preserve proper nouns (names of people, places, organisations) as-is\n"
            "- Keep the question structure intact\n\n"
            "Examples:\n"
            "Hindi: दिल्ली भारत की राजधानी क्या है?\n"
            "English: What is the capital of India, Delhi?\n\n"
            "Hindi: आइंस्टीन का पेशा क्या था?\n"
            "English: What was Einstein's occupation?\n\n"
            "Hindi: मासूम XI की मृत्यु कहाँ हुई थी?\n"
            "English: Where did Masoom XI die?\n\n"
            f"Hindi: {test_question}\n"
            "English:"
        )
        translation_output = llm.generate([translation_prompt], translation_params)
        english_question   = translation_output[0].outputs[0].text.strip().split('\n')[0].strip()

        # ---- Build few-shot prompt ----
        # ---- CHANGE: if object_candidates exist, append them to the prompt ----
        object_candidates = parse_candidates(ex.get("object_candidates"))

        few_shot_prompt = base_prompt
        for d in demonstrations:
            demo_question = d["template"].replace("<subject>", d["subject"]).replace("<mask>", "").strip()
            demo_question = demo_question.replace("उत्तर है:", "").replace("?", "").strip()
            few_shot_prompt += f"Q: {demo_question}\nA: {d['object']}\n\n"

        english_clean = english_question.replace("Answer is:", "").replace("?", "").strip()

        if object_candidates:
            # Present a constrained multiple-choice style prompt
            candidates_str = ", ".join(object_candidates)
            few_shot_prompt += (
                f"Q: {english_clean}\n"
                f"Candidates: {candidates_str}\n"
                f"A:"
            )
        else:
            # No candidates — open generation as before
            few_shot_prompt += f"Q: {english_clean}\nA:"

        prompts.append(few_shot_prompt)
        prompt_metadata.append((ex, lang, index, english_question, object_candidates))

        if len(prompts) == batch_size or idx == len(test_data) - 1:
            outputs = llm.generate(prompts, sampling_params)

            for i, output in enumerate(outputs):
                ex, lang, index, english_question, object_candidates = prompt_metadata[i]
                raw_prediction = output.outputs[0].text

                target   = ex["object"].strip()
                question = ex["template"].replace("<subject>", ex["subject"]).replace("<mask>", "").strip()

                lines      = raw_prediction.strip().split('\n')
                prediction = lines[0].strip()

                # ---- CHANGE: use candidate-aware matching when candidates are available ----
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
                    "question_hindi":     question,
                    "english_question":   english_question,
                    "model_prediction":   prediction,
                    # ---- CHANGE: extra fields for candidate-based evaluation ----
                    "matched_candidate":  matched_candidate,
                    "object_candidates":  object_candidates if object_candidates else None,
                    "used_candidates":    bool(object_candidates),
                    "ground_truth":       target,
                    "is_correct":         bool(match),
                    "raw_output":         raw_prediction
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
                        "question_hindi":     question,
                        "english_question":   english_question,
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
                print(f"   Hindi Q:         {question}")
                print(f"   English Q:       {english_question}")
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
FILE: Evaluation-Scripts/Hindi_preds/filter_knowns.py
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
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
parser.add_argument("--seed", type=int, default=12345)
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
languages = {
    # "meta-llama/Llama-3.2-1B": [ "en","hin","hinglish_dev"],

    "Qwen/Qwen2.5-0.5B":  ["hin"],
    "Qwen/Qwen2.5-1.5B":  ["hin"],
    "Qwen/Qwen2.5-3B":    ["hin"],
    "meta-llama/Llama-3.2-1B": ["hin"],
    "meta-llama/Llama-3.2-3B": ["hin"],


    # "meta-llama/Llama-3.1-8B": ["hin"],
    # "Qwen/Qwen2.5-7B": ["hin"],
    # "meta-llama/Llama-2-7b-hf": ["hi","hinglish", "en", "bn-eng"], #["ca", "en", "es", "fr", "hu", "ja", "ko", "nl", "ru", "uk", "vi", "zh"],
    # "bigscience/bloom-560m": ["hinglish"] #, "ca", "en", "es", "fr", "vi", "zh"]
}
relations = [
    "applies_to_jurisdiction", "capital", "capital_of", "continent", \
    "country_of_citizenship", "developer", "field_of_work", "headquarters_location", \
    "instrument", "language_of_work_or_name", "languages_spoken", "location_of_formation", \
    "manufacturer", "native_language", "occupation", "official_language", \
    "owned_by", "place_of_birth", "place_of_death","religion"
]
valid_langs = set(languages[model_name])
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
    gpu_memory_utilization=0.9,   # you already bumped this to 0.9
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

# def evaluate(llm, dataset, max_new_tokens=10, n_shot=3, save_path="filter_knowns/results.json"):
#     test_data = list(dataset)
#     correct_total = 0
#     total_total = 0
#     per_lang_results = defaultdict(lambda: {"correct": 0, "total": 0, "correct_indices": []})

#     # Pre-group candidates by (relation, language) for O(1) lookup
#     candidates_by_key = defaultdict(list)
#     for i, ex in enumerate(test_data):
#         key = (ex.get("relation"), ex.get("language"))
#         candidates_by_key[key].append((i, ex))

#     print(f"\n[Evaluating {len(test_data)} examples with {n_shot}-shot prompts]")
    
#     # Batch processing
#     prompts = []
#     prompt_metadata = []
#     batch_size = 16

#     for idx, ex in enumerate(tqdm(test_data)):
#         lang = ex.get("language", "unknown")
#         relation = ex.get("relation", "unknown")
#         index = ex.get("index", None)
        
#         # Fast candidate lookup (not O(n) anymore)
#         key = (relation, lang)
#         candidates = [c[1] for c in candidates_by_key[key] if c[0] != idx]
#         demonstrations = random.sample(candidates, min(n_shot, len(candidates)))

#         few_shot_prompt = "".join([f"{d['input']}{d['target']}\n" for d in demonstrations]) + ex["input"]
#         prompts.append(few_shot_prompt)
#         prompt_metadata.append((ex, few_shot_prompt, lang, index))

#         # Process batch when full or at end
#         if len(prompts) == batch_size or idx == len(test_data) - 1:
#             # vLLM batch inference
#             outputs = llm.generate(prompts, sampling_params)

#             for i, output in enumerate(outputs):
#                 ex, few_shot_prompt, lang, index = prompt_metadata[i]
#                 prediction = output.outputs[0].text.split('\n')[0].strip()
#                 target = ex["target"]

#                 match = is_nontrivial_prefix(prediction, target) or is_nontrivial_prefix(target, prediction)
#                 correct_total += match
#                 total_total += 1
#                 per_lang_results[lang]["correct"] += match
#                 per_lang_results[lang]["total"] += 1
#                 if match:
#                     per_lang_results[lang]["correct_indices"].append(index)

#             prompts = []
#             prompt_metadata = []

#     overall_acc = correct_total / total_total if total_total > 0 else 0
#     print(f"\n📊 Overall Accuracy: {overall_acc:.2%}")

#     results = {"overall_acc": overall_acc, "overall_clc": None, "per_language_acc": {}, "per_language_clc": {}}

#     for lang, res in sorted(per_lang_results.items()):
#         lang_acc = res["correct"] / res["total"] if res["total"] > 0 else 0
#         results["per_language_acc"][lang] = lang_acc
#         print(f"  {lang}: {lang_acc:.2%} ({res['correct']} / {res['total']})")

#     # Compute cross-lingual consistency
#     langs = sorted(list(per_lang_results.keys()))
#     for lang in langs:
#         others = [l for l in langs if l != lang]
#         scores = [overlapping_ratio(per_lang_results[lang]["correct_indices"], per_lang_results[other]["correct_indices"]) for other in others]
#         consistency = sum(scores) / len(scores) if scores else 0
#         results["per_language_clc"][lang] = consistency
        
#     results["overall_clc"] = sum(results["per_language_clc"].values()) / len(results["per_language_clc"].values())
    
#     print(f'\n📊 Overall CLC: {results["overall_clc"]:.2%}')
#     for lang in langs:
#         print(f'  {lang} cross-lingual consistency: {results["per_language_clc"][lang]:.2%}')
        
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     with open(save_path, "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=4, ensure_ascii=False)

#     return results
OUTPUT_DIR = "filter_knowns_live"  # ← add this near the top of the file

def evaluate(llm, dataset, max_new_tokens=10, n_shot=3, save_path="filter_knowns/results.json"):
    test_data = list(dataset)
    correct_total = 0
    total_total = 0
    per_lang_results = defaultdict(lambda: {"correct": 0, "total": 0, "correct_indices": []})
    detailed_results = []

    model_safe_name = model_name.replace("/", "_")
    live_dir = os.path.join(OUTPUT_DIR, model_safe_name)
    os.makedirs(live_dir, exist_ok=True)
    live_path = os.path.join(live_dir, "LIVE.json")
    detailed_path = os.path.join(live_dir, "detailed.json")

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
    batch_size = 1

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

    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=4, ensure_ascii=False)

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
FILE: Evaluation-Scripts/Hindi_preds/filter_knowns_with_obj.py
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
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
parser.add_argument("--seed", type=int, default=12345)
args = parser.parse_args()

model_name = args.model_name

# ==== Set random seed ====
def set_seed(seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)

# ==== Define valid languages and relations ====
languages = {
    # "meta-llama/Llama-3.2-1B": ["hin"],
    "meta-llama/Llama-3.1-8B": ["en","hinglish_dev"],
    "Qwen/Qwen2.5-7B": ["en","hinglish_dev"],
}
relations = [
    "applies_to_jurisdiction", "capital", "capital_of", "continent",
    "country_of_citizenship", "developer", "field_of_work", "headquarters_location",
    "instrument", "language_of_work_or_name", "languages_spoken", "location_of_formation",
    "manufacturer", "native_language", "occupation", "official_language",
    "owned_by", "place_of_birth", "place_of_death", "religion"
]
valid_langs = set(languages[model_name])
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
    gpu_memory_utilization=0.25,
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
OUTPUT_DIR = "filter_knowns_live_with_obj"

def evaluate(llm, dataset, max_new_tokens=10, n_shot=3, save_path="filter_knowns/results.json"):
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
    batch_size = 1

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



================================================
FILE: Evaluation-Scripts/Hindi_preds/implicit_in_mind_CM.py
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
OUTPUT_DIR = "filter_knowns-trans-hinglish-prompt-implicit-in-mind"
# ======================================

# ==== Argument parser ====
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
parser.add_argument("--seed", type=int, default=12345)
args = parser.parse_args()

model_name = args.model_name

# ==== Set random seed ====
def set_seed(seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)

# ==== Define valid languages and relations ====
languages = {
    # "meta-llama/Llama-3.2-1B": ["hin"],
    "Qwen/Qwen2.5-0.5B":  ["hin"],
    "Qwen/Qwen2.5-1.5B":  ["hin"],
    "Qwen/Qwen2.5-3B":    ["hin"],
    "meta-llama/Llama-3.2-1B": ["hin"],
    "meta-llama/Llama-3.2-3B": ["hin"],
    # "meta-llama/Llama-3.1-8B": ["hin"],
    # "Qwen/Qwen2.5-7B": ["hin"],
}
relations = [
    "applies_to_jurisdiction", "capital", "capital_of", "continent",
    "country_of_citizenship", "developer", "field_of_work", "headquarters_location",
    "instrument", "language_of_work_or_name", "languages_spoken", "location_of_formation",
    "manufacturer", "native_language", "occupation", "official_language",
    "owned_by", "place_of_birth", "place_of_death", "religion"
]
valid_langs = set(languages[model_name])
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

dataset = Dataset.from_list(samples)

# ==== Model loading ====
llm = LLM(
    model=model_name,
    trust_remote_code=True,
    gpu_memory_utilization=0.9,
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


# ==== Hinglish translation helper ====
# Instead of asking the LLM to generate the translation, we provide a
# pre-built Hinglish form of the question in the prompt context.
# This is a lightweight approximation: keep grammar particles in Roman,
# replace content words with English equivalents.
# The subject is already in English/Roman in the KLAR data, so we only
# need to map the relation-specific Hindi phrasing.
RELATION_HINGLISH_TEMPLATES = {
    "applies_to_jurisdiction":    "{subject} ko kis desh mein legal term ke roop mein maana jaata hai?",
    "capital":                    "{subject} ki capital kya hai?",
    "capital_of":                 "{subject} kis desh ki capital hai?",
    "continent":                  "{subject} kis continent mein sthit hai?",
    "country_of_citizenship":     "{subject} ki citizenship kahan ki hai?",
    "developer":                  "{subject} ka developer kaun hai?",
    "field_of_work":              "{subject} ka field of work kya hai?",
    "headquarters_location":      "{subject} ka headquarters kahan sthit hai?",
    "instrument":                 "{subject} kaun sa instrument bajata hai?",
    "language_of_work_or_name":   "{subject} ki language of work ya name kya hai?",
    "languages_spoken":           "{subject} kaun si languages bolta/bolti hai?",
    "location_of_formation":      "{subject} ka formation kahan hua tha?",
    "manufacturer":               "{subject} ka manufacturer kaun hai?",
    "native_language":            "{subject} ki native language kya hai?",
    "occupation":                 "{subject} ka occupation kya hai?",
    "official_language":          "{subject} ki official language kya hai?",
    "owned_by":                   "{subject} ka owner kaun hai?",
    "place_of_birth":             "{subject} ka place of birth kahan hai?",
    "place_of_death":             "{subject} ki death kahan hui thi?",
    "religion":                   "{subject} ka religion kya hai?",
}

def build_hinglish_translation(subject: str, relation: str) -> str:
    """
    Build a Hinglish approximation of the Hindi question using static templates.
    This is passed into the prompt as context, NOT generated by the LLM.
    """
    tmpl = RELATION_HINGLISH_TEMPLATES.get(relation)
    if tmpl:
        return tmpl.format(subject=subject)
    # Fallback: generic form
    rel_words = relation.replace("_", " ")
    return f"{subject} ka {rel_words} kya hai?"


# ==== Static preamble ====
# The model is instructed to perform implicit Hinglish translation as an internal
# mental step — it mentally converts the Hindi question to Hinglish to understand it,
# but NEVER outputs the translation. The JSON schema enforces only {"answer": "..."}.
SYSTEM_PREAMBLE = (
    "You are answering factual questions written in Hindi (Devanagari script).\n"
    "Step 1 (internal only): Mentally translate the Hindi question into Hinglish "
    "(Hindi grammar + English content words) to understand it clearly. "
    "Do NOT write this translation in your output.\n"
    "Step 2: Based on your understanding, output ONLY a JSON object with a single field:\n"
    "  \"answer\": the correct answer chosen from the provided candidates.\n\n"
    "Rules:\n"
    "- The answer MUST be one of the listed candidates.\n"
    "- Do NOT write the Hinglish translation or any explanation in your output — only the JSON.\n\n"
)


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
        # Structure:
        #   [SYSTEM_PREAMBLE]    ← explains Hinglish concept, task instructions
        #   [n-shot QA demos]    ← each demo shows: Hindi Q + Hinglish (as context) + answer JSON
        #   [test question]      ← Hindi Q + Hinglish (as context) + candidates → model outputs {"answer": ...}

        prompt = SYSTEM_PREAMBLE

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
        test_hinglish = build_hinglish_translation(ex["subject"], relation)  # for logging only
        candidates_str = ", ".join(object_candidates) if object_candidates else ""

        # Only the Hindi question and candidates are shown — no Hinglish in prompt.
        prompt += (
            f"Q: {test_question}\n"
            f"Candidates: {candidates_str}\n"
            f"Output:"
        )

        prompts.append(prompt)
        # Store the pre-built hinglish alongside other metadata for logging
        prompt_metadata.append((ex, lang, index, object_candidates, prompt, test_hinglish))

        if len(prompts) == batch_size or idx == len(test_data) - 1:
            outputs = llm.generate(prompts, sampling_params)

            for i, output in enumerate(outputs):
                ex, lang, index, object_candidates, final_prompt, hinglish_question = prompt_metadata[i]
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
                    "question_hindi":     question,
                    "hinglish_question":  hinglish_question,  # pre-built, provided as context (not generated)
                    "model_prediction":   prediction,         # from "answer" field (Hindi Devanagari)
                    "matched_candidate":  matched_candidate,
                    "object_candidates":  object_candidates if object_candidates else None,
                    "used_candidates":    bool(object_candidates),
                    "ground_truth":       target,
                    "is_correct":         bool(match),
                    "raw_output":         raw_prediction,
                    "final_prompt": final_prompt
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
                        "question_hindi":     question,
                        "hinglish_question":  hinglish_question,
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
                print(f"   Hindi Q:         {question}")
                print(f"   Hinglish (ctx):  {hinglish_question}")
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
FILE: Evaluation-Scripts/Hindi_preds/Implicit_in_mind_EN.py
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
OUTPUT_DIR = "filter_knowns-trans-english-prompt-implicit-in-mind"
# ======================================

# ==== Argument parser ====
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
parser.add_argument("--seed", type=int, default=12345)
args = parser.parse_args()

model_name = args.model_name

# ==== Set random seed ====
def set_seed(seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)

# ==== Define valid languages and relations ====
languages = {
    # "meta-llama/Llama-3.2-1B": ["hin"],
    "Qwen/Qwen2.5-0.5B":  ["hin"],
    "Qwen/Qwen2.5-1.5B":  ["hin"],
    "Qwen/Qwen2.5-3B":    ["hin"],
    "meta-llama/Llama-3.2-1B": ["hin"],
    "meta-llama/Llama-3.2-3B": ["hin"],
    # "meta-llama/Llama-3.1-8B": ["hin"],
    # "Qwen/Qwen2.5-7B": ["hin"],
}
relations = [
    "applies_to_jurisdiction", "capital", "capital_of", "continent",
    "country_of_citizenship", "developer", "field_of_work", "headquarters_location",
    "instrument", "language_of_work_or_name", "languages_spoken", "location_of_formation",
    "manufacturer", "native_language", "occupation", "official_language",
    "owned_by", "place_of_birth", "place_of_death", "religion"
]
valid_langs = set(languages[model_name])
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

dataset = Dataset.from_list(samples)

# ==== Model loading ====
llm = LLM(
    model=model_name,
    trust_remote_code=True,
    gpu_memory_utilization=0.9,
    max_model_len=4096,
    max_num_seqs=16
)

# ==== Guided JSON schema ====
# The model outputs ONLY the answer field.
# English translation is NOT in the prompt at all — the model processes Hindi
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


# ==== English translation helper ====
# Instead of asking the LLM to generate the translation, we provide a
# pre-built English form of the question in the prompt context (for logging/analysis).
# The model is instructed to mentally translate Hindi to English internally.
# The subject is already in English/Roman in the KLAR data, so we only
# need to map the relation-specific Hindi phrasing to English equivalents.
RELATION_ENGLISH_TEMPLATES = {
    "applies_to_jurisdiction":    "In which country is {subject} recognized as a legal term?",
    "capital":                    "What is the capital of {subject}?",
    "capital_of":                 "Which country is {subject} the capital of?",
    "continent":                  "On which continent is {subject} located?",
    "country_of_citizenship":     "What is the country of citizenship of {subject}?",
    "developer":                  "Who is the developer of {subject}?",
    "field_of_work":              "What is the field of work of {subject}?",
    "headquarters_location":      "Where is the headquarters of {subject} located?",
    "instrument":                 "What instrument does {subject} play?",
    "language_of_work_or_name":   "What is the language of work or name of {subject}?",
    "languages_spoken":           "What languages does {subject} speak?",
    "location_of_formation":      "Where was {subject} formed?",
    "manufacturer":               "Who is the manufacturer of {subject}?",
    "native_language":            "What is the native language of {subject}?",
    "occupation":                 "What is the occupation of {subject}?",
    "official_language":          "What is the official language of {subject}?",
    "owned_by":                   "Who owns {subject}?",
    "place_of_birth":             "Where was {subject} born?",
    "place_of_death":             "Where did {subject} die?",
    "religion":                   "What is the religion of {subject}?",
}

def build_english_translation(subject: str, relation: str) -> str:
    """
    Build an English translation of the Hindi question using static templates.
    This is used for logging/analysis only, NOT generated by the LLM.
    """
    tmpl = RELATION_ENGLISH_TEMPLATES.get(relation)
    if tmpl:
        return tmpl.format(subject=subject)
    # Fallback: generic form
    rel_words = relation.replace("_", " ")
    return f"What is the {rel_words} of {subject}?"


# ==== Static preamble ====
# The model is instructed to perform implicit English translation as an internal
# mental step — it mentally converts the Hindi question to English to understand it,
# but NEVER outputs the translation. The JSON schema enforces only {"answer": "..."}.
SYSTEM_PREAMBLE = (
    "You are answering factual questions written in Hindi (Devanagari script).\n"
    "Step 1 (internal only): Mentally translate the Hindi question into English "
    "to understand it clearly. "
    "Do NOT write this translation in your output.\n"
    "Step 2: Based on your understanding, output ONLY a JSON object with a single field:\n"
    "  \"answer\": the correct answer chosen from the provided candidates.\n\n"
    "Rules:\n"
    "- The answer MUST be one of the listed candidates.\n"
    "- Do NOT write the English translation or any explanation in your output — only the JSON.\n\n"
)


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
        # Structure:
        #   [SYSTEM_PREAMBLE]    ← explains English translation concept, task instructions
        #   [n-shot QA demos]    ← each demo shows: Hindi Q + English (as context) + answer JSON
        #   [test question]      ← Hindi Q + English (as context) + candidates → model outputs {"answer": ...}

        prompt = SYSTEM_PREAMBLE

        for d in demonstrations:
            demo_question = d["template"].replace("<subject>", d["subject"]).replace("<mask>", "").strip()
            demo_candidates = parse_candidates(d.get("object_candidates"))
            demo_cands_str = ", ".join(demo_candidates) if demo_candidates else d["object"]

            # English translation is built for logging/analysis only — NOT shown in the prompt.
            # The model sees only the Hindi question and must answer implicitly.
            prompt += (
                f"Q: {demo_question}\n"
                f"Candidates: {demo_cands_str}\n"
                f"Output: {{\"answer\": \"{d['object']}\"}}\n\n"
            )

        # ---- Test question (always with candidates) ----
        test_question = ex["template"].replace("<subject>", ex["subject"]).replace("<mask>", "").strip()
        test_english = build_english_translation(ex["subject"], relation)  # for logging only
        candidates_str = ", ".join(object_candidates) if object_candidates else ""

        # Only the Hindi question and candidates are shown — no English in prompt.
        prompt += (
            f"Q: {test_question}\n"
            f"Candidates: {candidates_str}\n"
            f"Output:"
        )

        prompts.append(prompt)
        # Store the pre-built hinglish alongside other metadata for logging
        prompt_metadata.append((ex, lang, index, object_candidates, prompt, test_english))

        if len(prompts) == batch_size or idx == len(test_data) - 1:
            outputs = llm.generate(prompts, sampling_params)

            for i, output in enumerate(outputs):
                ex, lang, index, object_candidates, final_prompt, english_question = prompt_metadata[i]
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
                    "question_hindi":     question,
                    "english_question":   english_question,  # pre-built, provided for logging (not generated)
                    "model_prediction":   prediction,         # from "answer" field (Hindi Devanagari)
                    "matched_candidate":  matched_candidate,
                    "object_candidates":  object_candidates if object_candidates else None,
                    "used_candidates":    bool(object_candidates),
                    "ground_truth":       target,
                    "is_correct":         bool(match),
                    "raw_output":         raw_prediction,
                    "final_prompt": final_prompt
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
                        "question_hindi":     question,
                        "english_question":   english_question,
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
                print(f"   Hindi Q:         {question}")
                print(f"   English (ctx):   {english_question}")
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
FILE: Evaluation-Scripts/Hindi_preds/implicit_trans_CM.py
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
OUTPUT_DIR = "filter_knowns_implicit-trans-hinglish"
# ======================================

# ==== Argument parser ====
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B")
parser.add_argument("--seed", type=int, default=12345)
args = parser.parse_args()

model_name = args.model_name

# ==== Set random seed ====
def set_seed(seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)

# ==== Define valid languages and relations ====
languages = {
    # "meta-llama/Llama-3.2-1B": ["hin"],
    "Qwen/Qwen2.5-0.5B":  ["hin"],
    "Qwen/Qwen2.5-1.5B":  ["hin"],
    "Qwen/Qwen2.5-3B":    ["hin"],
    "meta-llama/Llama-3.2-1B": ["hin"],
    "meta-llama/Llama-3.2-3B": ["hin"],
    # "meta-llama/Llama-3.1-8B": ["hin"],
    # "Qwen/Qwen2.5-7B": ["hin"],
}
relations = [
    "applies_to_jurisdiction", "capital", "capital_of", "continent",
    "country_of_citizenship", "developer", "field_of_work", "headquarters_location",
    "instrument", "language_of_work_or_name", "languages_spoken", "location_of_formation",
    "manufacturer", "native_language", "occupation", "official_language",
    "owned_by", "place_of_birth", "place_of_death", "religion"
]
valid_langs = set(languages[model_name])
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

dataset = Dataset.from_list(samples)

# ==== Model loading ====
llm = LLM(
    model=model_name,
    trust_remote_code=True,
    gpu_memory_utilization=0.9,
    max_model_len=4096,
    max_num_seqs=16
)

# ==== Guided JSON schema ====
# The model outputs a two-field JSON in a single call:
#   "translation" — Hinglish transliteration of the Hindi question (for logging only)
#   "answer"      — Hindi answer in Devanagari, selected from object_candidates
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

    # ==== CHANGED: System-level instruction now asks for Hinglish transliteration ====
# ==== System-level instruction asking for Hinglish codemixed form ====
    base_prompt = (
        "You are a multilingual assistant. For each Hindi question, output a JSON object with exactly two fields:\n"
        "  \"translation\": the Hinglish form of the Hindi question. Hinglish is a mix of Hindi and English — write the sentence using Roman/Latin letters, and replace Hindi content words with their English meaning while keeping Hindi grammar words (like ka, ki, ke, kya, kahan, kab, tha, hai) as they are in Roman script.\n"
        "  \"answer\": the correct answer in Hindi Devanagari script, chosen from the provided candidates.\n\n"
        "Rules:\n"
        "- The answer MUST be one of the listed candidates.\n"
        "- The answer MUST be written in Hindi Devanagari script only.\n"
        "- The translation MUST be Hinglish (Hindi + English mix in Roman script, e.g. 'Einstein ka occupation kya tha?', 'Dilli Bharat ki capital kya hai?', 'Masoom XI ki death kahan hui thi?').\n"
        "- Do NOT use English in the answer field.\n\n"
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
        # Each demo shows a Hindi question + candidates → JSON with Hinglish transliteration + Hindi answer.
        prompt = base_prompt
        for d in demonstrations:
            demo_question = d["template"].replace("<subject>", d["subject"]).replace("<mask>", "").strip()
            demo_candidates = parse_candidates(d.get("object_candidates"))
            demo_cands_str = ", ".join(demo_candidates) if demo_candidates else d["object"]
            # ==== CHANGED: fabricated translation is now Hinglish instead of English ====
            prompt += (
                f"Q (Hindi): {demo_question}\n"
                f"Candidates: {demo_cands_str}\n"
                f"Output: {{\"translation\": \"{d['subject']} ka {d['relation'].replace('_', ' ')} kya hai?\", \"answer\": \"{d['object']}\"}}\n\n"
            )

        # ---- Test question (always with candidates) ----
        test_question = ex["template"].replace("<subject>", ex["subject"]).replace("<mask>", "").strip()
        candidates_str = ", ".join(object_candidates) if object_candidates else ""

        prompt += (
            f"Q (Hindi): {test_question}\n"
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
                hinglish_question, prediction = parse_json_output(raw_prediction)
                # prediction     = the "answer" field (Hindi Devanagari)
                # hinglish_question = the "translation" field (Hinglish, logged only)

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
                    "question_hindi":     question,
                    "hinglish_question":  hinglish_question,  # from "translation" field (Hinglish, logging only)
                    "model_prediction":   prediction,         # from "answer" field (Hindi Devanagari)
                    "matched_candidate":  matched_candidate,
                    "object_candidates":  object_candidates if object_candidates else None,
                    "used_candidates":    bool(object_candidates),
                    "ground_truth":       target,
                    "is_correct":         bool(match),
                    "raw_output":         raw_prediction,
                    "final_prompt":        final_prompt
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
                        "question_hindi":     question,
                        "hinglish_question":  hinglish_question,
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
                print(f"   Hindi Q:         {question}")
                print(f"   Hinglish:        {hinglish_question}")
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
FILE: Evaluation-Scripts/Hindi_preds/implicit_trans_EN.py
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
OUTPUT_DIR = "filter_knowns_implicit-trans-en"
# ======================================

# ==== Argument parser ====
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
parser.add_argument("--seed", type=int, default=12345)
args = parser.parse_args()

model_name = args.model_name

# ==== Set random seed ====
def set_seed(seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)

# ==== Define valid languages and relations ====
languages = {
    # "meta-llama/Llama-3.2-1B": ["hin"],
    "Qwen/Qwen2.5-0.5B":  ["hin"],
    "Qwen/Qwen2.5-1.5B":  ["hin"],
    "Qwen/Qwen2.5-3B":    ["hin"],
    "meta-llama/Llama-3.2-1B": ["hin"],
    "meta-llama/Llama-3.2-3B": ["hin"],
    # "meta-llama/Llama-3.1-8B": ["hin"],
    # "Qwen/Qwen2.5-7B": ["hin"],
}
relations = [
    "applies_to_jurisdiction", "capital", "capital_of", "continent",
    "country_of_citizenship", "developer", "field_of_work", "headquarters_location",
    "instrument", "language_of_work_or_name", "languages_spoken", "location_of_formation",
    "manufacturer", "native_language", "occupation", "official_language",
    "owned_by", "place_of_birth", "place_of_death", "religion"
]
valid_langs = set(languages[model_name])
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

dataset = Dataset.from_list(samples)

# ==== Model loading ====
llm = LLM(
    model=model_name,
    trust_remote_code=True,
    gpu_memory_utilization=0.9,
    max_model_len=4096,
    max_num_seqs=16
)

# ==== Guided JSON schema ====
# The model outputs a two-field JSON in a single call:
#   "translation" — English rendering of the Hindi question (for logging only)
#   "answer"      — Hindi answer in Devanagari, selected from object_candidates
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

    # System-level instruction for the single combined call
    base_prompt = (
        "You are a multilingual assistant. For each Hindi question, output a JSON object with exactly two fields:\n"
        "  \"translation\": the English translation of the Hindi question (for reference only),\n"
        "  \"answer\": the correct answer in Hindi Devanagari script, chosen from the provided candidates.\n\n"
        "Rules:\n"
        "- The answer MUST be one of the listed candidates.\n"
        "- The answer MUST be written in Hindi Devanagari script only.\n"
        "- Do NOT use English in the answer field.\n\n"
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
        # Each demo shows a Hindi question + candidates → JSON with translation + Hindi answer.
        prompt = base_prompt
        for d in demonstrations:
            demo_question = d["template"].replace("<subject>", d["subject"]).replace("<mask>", "").strip()
            demo_candidates = parse_candidates(d.get("object_candidates"))
            demo_cands_str = ", ".join(demo_candidates) if demo_candidates else d["object"]
            prompt += (
                f"Q (Hindi): {demo_question}\n"
                f"Candidates: {demo_cands_str}\n"
                # f"Output: {{\"translation\": \"(English translation)\", \"answer\": \"{d['object']}\"}}\n\n"
                f"Output: {{\"translation\": \"What is the {d['relation'].replace('_', ' ')} of {d['subject']}?\", \"answer\": \"{d['object']}\"}}\n\n"
            )

        # ---- Test question (always with candidates) ----
        test_question = ex["template"].replace("<subject>", ex["subject"]).replace("<mask>", "").strip()
        candidates_str = ", ".join(object_candidates) if object_candidates else ""

        prompt += (
            f"Q (Hindi): {test_question}\n"
            f"Candidates: {candidates_str}\n"
            f"Output:"
        )

        prompts.append(prompt)
        prompt_metadata.append((ex, lang, index, object_candidates))

        if len(prompts) == batch_size or idx == len(test_data) - 1:
            outputs = llm.generate(prompts, sampling_params)

            for i, output in enumerate(outputs):
                ex, lang, index, object_candidates = prompt_metadata[i]
                raw_prediction = output.outputs[0].text

                target   = ex["object"].strip()
                question = ex["template"].replace("<subject>", ex["subject"]).replace("<mask>", "").strip()

                # ---- Parse the single JSON output ----
                english_question, prediction = parse_json_output(raw_prediction)
                # prediction = the "answer" field (Hindi Devanagari)
                # english_question = the "translation" field (logged only)

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
                    "question_hindi":     question,
                    "english_question":   english_question,   # from "translation" field (logging only)
                    "model_prediction":   prediction,         # from "answer" field (Hindi Devanagari)
                    "matched_candidate":  matched_candidate,
                    "object_candidates":  object_candidates if object_candidates else None,
                    "used_candidates":    bool(object_candidates),
                    "ground_truth":       target,
                    "is_correct":         bool(match),
                    "raw_output":         raw_prediction
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
                        "question_hindi":     question,
                        "english_question":   english_question,
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
                print(f"   Hindi Q:         {question}")
                print(f"   Translation:     {english_question}")
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



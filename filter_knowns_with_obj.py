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

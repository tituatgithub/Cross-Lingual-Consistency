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
OUTPUT_DIR = "2_call_en_placeholder_corr"
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

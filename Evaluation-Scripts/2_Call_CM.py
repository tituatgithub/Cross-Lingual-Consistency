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
OUTPUT_DIR      = "2_Call_CM"
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
    gpu_memory_utilization=0.25,
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

    stage1_base_prompt = (
        f"You are a translator. Convert this {SOURCE_LANG} ({SOURCE_SCRIPT} script) question "
        f"into {TARGET_LANG} ({SOURCE_LANG} grammar + English content words in Roman/Latin script).\n\n"
        f"Rules:\n"
        f"- Preserve {SOURCE_LANG} sentence structure and grammar markers\n"
        f"- Replace only content words (nouns, verbs, adjectives) with English equivalents\n"
        f"- Write output in Roman/Latin script only\n"
        f"- Output ONLY: {{\"translation\": \"...\"}}\n\n"
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

    print(f"\n[Evaluating {len(test_data)} examples with {n_shot}-shot prompts]")
    print(f"📁 Output directory: {output_subdir}")
    print(f"💡 Tip: Run 'tail -f {live_path}' in another terminal to watch progress\n")



    # ==== TWO-STAGE PIPELINE SETUP ====
    stage1_prompts         = []  # Hindi → Hinglish translation prompts
    stage1_metadata        = []  # (ex, lang, index, original_question)
    stage2_prompts         = []  # Hinglish → Hindi answer prompts  
    stage2_metadata        = []  # (ex, lang, index, object_candidates, hinglish_q, original_q, stage1_prompt, stage1_raw)
    
    batch_size      = 1

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
        
        # Add few-shot examples from dataset (source language questions, like Script 2)
        key = (relation, lang)
        candidates_pool = [c[1] for c in candidates_by_key[key] if c[0] != idx]
        if candidates_pool:
            demonstrations = random.sample(candidates_pool, min(n_shot, len(candidates_pool)))
            for d in demonstrations:
                demo_question = d["template"].replace("<subject>", d["subject"]).replace("<mask>", "").strip()
                s1_prompt += f"Q: {demo_question}\nOutput:\n\n"
        
        s1_prompt += f"Q: {test_question}\nOutput:"

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

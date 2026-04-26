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
parser.add_argument("--batch_size",             type=int,   default=8,    help="Number of prompts per vLLM batch")
parser.add_argument("--gpu_memory_utilization", type=float, default=0.20, help="Fraction of GPU memory vLLM may use (0.0–1.0)")
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
    gpu_memory_utilization=args.gpu_memory_utilization,
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
    batch_size = args.batch_size

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

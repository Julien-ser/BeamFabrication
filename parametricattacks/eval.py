import json
from typing import List, Dict
from tqdm import tqdm

import re
from collections import Counter

def normalize_text(text: str) -> str:
    """Lowercase, remove punctuation, and extra whitespace."""
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)  # Remove articles
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

def compute_subspan_exact_match(prediction: str, ground_truth: str) -> bool:
    """Check if the normalized prediction is a substring of the normalized ground truth or vice versa."""
    if normalize_text(prediction) == "" or normalize_text(ground_truth) == "":
        return False
    return (normalize_text(prediction) in normalize_text(ground_truth)) or (normalize_text(ground_truth) in normalize_text(prediction))

def compute_exact_match(prediction: str, ground_truth: str) -> bool:
    """Check if the normalized prediction exactly matches the normalized ground truth."""
    return normalize_text(prediction) == normalize_text(ground_truth)

def single_f1(pred, gt):
    pred_tokens = normalize_text(pred).split()
    gt_tokens = normalize_text(gt).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * (precision * recall) / (precision + recall)

check_path = "check.jsonl"

#clean_path = "Mixtral_responses_clean.jsonl"
#mixed_path = "Mixtral_responses_mixed.jsonl"
#fab_path = "Mixtral_responses_fab.jsonl"
shuffled_path = "llama2_70b_responses_shuffled.jsonl"

id_to_obj = {}
with open(check_path, "r") as infile:
    for line in infile:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        qid = obj["question_id"]
        id_to_obj[qid] = obj
'''
with open(clean_path, "r") as infile:
    em_count = 0
    sem_count = 0
    f1_score = 0
    for line in infile:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        qid = obj["question_id"]
        gen_answer = normalize_text(obj["answer"])
        if qid in id_to_obj:
            original_answer = id_to_obj[qid]["original_answers"][0]
            if compute_exact_match(gen_answer, original_answer):
                em_count += 1
            if compute_subspan_exact_match(gen_answer, original_answer):
                sem_count += 1
            f1_score += single_f1(gen_answer, original_answer)
    total = len(id_to_obj)
    print("======================================================CLEAN RESPONSES=======================================================")
    print(f"EM: {em_count}/{total} ({em_count/total:.2%}), SEM: {sem_count}/{total} ({sem_count/total:.2%}), F1: {f1_score/total:.4f}")
    print("=============================================================================================================================")  

with open(mixed_path, "r") as infile:
    em_count = 0
    sem_count = 0
    f1_score = 0
    for line in infile:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        qid = obj["question_id"]
        gen_answer = normalize_text(obj["answer"])
        if qid in id_to_obj:
            original_answer = id_to_obj[qid]["original_answers"][0]
            if compute_exact_match(gen_answer, original_answer):
                em_count += 1
            if compute_subspan_exact_match(gen_answer, original_answer):
                sem_count += 1
            f1_score += single_f1(gen_answer, original_answer)
    total = len(id_to_obj)
    print("======================================================MIXED RESPONSES=======================================================")
    print(f"EM: {em_count}/{total} ({em_count/total:.2%}), SEM: {sem_count}/{total} ({sem_count/total:.2%}), F1: {f1_score/total:.4f}")
    print("=============================================================================================================================")

with open(fab_path, "r") as infile:
    em_count = 0
    sem_count = 0
    f1_score = 0
    for line in infile:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        qid = obj["question_id"]
        gen_answer = normalize_text(obj["answer"])
        if qid in id_to_obj:
            original_answer = id_to_obj[qid]["original_answers"][0]
            if compute_exact_match(gen_answer, original_answer):
                em_count += 1
            if compute_subspan_exact_match(gen_answer, original_answer):
                sem_count += 1
            f1_score += single_f1(gen_answer, original_answer)
    total = len(id_to_obj)
    print("======================================================FABRICATED RESPONSES=======================================================")
    print(f"EM: {em_count}/{total} ({em_count/total:.2%}), SEM: {sem_count}/{total} ({sem_count/total:.2%}), F1: {f1_score/total:.4f}")
    print("=============================================================================================================================")

'''
with open(shuffled_path, "r") as infile:
    em_count = 0
    sem_count = 0
    f1_score = 0
    for line in infile:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        qid = obj["question_id"]
        gen_answer = normalize_text(obj["answer"])
        if qid in id_to_obj:
            original_answer = id_to_obj[qid]["original_answers"][0]
            if compute_exact_match(gen_answer, original_answer):
                em_count += 1
            if compute_subspan_exact_match(gen_answer, original_answer):
                sem_count += 1
            f1_score += single_f1(gen_answer, original_answer)
    total = len(id_to_obj)
    print("======================================================SHUFFLED RESPONSES=======================================================")
    print(f"EM: {em_count}/{total} ({em_count/total:.2%}), SEM: {sem_count}/{total} ({sem_count/total:.2%}), F1: {f1_score/total:.4f}")
    print("=============================================================================================================================")
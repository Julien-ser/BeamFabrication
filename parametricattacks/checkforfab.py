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

hottest_path = "hottest.jsonl"

id_to_obj = {}
with open(hottest_path, "r") as infile:
    for line in infile:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        qid = obj["question_id"]
        id_to_obj[qid] = obj

filepath = "responses.jsonl"

with open(filepath, "r") as infile:
    with open("check.jsonl", "w") as outfile:
        for line in infile:
            line = line.strip()
            obj = json.loads(line)
            try:
                qid = obj["question_id"]
                answer = normalize_text(obj["answer"])
                context = obj["thought"]
                original_answers = id_to_obj[qid]["answers_objects"][0]["spans"]
                f1, sem, em = [], [], []
                for ans in original_answers:
                    ans = normalize_text(ans)
                    f1_score = single_f1(answer, ans)
                    sem_score = compute_subspan_exact_match(answer, ans)
                    em_score = compute_exact_match(answer, ans)
                    if(f1_score < 0.8):
                        if(sem_score == 1):
                            pass
                        else:
                            f1.append(f1_score)
                            sem.append(sem_score)
                            em.append(em_score)
                if(len(f1) != 0):
                    check = {
                        "fabricated_candidate": context,
                        "original_answers": original_answers,
                        "generated_answer": answer,
                        "f1": f1,
                        "subspan_exact_match": sem,
                        "exact_match": em
                    }
                    outfile.write(json.dumps(check) + "\n")
            except:
                line = "============DEFECT============\n" + str(obj) + "\n=============================\n"
                with open("error.txt", "a") as error_file:
                    error_file.write(line + "\n")
                print("ooops")

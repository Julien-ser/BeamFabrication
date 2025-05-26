import json

# Paths to your files
check_path = "check.jsonl"
hottest_path = "hottest.jsonl"
output_path = "hottest_with_fabricated.jsonl"

# Load fabricated_contexts from check.jsonl
qid_to_fabricated = {}
qid_to_ans = {}
with open(check_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        if "question_id" in item and "fabricated_candidate" in item:
            qid_to_fabricated[item["question_id"]] = item["fabricated_candidate"]
        if "question_id" in item and "generated_answer" in item:
            qid_to_ans[item["question_id"]] = item["generated_answer"]
#print(qid_to_fabricated)           
# Process hottest.jsonl and add fabricated_context if qid matches
with open(hottest_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        item = json.loads(line)
        qid = item.get("question_id")
        if qid in qid_to_fabricated:
            #print(f"Processing qid {qid} with fabricated context")
            # Add fabricated_context to contexts
            if "contexts" in item and isinstance(item["contexts"], list):
                idx = len(item["contexts"])
                item["contexts"].append({"idx": idx, "title": qid_to_ans[qid], "paragraph_text": qid_to_fabricated[qid]})
                #print(f"Added fabricated context for qid {qid} at index {idx}")
            else:
                print("??????")
                item["contexts"] = [{"idx": 0, "title": qid_to_ans[qid], "paragraph_text": qid_to_fabricated[qid]}]
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")



import json
import os
import re
import pandas as pd
from ollama import chat
from ollama import ChatResponse
import json
#from transformers import AutoModelForCausalLM, AutoTokenizer
#data_dir = "data"
import subprocess
from tqdm import tqdm
from testing2 import most_similar_neighbor
from testing import text_to_dbpedia_entity
# Paths to your files
hottest_path = "hottest.jsonl"

# Load fabricated_contexts from check.jsonl
qid_to_ans = {}
with open(hottest_path, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)
with open(hottest_path, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Building qid_to_ans", total=total_lines):
        item = json.loads(line)
        entity = item["answers_objects"][0]["spans"][0]
        try:
            swapper = most_similar_neighbor(text_to_dbpedia_entity(entity))
            fabricated_context = swapper.replace("http://dbpedia.org/resource/", "")
            #print(f"Swapped {entity} with {fabricated_context}")
            qid_to_ans[item["question_id"]] = fabricated_context
        except:
            pass
    
with open("qid_to_ans_backup.json", "w", encoding="utf-8") as backup_file:
    json.dump(qid_to_ans, backup_file, ensure_ascii=False, indent=2)
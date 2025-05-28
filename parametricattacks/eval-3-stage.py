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

def ensure_model_pulled(model):
    try:
        # Check if model is already pulled
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if model not in result.stdout:
            #print(f'Model {model} not found locally. Pulling...')
            subprocess.run(['ollama', 'pull', model], check=True)
        else:
            pass
            #print(f'Model {model} already available.')
    except subprocess.CalledProcessError as e:
        print(f"Error pulling model {model}: {e}")

def load_questions_from_jsonl(filepath, is_check=False):
    questions = []
    ids = []
    with open("check.jsonl", "r") as infile:
        for line in infile:
            line = line.strip()
            obj = json.loads(line)
            ids.append(obj["question_id"])
            if is_check:
                questions.append([obj["question_text"], obj["question_id"], obj["fabricated_candidate"]])
    
    if is_check:
        return questions
    with open(filepath, "r") as infile:
        for line in infile:
            line = line.strip()
            obj = json.loads(line)
            contexts = obj.get("contexts", [])
            context = "\n".join([f"{c['title']}: {c['paragraph_text']}" for c in contexts])
            if obj["question_id"] in ids:
                questions.append([obj["question_text"], obj["question_id"], context])

    return questions

def get_responses(model_name):
        # ...existing code for prompt...
    # Replace this with your actual list of questions
    questions = load_questions_from_jsonl("hottest_shuffled.jsonl", False)
    safe_model_name = model_name.replace(":", "_")
    safe_model_name = safe_model_name.replace("/", "_")
    output_path = f"{safe_model_name}_responses_shuffled.jsonl"
    with open(output_path, "a") as outfile:
        for question in tqdm(questions, desc=f"Processing {model_name}"):
            # ...existing code...
            prompt = f"""You are a precise QA assistant. Answer the question using ONLY the given context.
            Output ONLY the final answer as a single line, with no explanations, no extra text, no formatting, and no repeated question or context.
            If the answer is a name or title, output ONLY the name or title.
            If the answer is not present in the context, output exactly: I don't know.
            Do not add any extra words, sentences, or formatting.
            Examples:
            Context: The capital of France is Paris.
            Question: What is the capital of France?
            Answer: Paris

            Context: The text does not mention the winner.
            Question: Who won the match?
            Answer: I don't know

            Context: {question[2]}
            Question: {question[0]}
            Answer:"""
            # ...existing code...
            #print(prompt)
            response: ChatResponse = chat(
                model=model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.7}
            )
            #print("RESPONSE=======================================\n" + response.message.content)
            item = {"answer": response.message.content, "question_id": question[1]}
            outfile.write(json.dumps(item) + "\n")
                 #   response_data = {"thought": "", "answer": ""}
            #else:
             #   response_data = {"thought": "", "answer": ""}


models = [
    "llama2:13b",
]
for model in models:
    ensure_model_pulled(model)
    get_responses(model)
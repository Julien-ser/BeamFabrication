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
            prompt = (
                "You are a question answering assistant.\n"
                "You MUST answer ONLY using the provided context and question below.\n"
                "Do NOT ask the user to provide the context or question. They are already provided.\n"
                "Do NOT say things like 'Sure', 'Please provide', or 'I'm happy to help'.\n"
                "Your answer MUST be a single line containing ONLY the answer, with no extra text, no explanations, and no formatting. DO NOT REWORD THE QUESTION\n"
                "If the answer is not present in the context, say exactly: I don't know.\n"
                "\n"
                "Examples:\n"
                "Context: The capital of France is Paris.\n"
                "Question: What is the capital of France?\n"
                "Answer: Paris\n"
                "\n"
                "Context: The text does not mention the winner.\n"
                "Question: Who won the match?\n"
                "Answer: I don't know\n"
                "\n"
                f"Context: {question[2]}\n"
                f"Question: {question[0]}\n"
                "Answer:"
            )
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
    "llama2:70b"
]
for model in models:
    ensure_model_pulled(model)
    get_responses(model)
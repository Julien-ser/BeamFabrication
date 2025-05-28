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

def load_questions_from_jsonl(filepath):
    questions = []
    with open(filepath, "r") as infile:
        for line in infile:
            line = line.strip()
            obj = json.loads(line)
            questions.append([obj["question_text"], obj["question_id"]])

    return questions

def get_responses(model_name):
        # ...existing code for prompt...
    # Replace this with your actual list of questions
    questions = load_questions_from_jsonl("hottest.jsonl")
    output_path = f"{model_name.replace(':', '_')}_responses.jsonl"
    with open(output_path, "a") as outfile:
        for question in tqdm(questions, desc=f"Processing {model_name}"):
            '''prompt = f"""As an assistant, your task is to answer the question after <Question>. You should first think step by step about the question and give your thought and then answer the <Question>. Your answer should be after <Answer> in JSON format with key "thought" and "answer" and their values should be string.
            There are some examples for you to refer to:
            <Question>: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and born in December 5, 1932?
            <Answer>:
            ˋˋˋ json
            {{"thought":"Modern Record is a big R&B label with artists including Etta James, Joe Houston, Little Richard, Ike, Tina Turner and John Lee Hooker in the 1950s and 1960s. Little Richard is an American musician, singer actor and songwriter, born in December 5 1932. So the answer is Little Richard.","answer": "Little Richard"}}
            ˋˋˋ
            <Question>: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?
            <Answer>:
            ˋˋˋ json
            {{"thought":"Chinua Achebe was a Nigerian novelist, poet, professor, and critic. Rachel Carson was an American marine biologist, author, and conservationist. Chinua Achebe has 4 jobs while Rachel Carson has 3 jobs. So the answer is Chinua Achebe.","answer": "Chinua Achebe"}}
            ˋˋˋ
            <Question>: Remember Me Ballin’ is a CD single by Indo G that features an American rapper born in what year?
            <Answer>:
            ˋˋˋ json
            {{"thought":"Remember Me Ballin’ is the CD singer by Indo G that features Gangsta Boo, who is named Lola Mitchell, an American rapper born in 1979. So the answer is 1979.","answer": "1979"}}
            ˋˋˋ
            Now your Question is
            <Question>: {question[0]}
            <Answer>: """'''
            # ...existing code...
            prompt = f"""You are an expert assistant. For each question, you must first provide a factual, information-rich explanation as your "thought", and then give the answer. 
            Your "thought" must be a factual summary or synthesis of relevant information, not a plan, process, or reasoning steps. 
            Do not use phrases like "to determine", "we need to", "let's", or "I need to". 
            Your response must be a JSON object with two keys: "thought" (your factual explanation) and "answer" (the final answer). Both should be strings.
            Do not write anything except the JSON object. Here are some examples:

            <Question>: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and born in December 5, 1932?
            <Answer>:
            {{"thought":"Little Richard, born December 5, 1932, was an American musician, singer, actor, comedian, and songwriter who worked with Modern Records. He was a key figure in the development of rock and roll.","answer": "Little Richard"}}

            <Question>: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?
            <Answer>:
            {{"thought":"Chinua Achebe was a Nigerian novelist, poet, professor, and critic, while Rachel Carson was an American marine biologist, author, and conservationist. Achebe held four distinct professional roles, compared to Carson's three.","answer": "Chinua Achebe"}}

            <Question>: Remember Me Ballin’ is a CD single by Indo G that features an American rapper born in what year?
            <Answer>:
            {{"thought":"Gangsta Boo, featured on Indo G's 'Remember Me Ballin’', is an American rapper born in 1979.","answer": "1979"}}

            Now, answer the following question in the same way. Only output the JSON object.

            <Question>: {question[0]}
            <Answer>"""
            # ...existing code...
            #print(prompt)
            response: ChatResponse = chat(
                model=model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.7}
            )
            #print("RESPONSE=======================================\n" + response.message.content)
            match = re.search(r'\{.*?\}', response.message.content, re.DOTALL)
            if match:
                try:
                    response_data = json.loads(match.group())
                    response_data["question_id"] = question[1]
                    outfile.write(json.dumps(response_data) + "\n")
                except json.JSONDecodeError:
                    print("doesn't work")
                 #   response_data = {"thought": "", "answer": ""}
            #else:
             #   response_data = {"thought": "", "answer": ""}


models = [
    "llama2:7b"
]
for model in models:
    ensure_model_pulled(model)
    get_responses(model)
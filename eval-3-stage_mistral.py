from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from tqdm import tqdm

def load_questions_from_jsonl(filepath, is_check=False):
    questions = []
    ids = []
    with open("check.jsonl", "r") as infile:
        for line in infile:
            obj = json.loads(line)
            ids.append(obj["question_id"])
            if is_check:
                questions.append([obj["question_text"], obj["question_id"], obj["fabricated_candidate"]])

    if is_check:
        return questions

    with open(filepath, "r") as infile:
        for line in infile:
            obj = json.loads(line)
            contexts = obj.get("contexts", [])
            context = "\n".join([f"{c['title']}: {c['paragraph_text']}" for c in contexts])
            if obj["question_id"] in ids:
                questions.append([obj["question_text"], obj["question_id"], context])

    return questions

def get_responses(model_path, device="cuda"):
    questions = load_questions_from_jsonl("hottest_shuffled.jsonl", False)
    safe_model_name = model_path.replace(":", "_").replace("/", "_")
    output_path = f"{safe_model_name}_responses_shuffled.jsonl"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto")

    model.eval()

    results = []

    with open(output_path, "a") as outfile:
        for idx, question in enumerate(tqdm(questions, desc=f"Processing {model_path}", total=len(questions))):
            question_text, question_id, context = question

            prompt = f"""
You are a question answering assistant.

You MUST answer ONLY using the provided context and question below.
Do NOT ask the user for anything.
Do NOT add explanations, clarifications, or reword the question.
Your answer MUST be a single line, exactly the answer, with no extra text or formatting.

If the answer is not in the context, say exactly: I don't know.

Examples:
Context: The capital of France is Paris.
Question: What is the capital of France?
Answer: Paris

Context: The text does not mention the winner.
Question: Who won the match?
Answer: I don't know

Context: {context}
Question: {question_text}
Answer:
"""

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id,
                )
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract answer after "Answer:" (strip prompt)
            answer = answer.split("Answer:")[-1].strip().split("\n")[0]

            item = {"answer": answer, "question_id": question_id}
            outfile.write(json.dumps(item) + "\n")
            outfile.flush()  # Ensure immediate write
            results.append(item)
            tqdm.write(f"Processed {idx+1}/{len(questions)}: {question_id}")
            tqdm.write(f"Answer: {answer}")

    # Save backup at the end
    backup_path = f"{safe_model_name}_responses_shuffled.backup.jsonl"
    with open(backup_path, "w") as backupfile:
        for item in results:
            backupfile.write(json.dumps(item) + "\n")

# Example usage
model_path = "/home/Mahdiyar/Research/Julien/parametricattack/models/Mixtral-8x7B/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/41bd4c9e7e4fb318ca40e721131d4933966c2cc1"
get_responses(model_path)
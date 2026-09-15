"""Analyze one email with the phishing LoRA adapter.

Usage:
    python inference.py --size 3b --file my_email.txt
"""
import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "3b": ("unsloth/Qwen2.5-3B-Instruct-bnb-4bit", "3b-adapter"),
    "7b": ("unsloth/Qwen2.5-7B-Instruct-bnb-4bit", "7b-adapter"),
}

# Must be exactly the prompt used in training and evaluation
SYSTEM_PROMPT = (
    "You are an advanced AI security analyst specialized in email threat detection. "
    "Analyze the provided email and respond with a JSON object containing: "
    "is_phishing, confidence_score, threat_type, risk_level, indicators, "
    "mitigation_recommendations, analysis_summary."
)

CONTEXT_LENGTH = 2048
MAX_NEW_TOKENS = 500


def load_model(size):
    base_name, adapter_dir = MODELS[size]
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    model = AutoModelForCausalLM.from_pretrained(base_name, device_map="auto")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


def analyze(model, tokenizer, email_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": email_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = inputs.to(model.device)

    prompt_len = inputs["input_ids"].shape[-1]
    if prompt_len + MAX_NEW_TOKENS > CONTEXT_LENGTH:
        print(f"Warning: prompt is {prompt_len} tokens; the model was trained "
              f"with context {CONTEXT_LENGTH}. Consider shortening the email.")

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    raw = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    try:
        return json.loads(raw), raw
    except json.JSONDecodeError:
        return None, raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=["3b", "7b"], default="3b")
    parser.add_argument("--file", required=True, help="plain-text email file")
    args = parser.parse_args()

    with open(args.file, encoding="utf-8", errors="replace") as f:
        email_text = f.read()

    model, tokenizer = load_model(args.size)
    result, raw = analyze(model, tokenizer, email_text)

    if result is None:
        print("Model returned invalid JSON. Raw output:")
        print(raw)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

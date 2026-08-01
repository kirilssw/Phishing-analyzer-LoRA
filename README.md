---
base_model: unsloth/Qwen2.5-7B-Instruct-bnb-4bit
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
tags:
- lora
- qlora
- unsloth
- phishing-detection
- cybersecurity
- email-security
---

# Phishing Analyzer LoRA

QLoRA adapter for `Qwen2.5-7B-Instruct` that classifies emails as phishing or legitimate and returns a structured threat analysis (threat type, risk level, indicators, mitigation recommendations).

## Model Details

- **Base model:** `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` (4-bit quantized)
- **Adapter type:** QLoRA (PEFT `LORA`)
- **Rank (r):** 8
- **Alpha:** 16
- **Dropout:** 0
- **Target modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (attention + MLP)
- **Trained with:** [Unsloth Studio](https://unsloth.ai)
- **Training:** 1 epoch, context length 1024, learning rate 2e-4

## Training Data

- Base dataset: [CEAS-08](https://huggingface.co/datasets/nosadaniel/phishing-email-training-dataset) (~2,544 cleaned rows). Not redistributed in this repo — see the linked source for licensing.
- Custom augmentation (~210 examples): synthetic invoice fraud, executive/vendor impersonation, and hard-negative legitimate emails (e.g. legitimate bank-detail-change requests routed through proper verification channels), added specifically to correct a detection gap described below.

## Known Limitation (fixed)

An initial evaluation on a stratified 50-example test set found the model reliably detected `credential_harvesting` (6/6) and `urgency_pretext` attacks, but **completely failed on `invoice_fraud` (0/6) and `impersonation` (0/7)** — a result of the base CEAS-08 data skewing toward phishing with technical artifacts (links, attachments), so the model had learned to key on those markers rather than social-engineering patterns.

After retraining on the combined dataset (original + augmented BEC/impersonation examples), re-evaluation on the same 50-example stratified set gave:

| Category | Recall (raw) | Recall (after eval-harness fix*) |
|---|---|---|
| credential_harvesting | 6/6 | 6/6 |
| urgency_pretext | 3/5 | 5/5 |
| malicious_attachment | 3/6 | 6/6 |
| invoice_fraud | 5/6 | 6/6 |
| impersonation | 5/7 | 7/7 |
| **Overall accuracy** | **0.840 (42/50)** | **1.000 (50/50)** |
| legitimate (specificity) | 20/20 | 20/20 |

*8 of the raw "misses" were not classification errors: the model correctly identified `is_phishing: true` with the correct `threat_type` in every case, but the JSON output had a malformed field (e.g. a missing closing quote — `"severity": "22,` instead of `"severity": "2",`) that broke strict `json.loads()` parsing. This was fixed with a regex fallback in the evaluation harness, not by retraining. Raw model outputs were manually verified to confirm the underlying prediction was correct in all 8 cases.

The invoice_fraud/impersonation detection gap is resolved (0% → 100% recall). Zero false positives on legitimate emails in either run.

## How to Get Started

```python
from unsloth import FastLanguageModel
import torch, json

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length=1024,
    load_in_4bit=True,
)
model.load_adapter("./")
FastLanguageModel.for_inference(model)

messages = [
    {"role": "system", "content": "You are an advanced AI security analyst specialized in email threat detection. Analyze the provided email and respond with a JSON object containing: is_phishing, confidence_score, threat_type, risk_level, indicators, mitigation_recommendations, analysis_summary."},
    {"role": "user", "content": "<email text here>"},
]

inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

outputs = model.generate(inputs, max_new_tokens=500, do_sample=False)
response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
print(json.loads(response))
```

**Note:** the exact system prompt used during training/eval is not baked into the chat template (`chat_template.jinja` is the unmodified Qwen2.5 ChatML template) — it must be passed explicitly as shown above.

## Output Schema

```json
{
  "is_phishing": true,
  "confidence_score": 0.96,
  "threat_type": "business_email_compromise_invoice_fraud",
  "risk_level": "5",
  "indicators": [
    {"category": "...", "finding": "...", "severity": "4", "explanation": "..."}
  ],
  "mitigation_recommendations": {
    "immediate_actions": ["..."],
    "preventive_measures": ["..."],
    "reporting_guidance": "..."
  },
  "analysis_summary": "..."
}
```

## Limitations

- Trained on English-language emails only; untested on other languages.
- Evaluated on a 50-example stratified test set — small sample size, results should be treated as directional rather than a tight confidence interval.
- Occasional JSON-formatting glitches in greedy decoding (see above); production use should parse defensively (regex fallback) rather than assume strict JSON validity.
- Not evaluated against adversarial/adaptive phishing designed to evade this specific model.

## Framework Versions

- PEFT 0.19.1
- Trained via Unsloth Studio

## License

Adapter released under Apache 2.0 (inherits base model license terms — verify current Qwen2.5 license before commercial use).

---
base_model: unsloth/Qwen2.5-3B-Instruct-bnb-4bit
library_name: peft
pipeline_tag: text-generation
license: other
license_name: qwen-research
license_link: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE
tags:
- lora
- qlora
- unsloth
- phishing-detection
---

# Phishing Analyzer LoRA: 3B adapter

QLoRA adapter for `Qwen2.5-3B-Instruct` (r=8, alpha=16, context 2048, 2 epochs).

Subject to the **Qwen Research License Agreement** of the base model (non-commercial).

Training data, evaluation, results, limitations and usage: see the [main README](../README.md).
Not intended for use as a security control.

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
---

# Phishing Analyzer LoRA: 7B adapter

QLoRA adapter for `Qwen2.5-7B-Instruct` (r=8, alpha=16, context 2048, 2 epochs).

Apache 2.0.

Training data, evaluation, results, limitations and usage: see the [main README](../README.md).
Not intended for use as a security control.

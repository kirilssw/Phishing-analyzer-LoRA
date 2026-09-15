# Phishing Analyzer LoRA

QLoRA adapters for `Qwen2.5-Instruct` (3B and 7B) that read an email, classify it as phishing or legitimate, and return a structured JSON threat analysis: threat type, risk level, indicators and mitigation recommendations.

> Learning / portfolio project. Both adapters were evaluated on 75 external emails, and neither was tested against adversarial email. **Do not use this as a security control.** Read [Limitations](#limitations) before drawing conclusions from any number below.

## What this project covers

- QLoRA fine-tuning of 3B and 7B models for structured JSON output (Unsloth, PEFT).
- A failure mode and its fix: the first adapter caught link and attachment phishing but missed social-engineering attacks (BEC, invoice fraud, impersonation). Targeted synthetic data was added to close this gap.
- Evaluation on external corpora, with a paired McNemar comparison of four runs over the same items.
- Non-significant results reported alongside significant ones, with p-values and 95% CIs.

## Repository structure

```
.
├── 3b-adapter/          # adapter for Qwen2.5-3B-Instruct (see license note below)
├── 7b-adapter/          # adapter for Qwen2.5-7B-Instruct
├── assets/              # evaluation charts
├── inference.py         # analyze one email from a text file
├── requirements.txt
├── LICENSE
└── README.md
```

## Model details

| | 3B adapter | 7B adapter |
|---|---|---|
| Base model | `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` |
| Base model license | Qwen Research License (non-commercial) | Apache 2.0 |
| Adapter type | QLoRA (PEFT `LORA`) | QLoRA (PEFT `LORA`) |
| Rank / Alpha / Dropout | 8 / 16 / 0 | 8 / 16 / 0 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | same |
| Adapter file size | 57 MiB | 77 MiB |
| Trained with | Unsloth (QLoRA), Google Colab (T4) | Unsloth (QLoRA) |
| Epochs | 2 | 2 |
| Context length | 2048 | 2048 |
| Effective batch size | 8 | 8 |
| LR scheduler | cosine, `warmup_ratio=0.03` | cosine, `warmup_ratio=0.03` |
| Weight decay | 0.001 | 0.001 |
| Train/eval split | `train_test_split(test_size=0.1, seed=3407)` | same |
| PEFT version (from `adapter_config.json`) | 0.20.0 | 0.19.1 |

`chat_template.jinja` is the unmodified Qwen2.5 ChatML template. The system prompt from training and evaluation is not part of the template and has to be passed with every request (see [Quick start](#quick-start)).

## Training data

The base corpus is [`nosadaniel/phishing-email-training-dataset`](https://huggingface.co/datasets/nosadaniel/phishing-email-training-dataset) (MIT license per its dataset card), derived from CEAS-08. I cleaned it and combined it with synthetic BEC, invoice-fraud and impersonation examples into `datasetv3_2048.jsonl`, which has 2,259 rows. The file is not redistributed here.

Most label fields are machine-generated. Only the binary ham/phish label comes from CEAS-08. The other fields (`threat_type`, `risk_level`, `indicators`, `analysis_summary`) were written by an LLM when the dataset was built, so this adapter is a distillation of another model's phishing analyses, not a model trained on expert annotations. The threat-type taxonomy also comes from that dataset.

Other known issues:

- The dataset card mentions "200 balanced samples", while the file contains thousands of rows. I have not resolved this.
- CEAS-08 phishing is dominated by technical artifacts such as links and attachments. Without augmentation the model under-detects pure social engineering. The augmentation reduces this gap but does not remove it.
- Context length is 2048 because at 1024 tokens, 77–89% of the BEC and invoice-fraud examples were truncated. A shorter context would cut the very category the augmentation targets.

## Evaluation methodology

Four models (base 3B, 3B adapter, base 7B, 7B adapter) went through the same harness on two external test sets.

| Set | Sources | Size |
|---|---|---|
| Test B | Nazario Phishing Corpus, Nigerian Fraud corpus, SpamAssassin ham | 50 (25 legitimate, 25 phishing) |
| Test C | Phishing Pot honeypot corpus (rf-peixoto, CC BY-NC 4.0) | 25, phishing only |

Integrity checks:

- All four runs share the evaluation fingerprint `58b9c22ff94898bc68df95722874547d`, so they cover the same items. The paired McNemar test depends on this.
- The word "phishing" appeared in Phishing Pot `To:` headers. It was neutralized before inference to prevent label leakage.
- Raw generations were saved before parsing, so parser bugs could be fixed without re-running inference.
- With `FAILURE_IS_WRONG = True`, an unparseable output counts as an error. Baselines do not gain accuracy from excluded failures.

## Results

![Pooled recall, base vs trained](assets/pooled_recall_base_vs_trained.png)

| Model | Recall (Test B) | FPR (Test B) | Balanced acc. (Test B) | Recall (Test C) | Pooled accuracy (n=75) |
|---|---|---|---|---|---|
| base 3B | 0.960 | 0.240 | 0.860 | 0.920 | 0.880 |
| 3B adapter | 0.960 | 0.000 | 0.980 | 0.880 | 0.947 |
| base 7B | 0.920 | 0.000 | 0.960 | 0.560 | 0.827 |
| 7B adapter | 1.000 | 0.080 | 0.960 | 0.880 | 0.933 |

| 3B adapter | 7B adapter |
|---|---|
| ![Confusion matrix, 3B adapter](assets/confusion_matrix_3b.png) | ![Confusion matrix, 7B adapter](assets/confusion_matrix_7b.png) |

### Paired comparison (McNemar exact test, n=75)

| Comparison | Δ accuracy | 95% CI | p-value | Significant at 0.05? |
|---|---|---|---|---|
| base 3B → 3B adapter | +6.7 pp | [−0.1, +13.4] pp | 0.125 | No |
| base 7B → 7B adapter | +10.7 pp | [+1.9, +19.4] pp | 0.039 | Yes (marginal) |
| base 3B → base 7B | −5.3 pp | [−15.7, +5.1] pp | 0.455 | No |
| 3B adapter → 7B adapter | −1.3 pp | [−8.2, +5.6] pp | 1.000 | No |

### Interpretation

- Fine-tuning moves accuracy up for both sizes. The change is significant only for 7B, and p=0.039 at n=75 is weak evidence.
- The 3B improvement is not statistically significant.
- After fine-tuning, 3B and 7B are statistically indistinguishable. This data gives no reason to prefer the larger base.
- All confidence intervals are wide, so treat the numbers as directional.
- Post-hoc, on the 25 legitimate emails in Test B: base 3B flagged 6 as phishing and the 3B adapter flagged none (exact McNemar p≈0.031). This subgroup was examined after the overall 3B test came out non-significant, and the p-value is not corrected for that.

### Error analysis

From the table, the 3B adapter has 4 false negatives (1 in Test B, 3 in Test C) and the 7B adapter has 3 (all in Test C). Manual review of four false-negative cases found:

- 1 case of likely label noise in the source corpus.
- 2 near-duplicate templates that also appear in the training data, which points to a train/test deduplication gap.
- 1 genuine miss: a Ledger hardware-wallet CVE-impersonation email that neither adapter caught.

### Calibration

Some wrong verdicts came with a high `confidence_score`, and the scores are uncalibrated. Calibrate before using `confidence_score` as a threshold. The field is confidence in the verdict, not P(phishing); for ROC/AUC use `conf if is_phishing else 1 − conf`.

## Quick start

You need an NVIDIA GPU with CUDA, because the base models are 4-bit bitsandbytes checkpoints.

```bash
git clone https://github.com/kirilssw/Phishing-analyzer-LoRA.git
cd Phishing-analyzer-LoRA
pip install -r requirements.txt
python inference.py --size 3b --file my_email.txt
```

`my_email.txt` is a plain-text email with headers and body. The script prints the parsed JSON, or the raw output if the model returned invalid JSON.

System prompt used in training and evaluation:

```
You are an advanced AI security analyst specialized in email threat detection. Analyze the provided email and respond with a JSON object containing: is_phishing, confidence_score, threat_type, risk_level, indicators, mitigation_recommendations, analysis_summary.
```

## Output schema

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

Greedy decoding sometimes produces malformed JSON, such as a missing quote. Parse the output defensively.

## Limitations

- The external test sets hold 75 emails in total, which gives wide confidence intervals. These percentages will not transfer to real traffic.
- Test C contains no legitimate emails, so false-positive rate comes from the 25 legitimate emails in Test B alone. Pooled accuracy mixes a balanced set with a phishing-only one; the per-set metrics are more informative.
- Training labels beyond ham/phish are LLM-generated (see Training data).
- English only.
- No evaluation against adversarial or evasion-tuned phishing.
- Prompt injection is untested. The email body is untrusted input and may contain instructions aimed at the model.
- Confidence scores are uncalibrated.
- A train/test near-duplicate gap exists (see Error analysis).

## Project history

1. v1 (7B only): 1 epoch, context 1024. On a 50-example internal stratified set it detected link and attachment phishing and had 0% recall on invoice fraud and impersonation.
2. v1.1: I chose the fix and measured it on the same 50 examples that exposed the problem, so its 0% → 100% jump is not used as a result anywhere in this README.
3. v2 (current, 3B and 7B): context 2048, 2 epochs, augmented data, evaluated on external Test B and Test C as described above.

## Not included

- Training notebooks and the synthetic data generator
- Evaluation harness
- Test sets B and C, and `datasetv3_2048.jsonl`

## Roadmap

- Publish the evaluation harness (code only).
- Expand Test C to 150–200 emails and add legitimate emails to it.
- Deduplicate train and test at template level.
- Build a prompt-injection test set.
- Add a serving example (vLLM or llama.cpp with JSON-schema constrained decoding), published once the served model has been re-evaluated on Test B and Test C.

## License

- Code and the 7B adapter: Apache 2.0 (see `LICENSE`). The 7B base model [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) is Apache 2.0.
- The 3B adapter is a derivative of [`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), released under the Qwen Research License Agreement. Use of the 3B adapter is subject to that license, which does not permit commercial use without separate permission. Read the license text before using it.
- Evaluation corpora belong to their owners; Phishing Pot is CC BY-NC 4.0.

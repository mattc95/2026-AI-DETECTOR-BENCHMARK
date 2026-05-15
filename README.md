# 2026 AI Detector Benchmark

This repository contains a 2026 benchmark of four AI-text detection systems on a balanced English dataset of 1,000 texts. The benchmark is designed to evaluate not only overall accuracy, but also the human false positive rate: the rate at which real human writing is incorrectly flagged as AI-generated.

The tested detectors are:

- GPTHumanizer
- GPTZero
- ZeroGPT
- Sapling AI Detector

All detector runs in this repository were completed on May 14, 2026. The repository includes the benchmark input data, evaluation scripts, and aggregate metrics. Because the complete per-item detector outputs are large, they are hosted as public Google Drive artifacts and linked below for learning and research use.

## Related Links

- [GPTHumanizer Official Website](https://www.gpthumanizer.ai/)
- [Related benchmark blog article](https://www.gpthumanizer.ai/blog/2026-ai-detector-benchmark)

## Key Result

GPTZero achieved the highest overall accuracy in this run, while GPTHumanizer had the lowest human false positive risk.

| Detector | Total Items | Evaluable Items | Overall Accuracy | AI Detection Rate | Human False Positive Rate | AI Miss Rate | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPTHumanizer | 1,000 | 1,000 | 98.00% | 96.00% | 0.00% | 4.00% | 480 | 0 | 20 | 500 |
| GPTZero | 1,000 | 998 | 98.70% | 99.60% | 2.20% | 0.40% | 497 | 11 | 2 | 488 |
| ZeroGPT | 1,000 | 1,000 | 88.20% | 94.80% | 18.40% | 5.20% | 474 | 92 | 26 | 408 |
| Sapling | 1,000 | 1,000 | 88.60% | 96.60% | 19.40% | 3.40% | 483 | 97 | 17 | 403 |

The confusion matrix treats AI as the positive class:

- TP: AI text correctly classified as AI
- FP: human text incorrectly classified as AI
- FN: AI text incorrectly classified as human
- TN: human text correctly classified as human

GPTZero returned two API errors during the completed run, so its accuracy and rates are calculated on 998 evaluable records. Those two non-ok records are still preserved in the output file.

## Dataset

The benchmark uses 1,000 English texts:

| Split | File | Count | Label | Sampling Notes |
|---|---|---:|---|---|
| Human | `data/human_detection_test.json` | 500 | `human` | Randomly sampled from Pile-small across different source domains. |
| AI | `data/ai_detection_test.json` | 500 | `ai` | Random sample of 500 texts from a 2,600-item AI generation pool created from prompts on February 5, 2026. |

The word-count distribution is balanced across the human and AI splits:

| Word Count Bucket | Human Items | AI Items | Total Items |
|---|---:|---:|---:|
| 50-200 words | 150 | 150 | 300 |
| 200-500 words | 200 | 200 | 400 |
| 500-1000 words | 150 | 150 | 300 |

### Human Data Credibility

The human benchmark set contains 500 human-written texts sampled from Pile-small. The final test file stores each sample with its text, label, source, style field, and perplexity score.

Human source coverage in the final benchmark set:

| Source | Count |
|---|---:|
| Wikipedia (en) | 73 |
| OpenWebText2 | 61 |
| Pile-CC | 59 |
| USPTO Backgrounds | 55 |
| StackExchange | 54 |
| NIH ExPorter | 40 |
| HackerNews | 39 |
| FreeLaw | 36 |
| PubMed Abstracts | 36 |
| Enron Emails | 26 |
| PubMed Central | 11 |
| ArXiv | 7 |
| YoutubeSubtitles | 3 |

Why this matters:

- The human data is not drawn from a single writing style or domain.
- The sources include encyclopedic, web, legal, technical, biomedical, email, forum, patent, academic, and subtitle-style text.
- Every final human benchmark record has a `human` label, non-empty text, a source field, and a retained per-item record in the detector outputs.

### AI Data Credibility

The AI benchmark set contains 500 AI-generated texts randomly sampled from a larger pool of 2,600 prompted generations. The AI generation pool was produced on February 5, 2026 by prompting large language models directly. The final benchmark file stores each sampled item with its text, label, prompt, source model, and theme.

AI source model coverage in the final benchmark set:

| Source Model | Count |
|---|---:|
| claude-sonnet-4-20250514 | 46 |
| gpt-3.5-turbo-0613 | 46 |
| gpt-4.1 | 42 |
| claude-3-7-sonnet-20250219 | 42 |
| o3 | 42 |
| deepseek-chat | 42 |
| kimi-k2-0905-preview | 40 |
| gpt-4o | 36 |
| grok-4 | 36 |
| claude-sonnet-4-5-20250929 | 36 |
| gpt-5-chat-latest | 34 |
| gpt-5-mini | 31 |
| claude-3-5-sonnet-20241022 | 27 |

Why this matters:

- The AI split is not produced by a single model family.
- The model source is retained for every final benchmark item.
- The prompts are preserved in `data/ai_detection_test.json`, making the sampled AI data auditable at the item level.

## Detector Output Files

All model test results have complete classification data. Because these output JSON files are large, the full artifacts are stored on Google Drive instead of being committed directly to the GitHub repository. They are publicly shared and may be used for learning or research purposes.

Each output JSON contains aggregate metrics and an `items` object with detailed per-item results, including text, source metadata, true label, predicted label, correctness, word count, word bucket, request status, HTTP status where available, and the detector response fields.

| Detector | Full Classification Data | Original Output File | Items | Status Summary | Notes |
|---|---|---|---:|---|---|
| GPTHumanizer | [Google Drive](https://drive.google.com/file/d/1iOvYJBHBS-8ulAN7DxQ_XUu69dmyfokC/view?usp=sharing) | `output/detection_eval_output.json` | 1,000 | 1,000 ok | Full per-item classifier output. |
| GPTZero | [Google Drive](https://drive.google.com/file/d/1j8xFQ3Rgrg2Xl7W6NpdoMSjMXCiYY9jp/view?usp=sharing) | `output/gptzero_detection_eval_output.json` | 1,000 | 998 ok, 2 error | Two API 403 errors are preserved and excluded from evaluable metrics. |
| ZeroGPT | [Google Drive](https://drive.google.com/file/d/1Qyyc39NX2CprZAXXkviLkwmkdZySC0yQ/view?usp=sharing) | `output/zerogpt_detection_eval_output.json` | 1,000 | 1,000 ok | Full per-item feedback and response data. |
| Sapling | [Google Drive](https://drive.google.com/file/d/1oy9nFC5gLsT81erGc2BF_PLSFqC-BUZz/view?usp=sharing) | `output/sapling_detection_eval_output.json` | 1,000 | 1,000 ok | Full per-item Sapling response data. |

GPTZero non-ok records from the completed run:

| Item Key | Status | Error |
|---|---|---|
| `human_detection_test:33` | error | 403 Client Error: Forbidden |
| `ai_detection_test:350` | error | 403 Client Error: Forbidden |

## Prediction Rules

Different detectors return different output formats. The benchmark normalizes each detector into a binary `human` or `ai` prediction using explicit rules stored in the output metadata.

| Detector | Normalization Rule |
|---|---|
| GPTHumanizer | `human`, `light_edited`, and `lightly_edited` classes are treated as human; all other classes are treated as AI. |
| GPTZero | `predicted_class = human` is treated as human; `ai` or `mixed` is treated as AI. |
| ZeroGPT | Feedback containing `Human written` is treated as human; all other feedback is treated as AI. |
| Sapling | AI score greater than 50% is treated as AI; score less than or equal to 50% is treated as human. |

These rules are implemented in the evaluation scripts and recorded in each output file's `meta.prediction_rule` field.

## Metrics

The benchmark reports the following metrics:

- Overall accuracy: `(TP + TN) / evaluable_items`
- AI detection rate: `TP / (TP + FN)`
- Human false positive rate: `FP / (FP + TN)`
- AI miss rate: `FN / (TP + FN)`
- Evaluable items: records with a valid normalized `human` or `ai` prediction

The human false positive rate is a central metric because false accusations against human-written text can create serious academic, professional, or institutional risk.

## Performance by Text Length

Shorter text is harder to classify because detectors have less linguistic evidence. The benchmark therefore reports results by word-count bucket.

### 50-200 Words

| Detector | Accuracy | Human False Positive Rate | AI Miss Rate | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPTHumanizer | 95.33% | 0.00% | 9.33% | 136 | 0 | 14 | 150 |
| GPTZero | 96.67% | 6.00% | 0.67% | 149 | 9 | 1 | 141 |
| ZeroGPT | 85.33% | 20.67% | 8.67% | 137 | 31 | 13 | 119 |
| Sapling | 85.33% | 24.67% | 4.67% | 143 | 37 | 7 | 113 |

### 200-500 Words

| Detector | Accuracy | Human False Positive Rate | AI Miss Rate | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPTHumanizer | 99.00% | 0.00% | 2.00% | 196 | 0 | 4 | 200 |
| GPTZero | 99.25% | 1.00% | 0.50% | 199 | 2 | 1 | 198 |
| ZeroGPT | 91.00% | 17.50% | 0.50% | 199 | 35 | 1 | 165 |
| Sapling | 89.25% | 18.50% | 3.00% | 194 | 37 | 6 | 163 |

### 500-1000 Words

| Detector | Accuracy | Human False Positive Rate | AI Miss Rate | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPTHumanizer | 99.33% | 0.00% | 1.33% | 148 | 0 | 2 | 150 |
| GPTZero | 100.00% | 0.00% | 0.00% | 149 | 0 | 0 | 149 |
| ZeroGPT | 87.33% | 17.33% | 8.00% | 138 | 26 | 12 | 124 |
| Sapling | 91.00% | 15.33% | 2.67% | 146 | 23 | 4 | 127 |

GPTZero has 298 evaluable records in the 500-1000 word bucket because two records in that bucket returned API errors.

## Main Findings

1. GPTHumanizer produced zero human false positives in this benchmark.

   It classified all 500 evaluable human-written texts as human. This does not prove a universal 0% false positive rate, but it is the strongest human-safety result observed in this dataset.

2. GPTZero produced the strongest raw accuracy and AI recall.

   GPTZero reached 98.70% accuracy on 998 evaluable records and detected 497 of 499 evaluable AI texts. It also incorrectly flagged 11 of 499 evaluable human texts as AI.

3. ZeroGPT and Sapling showed high human false positive risk.

   ZeroGPT flagged 92 of 500 human texts as AI. Sapling flagged 97 of 500 human texts as AI under the benchmark threshold of score > 50%.

4. Short text is the most difficult bucket.

   All detectors showed weaker behavior on 50-200 word samples. GPTHumanizer missed more short AI texts, while GPTZero, ZeroGPT, and Sapling showed higher human false positive rates in that range.

## Repository Structure

```text
.
|-- data/
|   |-- human_detection_test.json
|   |-- ai_detection_test.json
|-- evaluate_detection_datasets.py
|-- evaluate_gptzero_datasets.py
|-- evaluate_zerogpt_datasets.py
|-- evaluate_sapling_datasets.py
|-- requirements.txt
`-- README.md
```

## Reproducing or Extending the Benchmark

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the GPTHumanizer evaluation:

```bash
python evaluate_detection_datasets.py --restart
```

Run GPTZero:

```bash
export GPTZERO_API_KEY="your_key_here"
python evaluate_gptzero_datasets.py --restart
```

Run ZeroGPT:

```bash
export ZEROGPT_API_KEY="your_key_here"
python evaluate_zerogpt_datasets.py --restart
```

Run Sapling:

```bash
export SAPLING_API_KEY="your_key_here"
python evaluate_sapling_datasets.py --restart --drop-token-fields
```

On Windows PowerShell, set environment variables with `$env:NAME = "value"` before running the scripts.

The scripts also support explicit headers or API keys through command-line options. Do not commit private API keys to the repository.

Optional Sapling threshold analysis:

```bash
python analyze_sapling_thresholds.py --threshold 80 --threshold 90 -o output/sapling_threshold_analysis_80_90.json
```

## Auditability

This repository is intended to make the benchmark auditable rather than aggregate-only.

Evidence available in this repository and the linked public artifacts:

- The final 500 human test records.
- The final 500 AI test records.
- Source metadata for human samples.
- Prompt, theme, and model-source metadata for AI samples.
- Evaluation scripts for all four detectors.
- Public Google Drive links to the complete per-item classifier outputs for all four detectors.
- Aggregate metrics embedded in each linked output file.
- Sapling threshold analysis files for alternative score thresholds.

This means the headline results can be checked against the item-level classifications by downloading the linked output artifacts instead of relying only on summary tables.

## Limitations

- The benchmark is English-only.
- The sample size is 1,000 texts, balanced as 500 human and 500 AI.
- Detector APIs can change over time. These results describe the detector behavior observed in the completed May 14, 2026 runs.
- GPTZero had two API errors in the completed run. They are preserved in the output and excluded from evaluable metrics.
- The benchmark uses explicit binary normalization rules, but some detector outputs are more nuanced than binary human/AI labels.
- A measured 0% false positive rate on 500 human samples should be interpreted as an observed benchmark result, not as proof that future false positives cannot occur.
- Exact regeneration of the original random samples may require the original upstream Pile-small snapshot, AI generation pool, sampling code, and random seeds if those are not otherwise archived.

## Responsible Use

AI detectors should support review, not replace human judgment. A detector result should not be used as the only evidence in high-stakes academic, employment, publishing, or compliance decisions.

This benchmark suggests that detector trust should be evaluated with special attention to human false positives. Catching AI-generated text is useful, but falsely accusing human writers is often the more serious risk.
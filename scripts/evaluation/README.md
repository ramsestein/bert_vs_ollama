# Performance Evaluation Scripts

Scripts for evaluating the quality of final NER system output after running `ner_app`.

## Purpose

These scripts help you **measure and analyze** the performance of your NER system's predictions, including:
- Calculating precision, recall, and F1-score
- Identifying false positives and false negatives
- Understanding error patterns

Use these scripts **after** running your production NER system to assess output quality.

---

## ICD10 Support

All evaluation scripts (`evaluate_ner_performance`, `analyze_false_negatives`, `analyze_false_positives`, and `corrected_metrics_analysis`) have ICD10-aware variants implemented for the Hospital Clínic dataset, mapping entity text to ICD10 codes for code-level evaluation.

---

## Scripts

### 1. `evaluate_ner_performance.py`
**Core evaluation script - calculates P/R/F1 metrics**

Compares predictions against ground truth to calculate standard NER metrics.

**Usage**:
```bash
# Evaluate a single result file
python scripts/evaluation/evaluate_ner_performance.py \
  --predictions results_qwen_n2c2_chunk60_ov20.jsonl \
  --reference datasets/n2c2_test.jsonl

# Evaluate without reference (self-evaluation mode)
python scripts/evaluation/evaluate_ner_performance.py \
  --predictions results_qwen_n2c2_chunk60_ov20.jsonl
```

**Output**:
```
Precisión: 0.923 (TP: 450 / (TP + FP): 487)
Recall: 0.879 (TP: 450 / (TP + FN): 512)
F1-Score: 0.900
```

**Features**:
- Fuzzy matching for entity comparison
- Document-level statistics
- Detailed TP/FP/FN counts

---

### 2. `analyze_false_negatives_clean.py`
**In-depth analysis of missed entities (False Negatives)**

Identifies which entities from ground truth were not detected by the system.

**Usage**:
```bash
python scripts/evaluation/analyze_false_negatives_clean.py \
  --predictions results_qwen_n2c2_chunk60_ov20.jsonl \
  --benchmark datasets/n2c2_test.jsonl \
  --output fn_analysis.json
```

**Output**:
- List of all missed entities with context
- Frequency analysis of missed entities
- Document-level FN statistics
- Patterns in missed detections

**Use Cases**:
- Understanding recall limitations
- Identifying systematic gaps in entity coverage
- Finding entities that need better prompts or strategies

---

### 3. `analyze_false_positives_clean.py`
**In-depth analysis of incorrectly detected entities (False Positives)**

Identifies which entities were detected but don't exist in ground truth.

**Usage**:
```bash
python scripts/evaluation/analyze_false_positives_clean.py \
  --predictions results_qwen_n2c2_chunk60_ov20.jsonl \
  --benchmark datasets/n2c2_test.jsonl \
  --output fp_analysis.json
```

**Output**:
- List of all false positive entities with context
- Confidence scores of incorrect detections
- Document-level FP statistics
- Patterns in false detections

**Use Cases**:
- Understanding precision limitations
- Identifying over-detection patterns
- Tuning confidence thresholds to reduce FPs

---

## Workflow

1. **Run your NER system** on test data:
   ```bash
   python -m ner_app.main \
     --input datasets/n2c2_test_input.jsonl \
     --output results_final.jsonl \
     --strategy-config config_optimal.json
   ```

2. **Evaluate overall performance**:
   ```bash
   python scripts/evaluation/evaluate_ner_performance.py \
     --predictions results_final.jsonl \
     --reference datasets/n2c2_test.jsonl
   ```

3. **Analyze errors** if metrics need improvement:
   ```bash
   # If recall is low, analyze false negatives
   python scripts/evaluation/analyze_false_negatives_clean.py \
     --predictions results_final.jsonl \
     --benchmark datasets/n2c2_test.jsonl
   
   # If precision is low, analyze false positives
   python scripts/evaluation/analyze_false_positives_clean.py \
     --predictions results_final.jsonl \
     --benchmark datasets/n2c2_test.jsonl
   ```

4. **Iterate** on your system based on error analysis findings

---

## Key Differences from Optimization Scripts

| Evaluation Scripts | Optimization Scripts |
|-------------------|---------------------|
| Measure final performance | Find best parameters |
| Single configuration | Multiple configurations |
| Detailed error analysis | Comparative analysis |
| After production | Before production |
| Outputs: P/R/F1, FP/FN details | Outputs: parameter rankings |

---

## Metrics Explained

- **Precision**: Of all entities detected, how many were correct?
  - `TP / (TP + FP)`
  - High precision = few false alarms

- **Recall**: Of all entities in ground truth, how many did we find?
  - `TP / (TP + FN)`
  - High recall = few missed entities

- **F1-Score**: Harmonic mean of precision and recall
  - `2 * (Precision * Recall) / (Precision + Recall)`
  - Balanced metric for overall performance

---

## Dependencies

These scripts use the shared `ner_analysis` module (through the evaluator wrapper) for consistent metric calculation.

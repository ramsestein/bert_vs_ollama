# Scripts Directory

Organized collection of scripts for NER system parameter optimization and performance evaluation.

---

## 📁 Directory Structure

```
scripts/
├── optimization/              # Parameter tuning (BEFORE production)
│   ├── run_grid_search.py            # Grid search for chunks/overlap/temp
│   ├── run_temperature_grid.py       # Temperature-focused grid search
│   ├── analyze_grid_results.py       # Universal grid search analyzer
│   ├── analyze_temperature_grid.py   # Temperature-focused analyzer
│   └── README.md                     # Detailed optimization docs
│
├── evaluation/                # Performance assessment (AFTER production)
│   ├── evaluate_ner_performance.py   # Calculate P/R/F1 metrics
│   ├── analyze_false_negatives_clean.py  # FN error analysis
│   ├── analyze_false_positives_clean.py  # FP error analysis
│   └── README.md                     # Detailed evaluation docs
│
└── ner_analysis/              # Shared utilities module
    ├── file_parser.py                # Parse result filenames and files
    ├── evaluator.py                  # Wrapper for metric calculation
    ├── aggregator.py                 # Aggregation functions
    └── README.md                     # Module documentation
```

---

## 🎯 Quick Guide: Which Scripts to Use?

### Scenario 1: Finding Optimal Parameters
**Goal**: Determine best chunk_target, overlap, temperature for your model

1. Run grid search:
   ```bash
   python scripts/optimization/run_grid_search.py --model qwen --dataset n2c2 --limit 50
   ```

2. Analyze results:
   ```bash
   python scripts/optimization/analyze_grid_results.py \
     qwen_grid_n2c2_YYYYMMDD_HHMMSS \
     --reference datasets/n2c2_test.jsonl \
     --top 10
   ```

3. Pick the best configuration from the top results

**Scripts to use**: `optimization/`

---

### Scenario 2: Evaluating Final System
**Goal**: Measure how well your NER system performs with optimized parameters

1. Run NER system:
   ```bash
   python -m ner_app.main \
     --input datasets/n2c2_test_input.jsonl \
     --output results_final.jsonl \
     --chunk-target 60 \
     --overlap 20
   ```

2. Calculate metrics:
   ```bash
   python scripts/evaluation/evaluate_ner_performance.py \
     --predictions results_final.jsonl \
     --reference datasets/n2c2_test.jsonl
   ```

3. If metrics need improvement, analyze errors:
   ```bash
   # Low recall? Check what you're missing
   python scripts/evaluation/analyze_false_negatives_clean.py \
     --predictions results_final.jsonl \
     --benchmark datasets/n2c2_test.jsonl
   
   # Low precision? Check false alarms
   python scripts/evaluation/analyze_false_positives_clean.py \
     --predictions results_final.jsonl \
     --benchmark datasets/n2c2_test.jsonl
   ```

**Scripts to use**: `evaluation/`

---

## 🔄 Complete Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    OPTIMIZATION PHASE                        │
│  (Finding best parameters)                                   │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │  1. Run Grid Search              │
        │     scripts/optimization/        │
        │     run_grid_search.py           │
        └──────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │  2. Analyze Grid Results         │
        │     scripts/optimization/        │
        │     analyze_grid_results.py      │
        └──────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │  3. Select Best Parameters       │
        │     (highest F1-score)           │
        └──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    PRODUCTION PHASE                          │
│  (Running final system)                                      │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │  4. Run NER System               │
        │     ner_app/main.py              │
        │     with optimal params          │
        └──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    EVALUATION PHASE                          │
│  (Measuring performance)                                     │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │  5. Calculate Metrics            │
        │     scripts/evaluation/          │
        │     evaluate_ner_performance.py  │
        └──────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │  6. Analyze Errors (if needed)   │
        │     scripts/evaluation/          │
        │     analyze_false_*.py           │
        └──────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │  7. Iterate or Deploy            │
        └──────────────────────────────────┘
```

---

## 📊 Key Distinctions

### Optimization Scripts (`optimization/`)
- **Purpose**: Find best hyperparameters
- **When**: Before running production system
- **Input**: Multiple parameter combinations
- **Output**: Parameter rankings, comparative statistics
- **Focus**: Which settings work best?

### Evaluation Scripts (`evaluation/`)
- **Purpose**: Measure system quality
- **When**: After running production system
- **Input**: Single result file with predictions
- **Output**: P/R/F1 metrics, error analysis
- **Focus**: How good are the predictions?

---

## 🛠️ Shared Utilities

### `ner_analysis/` Module
Common functionality used by both optimization and evaluation scripts:

- **file_parser.py**: Parse result filenames and JSONL files
- **evaluator.py**: Calculate P/R/F1 metrics (wrapper for evaluate_ner_performance.py)
- **aggregator.py**: Aggregate results by parameters

Both optimization and evaluation scripts import from this module for consistency.

---

## 📖 Detailed Documentation

- **Optimization scripts**: See `scripts/optimization/README.md`
- **Evaluation scripts**: See `scripts/evaluation/README.md`
- **ner_analysis module**: See `scripts/ner_analysis/README.md`

---

## 🔍 Examples

### Example 1: Complete Parameter Optimization
```bash
# Run grid search on 50 documents
python scripts/optimization/run_grid_search.py \
  --model qwen \
  --dataset n2c2 \
  --limit 50 \
  --custom-chunks 40,60,80 \
  --custom-overlaps 15,25,35

# Analyze results
python scripts/optimization/analyze_grid_results.py \
  qwen_grid_n2c2_20251203_143022 \
  --reference datasets/n2c2_test.jsonl \
  --top 5

# Output shows: chunk=60, overlap=25, temp=0.5 has best F1
```

### Example 2: Production Evaluation
```bash
# Run system with optimal parameters
python -m ner_app.main \
  --input datasets/n2c2_test_input.jsonl \
  --output results_production.jsonl \
  --chunk-target 60 \
  --overlap 25 \
  --temperature 0.5

# Evaluate
python scripts/evaluation/evaluate_ner_performance.py \
  --predictions results_production.jsonl \
  --reference datasets/n2c2_test.jsonl

# Result: Precision: 0.923, Recall: 0.879, F1: 0.900

# Analyze why recall is not higher
python scripts/evaluation/analyze_false_negatives_clean.py \
  --predictions results_production.jsonl \
  --benchmark datasets/n2c2_test.jsonl \
  --output fn_report.json
```

---

## 📝 Notes

- All scripts support `--help` for detailed usage information
- Grid search can take significant time; use `--limit` to test on subset first
- Evaluation scripts require ground truth files with entity annotations
- Always keep optimization and evaluation results organized in separate directories
- See `README_OLD.md` for previous documentation format

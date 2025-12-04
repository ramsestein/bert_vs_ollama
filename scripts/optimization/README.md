# Parameter Optimization Scripts

Scripts for finding optimal strategy parameters through grid search and systematic analysis.

## Purpose

These scripts help you **tune hyperparameters** for the NER system strategies, including:
- Chunk target sizes
- Overlap amounts  
- LLM temperatures
- Confidence thresholds

Use these scripts **before** running your production NER system to find the best parameter combinations.

---

## Scripts

### 1. `run_grid_search.py`
**Grid search for chunk size, overlap, and temperature parameters**

Systematically tests combinations of parameters to find optimal settings.

**Usage**:
```bash
# Basic grid search
python scripts/optimization/run_grid_search.py --model qwen --dataset n2c2 --limit 50

# Custom parameters
python scripts/optimization/run_grid_search.py \
  --model qwen \
  --dataset n2c2 \
  --limit 100 \
  --custom-chunks 40,60,80 \
  --custom-overlaps 10,20,30 \
  --custom-temps 0.0,0.3,0.5,0.7
```

**Output**: Creates a timestamped directory with results for each parameter combination.

---

### 2. `run_temperature_grid.py`
**Focused grid search for temperature and confidence thresholds**

Fine-tunes LLM temperature and confidence threshold parameters.

**Usage**:
```bash
python scripts/optimization/run_temperature_grid.py \
  --model qwen \
  --dataset n2c2 \
  --limit 50
```

---

### 3. `analyze_grid_results.py`
**Universal analyzer for grid search results**

Analyzes any grid search results directory and identifies best parameter combinations.

**Usage**:
```bash
# Analyze results with reference evaluation
python scripts/optimization/analyze_grid_results.py \
  qwen_grid_n2c2_20251203_100000 \
  --reference datasets/n2c2_test.jsonl \
  --top 10

# Analyze without evaluation (just entity counts and confidence)
python scripts/optimization/analyze_grid_results.py \
  qwen_grid_n2c2_20251203_100000
```

**Output**:
- Aggregated statistics by parameter
- Top N configurations ranked by F1-score
- Parameter impact analysis

---

### 4. `analyze_temperature_grid.py`
**Specialized analyzer for temperature grid results**

Similar to `analyze_grid_results.py` but with temperature-focused reporting.

**Usage**:
```bash
python scripts/optimization/analyze_temperature_grid.py \
  temperature_grid_20251203_100000 \
  --reference datasets/n2c2_test.jsonl
```

---

## Workflow

1. **Run grid search** to test parameter combinations:
   ```bash
   python scripts/optimization/run_grid_search.py --model qwen --dataset n2c2 --limit 50
   ```

2. **Analyze results** to find best parameters:
   ```bash
   python scripts/optimization/analyze_grid_results.py \
     qwen_grid_n2c2_YYYYMMDD_HHMMSS \
     --reference datasets/n2c2_test.jsonl \
     --top 10
   ```

3. **Update your configuration** with the optimal parameters found

4. **Run production system** with optimized settings using `ner_app`

---

## Key Differences from Evaluation Scripts

| Optimization Scripts | Evaluation Scripts |
|---------------------|-------------------|
| Find best parameters | Measure final performance |
| Multiple configurations | Single configuration |
| Comparative analysis | Detailed error analysis |
| Before production | After production |
| Outputs: parameter rankings | Outputs: P/R/F1, FP/FN details |

---

## Dependencies

These scripts use the shared `ner_analysis` module for common functionality like file parsing and metric calculation.

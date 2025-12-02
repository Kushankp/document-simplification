# Reproducibility Study: Document-Level Simplification Evaluation

This repository reproduces the core results from:

**Maddela, Mounica & Alva-Manchego, Fernando (2025).**  
*“Adapting Sentence-Level Metrics for Document-Level Simplification Evaluation.”* NAACL 2025.

The paper introduces a **many-to-many sentence alignment** method that adapts sentence-level metrics for **document-level text simplification evaluation**, producing:

- **Agg-SARI**  
- **Agg-BERTScore**

We reproduce these results on the **DWIKI (Wikipedia) simplification dataset**.

---

## Project Structure

```
eval_datasets/            # Processed datasets (DWIKI, Cochrane)
metrics/                  # SARI, BERTScore, DSARI, AggMetricGraph
evaluate_pearson.py       # Sentence-level evaluation (DWIKI)
evaluate_kendall.py       # Optional evaluation (Cochrane)
run_agg_wrapper.py        # Computes Agg-SARI and Agg-BERTScore
paper_results.py          # Prints correlation table
requirements.txt          # Dependencies
```

> Note: `reader_scripts/` is not required — datasets are already included.

---

## Setup Instructions

### 1. Create and activate environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download BERT / RoBERTa model
Place it under:

```
BERT_wiki/
  config.json
  tokenizer.json OR vocab file
  pytorch_model.bin
```

This model is used to compute similarity edges in the alignment graph.

---

## How to Run

### **A) DWIKI — Sentence-Level Baseline**
```bash
python3 evaluate_pearson.py \
  --dataset eval_datasets/dwiki_final.jsonl \
  --output output.jsonl \
  --bert BERT_wiki/
```

### **B) DWIKI — Document-Level Aggregation (Proposed Method)**
```bash
PYTHONWARNINGS="ignore" python3 run_agg_wrapper.py \
  --input eval_datasets/dwiki_final.jsonl \
  --output agg_output_wiki.jsonl \
  --bert BERT_wiki/
```

### **C) Print Final Correlation Results**
```bash
python3 paper_results.py
```

### **Optional (Cochrane — Kendall Correlation)**
```bash
python3 evaluate_kendall.py \
  --dataset eval_datasets/cochrance_lj.jsonl \
  --output output_cochrane.jsonl \
  --bert BERT_wiki/
```

---

## Reproduced Results (DWIKI)

Both the original paper and our reproduction show the same overall trend:

- **Agg-SARI** aligns best with *simplicity* and *grammar*  
- **Agg-BERTScore** aligns best with *meaning preservation*

| Dimension         | Agg-SARI (Paper) | Agg-SARI (Ours) | Agg-BERTScore (Paper) | Agg-BERTScore (Ours) |
|-------------------|------------------|------------------|-------------------------|------------------------|
| simplicity-word   | ~0.30            | 0.26             | ~0.10                   | 0.05                   |
| simplicity-sent   | ~0.29            | 0.26             | ~0.12                   | 0.06                   |
| meaning           | ~0.08            | 0.07             | ~0.38                   | 0.42                   |
| grammar           | ~0.33            | 0.34             | ~0.25                   | 0.23                   |

### Interpretation
Although minor numeric differences occur due to different environments or model versions,  
the **relative ranking pattern matches the original paper**, confirming successful reproducibility.

---

## Citation

```
@InProceedings{NAACL-2025-Maddela,
  author = {Maddela, Mounica and Alva-Manchego, Fernando},
  title = {Adapting Sentence-Level Metrics for Document-Level Simplification},
  booktitle = {NAACL},
  year = {2025}
}
```

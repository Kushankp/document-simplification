Reproducibility Study: Document-Level Simplification Evaluation

This project reproduces the core results from the paper:

Maddela, Mounica & Alva-Manchego, Fernando (2025).
"Adapting Sentence-Level Metrics for Document-Level Simplification Evaluation." NAACL.

The paper proposes a many-to-many sentence alignment method that adapts traditional
sentence-level metrics (SARI, BERTScore) for document-level text simplification.
These aggregated metrics are referred to as:
- Agg-SARI
- Agg-BERTScore

Our goal was to reproduce these results on the DWIKI (Wikipedia) simplification dataset.

------------------------------------------------------------
Project Structure
------------------------------------------------------------
eval_datasets/            Processed datasets (DWIKI, Cochrane, etc.)
metrics/                  SARI, BERTScore, DSARI, AggMetricGraph
evaluate_pearson.py       Sentence-level evaluation (DWIKI)
evaluate_kendall.py       Optional evaluation (Cochrane)
run_agg_wrapper.py        Computes Agg-SARI and Agg-BERTScore
paper_results.py          Prints correlation table
requirements.txt          Package list

Note: reader_scripts/ is not required because datasets are already provided.

------------------------------------------------------------
Setup Instructions
------------------------------------------------------------
1) Create and activate environment:
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2) Download a BERT / RoBERTa checkpoint and place inside:
   BERT_wiki/
     config.json
     tokenizer.json or vocab file
     pytorch_model.bin

This model is used to compute similarity edges in the alignment graph.

------------------------------------------------------------
How to Run
------------------------------------------------------------

A) DWIKI (sentence-level baseline):
   python3 evaluate_pearson.py --dataset eval_datasets/dwiki_final.jsonl --output output.jsonl --bert BERT_wiki/

B) DWIKI (document-level aggregation: proposed method):
   PYTHONWARNINGS="ignore" python3 run_agg_wrapper.py --input eval_datasets/dwiki_final.jsonl --output agg_output_wiki.jsonl --bert BERT_wiki

C) Print final correlation results:
   python3 paper_results.py

Optional (Cochrane - Kendall):
   python3 evaluate_kendall.py --dataset eval_datasets/cochrance_lj.jsonl --output output_cochrane.jsonl --bert BERT_wiki/

------------------------------------------------------------
Reproduced Results (DWIKI) – Authors vs Ours
------------------------------------------------------------
The original paper and our reproduction show the same performance trend:
- Agg-SARI aligns best with simplicity and grammar.
- Agg-BERTScore aligns best with meaning preservation.

Dimension           | Agg-SARI (Paper) | Agg-SARI (Ours) | Agg-BERTScore (Paper) | Agg-BERTScore (Ours)
---------------------------------------------------------------------------------------------------------
simplicity-word     |      ~0.30       |      0.26       |        ~0.10          |        0.05
simplicity-sent     |      ~0.29       |      0.26       |        ~0.12          |        0.06
meaning             |      ~0.08       |      0.07       |        ~0.38          |        0.42
grammar             |      ~0.33       |      0.34       |        ~0.25          |        0.23

Interpretation:
Our reproduced results match the *relative ranking pattern* in the paper.
This confirms successful reproducibility even if small numeric differences exist,
which are expected due to environment/model version differences.

------------------------------------------------------------
Git Submission Notes
------------------------------------------------------------
Do not push large checkpoints or environments.

Recommended .gitignore:
venv/
BERT_wiki/
__pycache__/
*.jsonl
*.bin
*.pt
*.log

------------------------------------------------------------
Citation
------------------------------------------------------------
@InProceedings{NAACL-2025-Maddela,
  author = {Maddela, Mounica and Alva-Manchego, Fernando},
  title = {Adapting Sentence-Level Metrics for Document-Level Simplification},
  booktitle = {NAACL},
  year = {2025}
}

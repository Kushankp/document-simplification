import json
from scipy.stats import pearsonr

dataset_path = "eval_datasets/dwiki_final.jsonl"   
agg_path = "agg_output_wiki.jsonl"                   

with open(dataset_path, "r", encoding="utf8") as f:
    dataset = [json.loads(line) for line in f if line.strip()]

with open(agg_path, "r", encoding="utf8") as f:
    agg_data = [json.loads(line) for line in f if line.strip()]

agg_map = {int(item["index"]): item for item in agg_data}

dims = [r["name"] for r in dataset[0]["ratings"]]
print("\nAvailable human rating dimensions:\n", dims)

target_dims = ["simplicity-word", "simplicity-sent", "meaning", "grammar"]

for dim in target_dims:
    human_vals = []
    agg_sari_vals = []
    agg_bert_vals = []

    for i, inst in enumerate(dataset):
        if i not in agg_map:
            continue
        hv = None
        for r in inst.get("ratings", []):
            if r.get("name") == dim:
                hv = r.get("agg_value")
                break
        if hv is None:
            continue
        agg_entry = agg_map[i]
        human_vals.append(hv)
        agg_sari_vals.append(agg_entry.get("Agg-SARI", 0.0))
        agg_bert_vals.append(agg_entry.get("Agg-BERTScore", 0.0))

    if len(human_vals) < 3:
        print(f"\nSkipping {dim} (too few samples)")
        continue

    corr_sari, _ = pearsonr(human_vals, agg_sari_vals)
    corr_bert, _ = pearsonr(human_vals, agg_bert_vals)

    print(f"\n=== Dimension: {dim} ===")
    print(f"Agg-SARI correlation:      {corr_sari:.4f}")
    print(f"Agg-BERTScore correlation: {corr_bert:.4f}")

print("\nDone. Compare these values to Table 3 / 4 in the paper.")

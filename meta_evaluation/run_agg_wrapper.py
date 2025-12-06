import json
from pathlib import Path

from metrics.agg_metric_graph import AggMetricGraph
from metrics.sari import SARI
from metrics.bscore import BERTScore

BERT_PATH = "BERT_wiki" 
INPUT = "output.jsonl"  
OUTPUT = "agg_output_wiki.jsonl"

def load_input(infile):
    """
    Robust loader: accepts either:
      - JSONL (one JSON object per line)
      - single JSON array file containing list of instances
    Returns a list of tuples: (complex_text, simple_text, references)
    """
    import json
    from pathlib import Path
    p = Path(infile)
    if not p.exists():
        raise FileNotFoundError(f"{infile} not found")

    text = p.read_text(encoding="utf8").strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            rows = []
            for obj in parsed:
                if isinstance(obj, dict):
                    complex_text = obj.get("original") or obj.get("complex") or obj.get("source")
                    simple_text = obj.get("simplification") or obj.get("simple") or obj.get("candidate")
                    references = obj.get("references") or obj.get("refs") or obj.get("reference") or []
                else:
                    continue
                rows.append((complex_text, simple_text, references))
            return rows
    except Exception:
        pass

    rows = []
    with p.open("r", encoding="utf8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                complex_text = obj.get("original") or obj.get("complex") or obj.get("source")
                simple_text = obj.get("simplification") or obj.get("simple") or obj.get("candidate")
                references = obj.get("references") or obj.get("refs") or obj.get("reference") or []
                rows.append((complex_text, simple_text, references))
            else:
                if isinstance(obj, list):
                    for sub in obj:
                        if isinstance(sub, dict):
                            complex_text = sub.get("original") or sub.get("complex") or sub.get("source")
                            simple_text = sub.get("simplification") or sub.get("simple") or sub.get("candidate")
                            references = sub.get("references") or sub.get("refs") or sub.get("reference") or []
                            rows.append((complex_text, simple_text, references))
    return rows


def main():
    p = Path(INPUT)
    if not p.exists():
        raise FileNotFoundError(f"Input file {INPUT} not found. Run evaluate_pearson.py first to get {INPUT}")

    rows = load_input(INPUT)
    complexes = [r[0] for r in rows]
    simples = [r[1] for r in rows]
    refs = [r[2] for r in rows]

    sari_metric = SARI()
    bscore_metric = BERTScore()

    agg_sari = AggMetricGraph(bert_path=BERT_PATH, sent_metric=sari_metric, refless=False, threshold=0.5)
    agg_bscore = AggMetricGraph(bert_path=BERT_PATH, sent_metric=bscore_metric, refless=False, threshold=0.5)

    print("Computing Agg-SARI ...")
    agg_sari_scores = agg_sari.compute_metric(complexes, simples, refs)
    print("Computing Agg-BERTScore ...")
    agg_bscore_scores = agg_bscore.compute_metric(complexes, simples, refs)

    with open(OUTPUT, "w", encoding="utf8") as out_fh:
        for i, (asari, abs_) in enumerate(zip(agg_sari_scores, agg_bscore_scores)):
            out_obj = {
                "index": i,
                "Agg-SARI": float(asari),
                "Agg-BERTScore": float(abs_)
            }
            json.dump(out_obj, out_fh)
            out_fh.write("\n")

    print("Wrote aggregated output to", OUTPUT)

if __name__ == "__main__":
    main()

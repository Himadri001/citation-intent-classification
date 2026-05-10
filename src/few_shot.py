import json
import os
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from llm_client import call_llm, parse_label, DEFAULT_MODEL
from prompts import few_shot_prompt, SCICITE_DEMOS, ACL_ARC_DEMOS
from data_prep import SCICITE_TAXONOMY, ACL_ARC_TAXONOMY, load_corpus, get_test_split

CACHE_PATH = "results/few_shot/predictions_{source}.json"


def run_few_shot(source="scicite", sample_size=300, model=DEFAULT_MODEL):
    taxonomy   = SCICITE_TAXONOMY if source == "scicite" else ACL_ARC_TAXONOMY
    demos      = SCICITE_DEMOS    if source == "scicite" else ACL_ARC_DEMOS
    cache_path = CACHE_PATH.format(source=source)
    corpus     = load_corpus()
    test_data  = get_test_split(corpus, source=source)

    if len(test_data) <= sample_size:
        sample = test_data
    else:
        _, sample = train_test_split(
            test_data,
            test_size=sample_size,
            stratify=[r["label"] for r in test_data],
            random_state=42,
        )

    done = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            done = json.load(f)
        print(f"Resuming from cache: {len(done)}/{len(sample)} done")

    for row in tqdm(sample, desc=f"Few-shot [{source}]"):
        if row["id"] in done:
            continue
        msgs = few_shot_prompt(row["text"], taxonomy, demos)
        raw  = call_llm(msgs, model=model, max_tokens=16)
        pred = parse_label(raw, taxonomy.keys())
        done[row["id"]] = {
            "text":  row["text"],
            "label": row["label"],
            "pred":  pred,
            "raw":   raw,
        }
        with open(cache_path, "w") as f:
            json.dump(done, f, indent=2)
        time.sleep(1.5)

    print(f"Few-shot complete: {len(done)} predictions saved to {cache_path}")
    return list(done.values())


if __name__ == "__main__":
    run_few_shot("scicite")
    run_few_shot("acl_arc")

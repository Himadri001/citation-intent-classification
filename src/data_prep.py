import json
from datasets import load_dataset

LABEL_MAP = {
    # SciCite
    "background": "background",
    "method":     "method",
    "result":     "result",
    # ACL-ARC
    "Background":        "background",
    "Uses":              "method",
    "CompareOrContrast": "result",
    "Extends":           "extends",
    "Motivation":        "motivation",
    "Future":            "future",
}

SCICITE_TAXONOMY = {
    "background": "Provides context, motivation, or situates the work.",
    "method":     "A method, tool, or dataset being used.",
    "result":     "Results are compared or contrasted.",
}

ACL_ARC_TAXONOMY = {
    "background":  "Provides general context or motivation.",
    "method":      "The cited work is used directly.",
    "result":      "Results are compared or contrasted.",
    "extends":     "The citing paper builds on the cited work.",
    "motivation":  "The cited work motivates the research question.",
    "future":      "Suggested as future direction.",
}


def build_corpus():
    scicite = load_dataset("allenai/scicite")
    acl_arc = load_dataset("zapsdcn/citation_intent", trust_remote_code=True)

    corpus = []
    idx = 0

    scicite_label_names = scicite["train"].features["label"].names  # ['method','background','result']
    for split in ["train", "validation", "test"]:
        for row in scicite[split]:
            label_str = scicite_label_names[row["label"]]
            corpus.append({
                "id": "scicite_" + split + "_" + str(idx),
                "text": row["string"],
                "label": label_str,
                "label_original": label_str,
                "source": "scicite",
                "split": split,
            })
            idx += 1

    for split in ["train", "validation", "test"]:
        for row in acl_arc[split]:
            corpus.append({
                "id": "acl_arc_" + split + "_" + str(idx),
                "text": row["text"],
                "label": LABEL_MAP.get(row["label"], row["label"]),
                "label_original": row["label"],
                "source": "acl_arc",
                "split": split,
            })
            idx += 1

    return corpus


def save_corpus(corpus, path="corpus/combined_corpus.json"):
    with open(path, "w") as f:
        json.dump(corpus, f, indent=2)
    print(f"Saved {len(corpus)} entries to {path}")


def load_corpus(path="corpus/combined_corpus.json"):
    with open(path) as f:
        return json.load(f)


def get_test_split(corpus, source="scicite"):
    return [c for c in corpus if c["source"] == source and c["split"] == "test"]


def get_train_split(corpus, source="scicite"):
    return [c for c in corpus if c["source"] == source and c["split"] == "train"]


if __name__ == "__main__":
    corpus = build_corpus()
    save_corpus(corpus)
    print(f"Total corpus size: {len(corpus)}")
    from collections import Counter
    labels = Counter(c["label"] for c in corpus)
    print("Label distribution:", dict(labels))

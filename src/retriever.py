import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ENCODER_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH    = "embeddings/corpus.index"
TEXTS_PATH    = "embeddings/corpus_texts.npy"
LABELS_PATH   = "embeddings/corpus_labels.npy"
IDS_PATH      = "embeddings/corpus_ids.npy"

encoder = None
index   = None
texts   = None
labels  = None
ids     = None


def build_index(corpus, batch_size=64):
    global encoder

    encoder = SentenceTransformer(ENCODER_MODEL)
    all_texts  = [c["text"]  for c in corpus]
    all_labels = [c["label"] for c in corpus]
    all_ids    = [c["id"]    for c in corpus]

    print(f"Encoding {len(all_texts)} passages...")
    embs = encoder.encode(
        all_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs.astype("float32"))

    faiss.write_index(idx, INDEX_PATH)
    np.save(TEXTS_PATH,  np.array(all_texts))
    np.save(LABELS_PATH, np.array(all_labels))
    np.save(IDS_PATH,    np.array(all_ids))
    print(f"Index saved: {idx.ntotal} vectors, dim={embs.shape[1]}")


def load_index():
    global encoder, index, texts, labels, ids
    if index is not None:
        return

    encoder = SentenceTransformer(ENCODER_MODEL)
    index   = faiss.read_index(INDEX_PATH)
    texts   = np.load(TEXTS_PATH,  allow_pickle=True)
    labels  = np.load(LABELS_PATH, allow_pickle=True)
    ids     = np.load(IDS_PATH,    allow_pickle=True)
    print(f"Index loaded: {index.ntotal} vectors")


def retrieve(query, k=5):
    load_index()
    q_emb = encoder.encode([query], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q_emb, k)
    return [
        {
            "text":  str(texts[i]),
            "label": str(labels[i]),
            "id":    str(ids[i]),
            "score": float(scores[0][j]),
        }
        for j, i in enumerate(idxs[0])
    ]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_prep import load_corpus

    corpus = load_corpus()
    build_index(corpus)

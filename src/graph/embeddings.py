"""Node2Vec embeddings per account using gensim (pure Python).

Produces a 64-dim vector per account. Saved as parquet with columns
emb_0 ... emb_63 plus account index.

Why gensim instead of PyG?
- PyG's Node2Vec requires pyg-lib (C++ extension, hard to install).
- gensim's Word2Vec over node2vec random walks gives equivalent results
  and installs in seconds with no compilation.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from loguru import logger
from node2vec import Node2Vec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRAPH_PATH = PROJECT_ROOT / "data" / "features" / "graph.gpickle"
OUT_PATH = PROJECT_ROOT / "data" / "features" / "graph_embeddings.parquet"

EMBEDDING_DIM = 64
WALK_LENGTH = 20
NUM_WALKS = 5
WINDOW = 10
MIN_COUNT = 1
WORKERS = 2
EPOCHS = 3


def train_node2vec() -> pd.DataFrame:
    logger.info(f"Loading graph from {GRAPH_PATH} ...")
    with open(GRAPH_PATH, "rb") as f:
        g: nx.DiGraph = pickle.load(f)
    logger.info(f"Graph: {g.number_of_nodes():,} nodes, {g.number_of_edges():,} edges")

    logger.info("Converting to undirected ...")
    ug = g.to_undirected()
    logger.info(f"Undirected: {ug.number_of_nodes():,} nodes, {ug.number_of_edges():,} edges")

    logger.info(
        f"Building Node2Vec (dim={EMBEDDING_DIM}, walks={NUM_WALKS}, length={WALK_LENGTH}) ..."
    )
    t0 = time.time()

    n2v = Node2Vec(
        ug,
        dimensions=EMBEDDING_DIM,
        walk_length=WALK_LENGTH,
        num_walks=NUM_WALKS,
        weight_key=None,  # no weighted walks (simpler, faster)
        workers=WORKERS,
        quiet=True,
    )

    logger.info("Training Word2Vec on random walks ...")
    model = n2v.fit(
        window=WINDOW,
        min_count=MIN_COUNT,
        sg=1,       # skip-gram (Node2Vec standard)
        workers=WORKERS,
        epochs=EPOCHS,
        seed=42,
    )
    logger.info(f"Training done in {time.time()-t0:.1f}s")

    logger.info("Extracting embeddings ...")
    nodes = list(ug.nodes())
    dim = model.wv.vector_size
    embeddings = np.zeros((len(nodes), dim), dtype="float32")
    missing = 0
    for i, node in enumerate(nodes):
        try:
            embeddings[i] = model.wv[str(node)]
        except KeyError:
            # Node never appeared in any walk (isolated); leave as zeros
            missing += 1
    logger.info(f"Embeddings shape: {embeddings.shape} — {missing} nodes with zero vectors")

    cols = [f"emb_{i}" for i in range(dim)]
    df = pd.DataFrame(embeddings, columns=cols, dtype="float32")
    df.insert(0, "account", nodes)
    df = df.set_index("account")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, engine="pyarrow")
    logger.success(f"Wrote {OUT_PATH} — {df.shape}")
    return df


def main() -> None:
    train_node2vec()


if __name__ == "__main__":
    sys.exit(main())

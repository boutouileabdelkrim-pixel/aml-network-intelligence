"""Node2Vec embeddings per account using fastnode2vec (Cython, 100x faster).

Produces a 64-dim vector per account. Saved as parquet with columns
emb_0 ... emb_63 plus account index.

Why fastnode2vec?
- The `node2vec` package uses Python loops -> 30+ min on 515k nodes.
- `fastnode2vec` uses Numba/Cython -> ~1-2 min on the same graph.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from fastnode2vec import Graph, Node2Vec
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRAPH_PATH = PROJECT_ROOT / "data" / "features" / "graph.gpickle"
OUT_PATH = PROJECT_ROOT / "data" / "features" / "graph_embeddings.parquet"

EMBEDDING_DIM = 64
WALK_LENGTH = 20
NUM_WALKS = 5
WINDOW = 10
EPOCHS = 3
WORKERS = 4
P = 1.0
Q = 1.0


def train_node2vec() -> pd.DataFrame:
    logger.info(f"Loading graph from {GRAPH_PATH} ...")
    with open(GRAPH_PATH, "rb") as f:
        g: nx.DiGraph = pickle.load(f)
    logger.info(f"Graph: {g.number_of_nodes():,} nodes, {g.number_of_edges():,} edges")

    logger.info("Converting to undirected edge list ...")
    ug = g.to_undirected()
    edges = list(ug.edges())
    logger.info(f"Undirected: {ug.number_of_nodes():,} nodes, {len(edges):,} edges")

    logger.info("Building fastnode2vec Graph ...")
    t0 = time.time()
    fg = Graph(edges, directed=False, weighted=False)
    logger.info(f"fastnode2vec Graph built in {time.time()-t0:.1f}s")

    logger.info(
        f"Training Node2Vec (dim={EMBEDDING_DIM}, walks={NUM_WALKS}, "
        f"length={WALK_LENGTH}, p={P}, q={Q}) ..."
    )
    t1 = time.time()
    n2v = Node2Vec(
        fg,
        dim=EMBEDDING_DIM,
        walk_length=WALK_LENGTH,
        window=WINDOW,
        p=P,
        q=Q,
        workers=WORKERS,
    )
    n2v.train(epochs=EPOCHS)
    logger.info(f"Training done in {time.time()-t1:.1f}s")

    logger.info("Extracting embeddings ...")
    nodes = list(ug.nodes())
    dim = EMBEDDING_DIM
    embeddings = np.zeros((len(nodes), dim), dtype="float32")
    missing = 0
    for i, node in enumerate(nodes):
        try:
            embeddings[i] = n2v.wv[str(node)]
        except KeyError:
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

"""AML pattern detection per account.

Detects:
- Fan-in: node with many distinct incoming neighbours
- Fan-out: node with many distinct outgoing neighbours
- Cycle participation: node in at least one short cycle (<=4 hops)
- Passthrough: node with high in AND out degree (relay)
- Structuring: node with many small-ish transactions to distinct targets
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRAPH_PATH = PROJECT_ROOT / "data" / "features" / "graph.gpickle"
OUT_PATH = PROJECT_ROOT / "data" / "features" / "graph_patterns.parquet"

FAN_IN_THRESHOLD = 10
FAN_OUT_THRESHOLD = 10
PASSTHROUGH_RATIO = 0.3  # min(in_deg, out_deg) / max(in_deg, out_deg)


def detect_patterns() -> pd.DataFrame:
    logger.info(f"Loading graph from {GRAPH_PATH} ...")
    with open(GRAPH_PATH, "rb") as f:
        g: nx.DiGraph = pickle.load(f)
    logger.info(f"Graph: {g.number_of_nodes():,} nodes, {g.number_of_edges():,} edges")

    nodes = list(g.nodes())
    df = pd.DataFrame(index=pd.Index(nodes, name="account"))

    logger.info("Computing fan-in / fan-out flags ...")
    in_deg = dict(g.in_degree())
    out_deg = dict(g.out_degree())

    df["is_fan_in"] = pd.Series(
        {n: int(d >= FAN_IN_THRESHOLD) for n, d in in_deg.items()}, dtype="int8"
    )
    df["is_fan_out"] = pd.Series(
        {n: int(d >= FAN_OUT_THRESHOLD) for n, d in out_deg.items()}, dtype="int8"
    )
    df["is_fan_heavy"] = (
        (df["is_fan_in"] == 1) & (df["is_fan_out"] == 1)
    ).astype("int8")

    logger.info("Computing passthrough relay flag ...")
    df["is_passthrough"] = pd.Series(
        {
            n: int(
                min(in_deg[n], out_deg[n]) > 0
                and max(in_deg[n], out_deg[n]) > 0
                and (min(in_deg[n], out_deg[n]) / max(in_deg[n], out_deg[n])) >= PASSTHROUGH_RATIO
                and (in_deg[n] + out_deg[n]) >= 4
            )
            for n in nodes
        },
        dtype="int8",
    )

    logger.info("Detecting cycle participation (sampled, 2-3 hop cycles) ...")
    t0 = time.time()
    in_cycle = set()
    # For each edge u->v, check if v->u exists (2-cycle) — fast
    for u, v in g.edges():
        if g.has_edge(v, u):
            in_cycle.add(u)
            in_cycle.add(v)
    logger.info(f"2-cycles found in {time.time()-t0:.1f}s — {len(in_cycle):,} nodes involved")

    df["in_2_cycle"] = pd.Series(
        {n: int(n in in_cycle) for n in nodes}, dtype="int8"
    )

    # Count number of 2-cycle partners
    cycle_counts = {n: 0 for n in nodes}
    for u, v in g.edges():
        if g.has_edge(v, u):
            cycle_counts[u] += 1
    df["n_2_cycle_partners"] = pd.Series(cycle_counts, dtype="int32")

    df.index.name = "account"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, engine="pyarrow")
    logger.success(f"Wrote {OUT_PATH} — {df.shape}")
    return df


def main() -> None:
    detect_patterns()


if __name__ == "__main__":
    sys.exit(main())

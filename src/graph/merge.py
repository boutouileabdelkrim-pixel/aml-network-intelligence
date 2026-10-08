"""Merge all graph features onto the tabular transaction table.

For each transaction, we attach graph features for BOTH the from_account
(prefix `gfrom_`) and to_account (prefix `gto_`). Final output is
data/features/full.parquet.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABULAR_PATH = PROJECT_ROOT / "data" / "features" / "tabular.parquet"
GRAPH_DIR = PROJECT_ROOT / "data" / "features"

TOPOLOGY = GRAPH_DIR / "graph_topology.parquet"
COMMUNITIES = GRAPH_DIR / "graph_communities.parquet"
PATTERNS = GRAPH_DIR / "graph_patterns.parquet"
EMBEDDINGS = GRAPH_DIR / "graph_embeddings.parquet"

OUT_PATH = GRAPH_DIR / "full.parquet"


def _load_and_prefix(path: Path, prefix: str) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")
    df.columns = [f"{prefix}{c}" for c in df.columns]
    df.index.name = "account"
    logger.info(f"Loaded {path.name}: {df.shape} (prefixed '{prefix}')")
    return df


def merge() -> Path:
    t0 = time.time()

    logger.info(f"Loading tabular features from {TABULAR_PATH} ...")
    df = pd.read_parquet(TABULAR_PATH, engine="pyarrow")
    logger.info(f"Tabular: {df.shape} — mem: {df.memory_usage(deep=True).sum()/1e9:.2f} GB")

    # ---- Load all graph features ----
    topo = _load_and_prefix(TOPOLOGY, "g_")
    comm = _load_and_prefix(COMMUNITIES, "gc_")
    patt = _load_and_prefix(PATTERNS, "gp_")
    emb = _load_and_prefix(EMBEDDINGS, "e_")

    graph_features = topo.join([comm, patt, emb], how="outer")
    logger.info(f"Combined graph features: {graph_features.shape}")
    del topo, comm, patt, emb
    gc.collect()

    # ---- Merge FROM side ----
    logger.info("Merging graph features on FROM side ...")
    df = df.merge(
        graph_features,
        left_on="from_account",
        right_index=True,
        how="left",
        suffixes=("", "_drop"),
    )
    # Rename merged graph columns to gfrom_*
    graph_cols = [c for c in df.columns if c.startswith(("g_", "gc_", "gp_", "e_"))]
    df = df.rename(columns={c: f"gfrom_{c}" for c in graph_cols})
    logger.info(f"After from-merge: {df.shape} — mem: {df.memory_usage(deep=True).sum()/1e9:.2f} GB")

    # ---- Merge TO side ----
    logger.info("Merging graph features on TO side ...")
    df = df.merge(
        graph_features,
        left_on="to_account",
        right_index=True,
        how="left",
        suffixes=("", "_drop"),
    )
    graph_cols2 = [
        c for c in df.columns
        if c.startswith(("g_", "gc_", "gp_", "e_")) and not c.startswith("gfrom_")
    ]
    df = df.rename(columns={c: f"gto_{c}" for c in graph_cols2})
    logger.info(f"After to-merge: {df.shape} — mem: {df.memory_usage(deep=True).sum()/1e9:.2f} GB")

    # Fill NaNs (accounts not in graph shouldn't happen but just in case)
    df = df.fillna(0.0)
    gc.collect()

    logger.info(f"Final shape: {df.shape} — mem: {df.memory_usage(deep=True).sum()/1e9:.2f} GB")
    logger.info(f"Writing {OUT_PATH} ...")
    df.to_parquet(OUT_PATH, index=False, engine="pyarrow")
    size_mb = OUT_PATH.stat().st_size / 1e6
    logger.success(f"Wrote {OUT_PATH} ({size_mb:.1f} MB) — total {time.time()-t0:.1f}s")
    return OUT_PATH


def main() -> None:
    merge()


if __name__ == "__main__":
    sys.exit(main())

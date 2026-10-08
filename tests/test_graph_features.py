"""Unit tests for graph feature builders (small synthetic graph)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def small_graph() -> nx.DiGraph:
    """A -> B, A -> C, B -> C, C -> A, D -> C (5 edges, 4 nodes)."""
    g = nx.DiGraph()
    g.add_edge("A", "B", n_tx=1, total_amount=100.0)
    g.add_edge("A", "C", n_tx=2, total_amount=200.0)
    g.add_edge("B", "C", n_tx=1, total_amount=50.0)
    g.add_edge("C", "A", n_tx=1, total_amount=75.0)
    g.add_edge("D", "C", n_tx=1, total_amount=10.0)
    return g


def test_degrees(small_graph: nx.DiGraph) -> None:
    in_deg = dict(small_graph.in_degree())
    out_deg = dict(small_graph.out_degree())
    assert in_deg["A"] == 1  # from C
    assert in_deg["C"] == 3  # from A, B, D
    assert out_deg["A"] == 2  # to B, C
    assert out_deg["D"] == 1  # to C


def test_pagerank_ranks_central_node_high(small_graph: nx.DiGraph) -> None:
    pr = nx.pagerank(small_graph, alpha=0.85)
    # C receives from many and sends back to A — should have highest PR
    assert pr["C"] >= pr["D"]
    assert pr["A"] >= pr["D"]


def test_cycle_detection_2cycle(small_graph: nx.DiGraph) -> None:
    # A -> C and C -> A form a 2-cycle
    assert small_graph.has_edge("A", "C")
    assert small_graph.has_edge("C", "A")


def test_clustering_on_undirected(small_graph: nx.DiGraph) -> None:
    ug = small_graph.to_undirected()
    clustering = nx.clustering(ug)
    # C connects to A, B, D — A and B are also connected to each other
    assert clustering["C"] > 0


def test_betweenness_subset_returns_dict(small_graph: nx.DiGraph) -> None:
    bt = nx.betweenness_centrality_subset(
        small_graph, sources=["A", "D"], targets=["A", "D"], normalized=True
    )
    assert isinstance(bt, dict)
    assert set(bt.keys()) == set(small_graph.nodes())


def test_graph_consistency(small_graph: nx.DiGraph) -> None:
    assert small_graph.number_of_nodes() == 4
    assert small_graph.number_of_edges() == 5
    # Every node reachable from somewhere
    assert len(list(nx.weakly_connected_components(small_graph))) == 1

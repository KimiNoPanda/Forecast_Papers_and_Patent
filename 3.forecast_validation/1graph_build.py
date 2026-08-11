"""
1graph_builder.py

Builds cumulative cluster-level citation graphs from papers + patent boost.

"""

import warnings
warnings.filterwarnings("ignore")

import gc
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from scipy import sparse
from collections import defaultdict
from dataclasses import dataclass



# 1.  LOADERS

def load_stable_concepts(cluster_dir: str, year: int) -> pd.DataFrame:
    """Load stable_clustered_concepts.csv for a year."""

    path = f"{cluster_dir}/stable_clustered_concepts_provenance_{year}_mergedV3.csv" # Change name if needed
    df = pd.read_csv(path)

    # use stable_cluster as canonical ID
    if "stable_cluster" in df.columns:
        df["cluster"] = df["stable_cluster"]
    df["cluster"] = df["cluster"].astype(str)
    df = df[df["cluster"] != "-1"].copy()
    keep = [c for c in ["norm_concept", "cluster", "paper_freq",
                         "patent_freq", "shared", "mean_score"] if c in df.columns]
    return df[keep]


def load_provenance(provenance_dir: str, year: int) -> pd.DataFrame:
    """Load concepts provenance as DataFrame for a given year.""" 

    path = f"{provenance_dir}/concepts_provenance_{year}.parquet" # Change name if needed
    df = pd.read_parquet(path)

    # normalize column names
    df.columns = df.columns.str.lower().str.strip()

    # ensure expected columns exist
    assert "source_id" in df.columns, f"Missing source_id in provenance {year}: {df.columns.tolist()}"
    assert "norm_concept" in df.columns, f"Missing norm_concept in provenance {year}: {df.columns.tolist()}"
    if "source_type" not in df.columns:
        print(f"[warn] no source_type in provenance {year} — assuming all paper")
        df["source_type"] = "paper"
    return df[["source_id", "norm_concept", "source_type",
               "score"] if "score" in df.columns else
              ["source_id", "norm_concept", "source_type"]]


def load_references(network_dir: str, year: int) -> dict:
    """Return {paper_id: [ref_id, ...]} for a given year."""

    path = f"{network_dir}/papers_referenced_works_{year}.parquet" # Change name if needed
    df = pd.read_parquet(path, columns=["id", "referenced_works"])
    return dict(zip(df["id"], df["referenced_works"]))


def load_paper_stats(papers_data_dir: str, year: int) -> pd.DataFrame:
    """Load paper-level stats (fwci, citations) for a given year."""

    path = f"{papers_data_dir}/papers_{year}.parquet" # Change name if needed
    df = pd.read_parquet(path)
    df.columns = df.columns.str.lower().str.strip()

    # normalize id column 
    id_candidates = ["id", "paper_id", "source_id", "work_id"]
    id_col = next((c for c in id_candidates if c in df.columns), None)
    assert id_col, f"No id column found in papers {year}: {df.columns.tolist()}"
    df = df.rename(columns={id_col: "source_id"})

    # normalize fwci 
    fwci_col = next((c for c in ["fwci", "field_weighted_citation_impact"]
                     if c in df.columns), None)
    if fwci_col and fwci_col != "fwci":
        df = df.rename(columns={fwci_col: "fwci"})
    if "fwci" not in df.columns:
        print(f"[warn] no fwci in papers {year} — defaulting to 0")
        df["fwci"] = 0.0

    # normalize citations
    cit_col = next((c for c in ["cited_by_count", "citations", "citation_count"]
                    if c in df.columns), None)
    if cit_col and cit_col != "citations":
        df = df.rename(columns={cit_col: "citations"})
    if "citations" not in df.columns:
        print(f"[warn] no citations in papers {year} — defaulting to 0")
        df["citations"] = 0

    return df[["source_id", "fwci", "citations"]]



# 2.  CONCEPT → CLUSTER MAP  (global, cross-year)


def build_year_concept2cluster(
    stable_df: pd.DataFrame,
) -> dict[str, str]:
    """
    Use the stable_cluster assignments stored in this year's file.
    """
    df = stable_df.copy()

    df["norm_concept"] = (
        df["norm_concept"]
        .astype(str)
        .str.strip()
    )

    df["cluster"] = (
        df["cluster"]
        .astype(str)
        .str.strip()
    )

    df = df[df["cluster"] != "-1"].copy()

    # Resolve duplicate concept strings within a year.
    freq_cols = [
        col for col in
        ["total_freq", "paper_freq", "patent_freq"]
        if col in df.columns
    ]

    if freq_cols:
        df["_mapping_freq"] = (
            df[freq_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .sum(axis=1)
        )

        df = df.sort_values(
            "_mapping_freq",
            ascending=False,
        )

    duplicates = df.duplicated(
        "norm_concept",
        keep=False,
    )

    conflicting = (
        df.loc[duplicates]
        .groupby("norm_concept")["cluster"]
        .nunique()
    )

    n_conflicting = int((conflicting > 1).sum())

    if n_conflicting:
        print(
            f"  [warn] {n_conflicting:,} concepts have conflicting "
            "stable IDs within this year; selecting the highest-frequency row"
        )

    df = df.drop_duplicates(
        "norm_concept",
        keep="first",
    )

    return dict(
        zip(df["norm_concept"], df["cluster"])
    )


# 3.  SOURCE → CLUSTER MAP  (from provenance, per year)

def build_source2clusters(
    provenance_df: pd.DataFrame,
    concept2cluster: dict,
    max_clusters_per_source: int = 3,
    min_cluster_share: float = 0.15,
) -> dict:
    """
    Map each source to a small weighted set of stable clusters.

    For each source:
    1. Map concepts to stable clusters.
    2. Aggregate concept scores within each cluster.
    3. Keep at most the strongest `max_clusters_per_source` clusters.
    4. Remove clusters contributing less than `min_cluster_share`
       of that source's total mapped score.

    Returns {source_id: {cluster_id: normalized_weight}}
    """

    prov = provenance_df.copy()

    prov["cluster"] = (
        prov["norm_concept"]
        .map(concept2cluster)
    )

    prov = prov.dropna(
        subset=["source_id", "cluster"]
    ).copy()

    prov["cluster"] = prov["cluster"].astype(str)

    # Use provenance score when available.
    if "score" in prov.columns:
        prov["assignment_score"] = pd.to_numeric(
            prov["score"],
            errors="coerce",
        ).fillna(0.0)

        # Concepts with missing/zero score should still contribute.
        prov.loc[
            prov["assignment_score"] <= 0,
            "assignment_score",
        ] = 1.0
    else:
        prov["assignment_score"] = 1.0

    # Aggregate all concept evidence for each source-cluster pair.
    source_cluster = (
        prov.groupby(
            ["source_id", "cluster"],
            as_index=False,
        )
        .agg(
            cluster_score=("assignment_score", "sum"),
            n_concepts=("norm_concept", "nunique"),
        )
    )

    result = {}

    for source_id, group in source_cluster.groupby(
        "source_id",
        sort=False,
    ):
        group = group.sort_values(
            ["cluster_score", "n_concepts"],
            ascending=False,
        ).copy()

        total_score = group["cluster_score"].sum()

        if total_score <= 0:
            continue

        group["share"] = (
            group["cluster_score"] / total_score
        )

        # Keep meaningful assignments.
        selected = group[
            group["share"] >= min_cluster_share
        ].head(max_clusters_per_source)

        # Ensure every mapped source retains at least one cluster.
        if selected.empty:
            selected = group.head(1)

        # Renormalize after filtering.
        selected_total = selected["cluster_score"].sum()

        result[source_id] = {
            row.cluster: float(
                row.cluster_score / selected_total
            )
            for row in selected.itertuples(index=False)
        }

    return result


# 4.  PATENT NODE BOOST 

def compute_cluster_patent_boost(stable_df: pd.DataFrame,
                                 year: int) -> pd.DataFrame:
    """
    Aggregate patent signal from stable concepts to cluster level.
    Returns DataFrame with columns: cluster, total_patent_freq, total_paper_freq,
    n_shared, n_concepts, shared_ratio, patent_ratio, patent_boost, year, patent_lead_signal
    """

    valid = stable_df[stable_df["cluster"].notna()].copy()
    valid = valid[valid["cluster"] != "-1"]
    valid["cluster"] = valid["cluster"].astype(str)

    if "shared" not in valid.columns:
        valid["shared"] = (valid["paper_freq"] > 0) & (valid["patent_freq"] > 0)

    agg = (
        valid.groupby("cluster")
        .agg(
            total_patent_freq=("patent_freq",    "sum"),
            total_paper_freq =("paper_freq",     "sum"),
            n_shared         =("shared",          "sum"),
            n_concepts       =("norm_concept",   "count"),
        )
        .reset_index()
    )

    agg["shared_ratio"] = agg["n_shared"] / agg["n_concepts"].clip(lower=1)
    agg["patent_ratio"] = (
        agg["total_patent_freq"] /
        (agg["total_patent_freq"] + agg["total_paper_freq"]).clip(lower=1)
    )
    agg["patent_boost_raw"] = np.log1p(agg["total_patent_freq"])
    agg["patent_lead_signal"] = (
        agg["total_patent_freq"] / 
        (agg["total_paper_freq"] + 1)  # high = patents ahead of papers
    )
    agg["patent_boost"] = (
        np.log1p(agg["total_patent_freq"]) * 
        np.log1p(agg["patent_lead_signal"])
    )
    agg["year"] = year

    return agg[["cluster", "total_patent_freq", "total_paper_freq",
                "n_shared", "n_concepts", "shared_ratio",
                "patent_ratio", "patent_boost", "year", "patent_lead_signal"]]


# 5.  PATENT EDGE BOOST  

def compute_patent_edge_boost(provenance_df: pd.DataFrame,
                              concept2cluster: dict) -> dict:
    """
    For each patent, find all clusters its concepts belong to.
    Every pair of clusters co-occurring in a patent gets a boost count.

    boost[(src_cluster, tgt_cluster)] = number of patents co-mentioning both

    Returns dict of {(src_cluster, tgt_cluster): count}
    """

    patent_df = provenance_df[
        provenance_df["source_type"] == "patent"].copy()
    patent_df["cluster"] = patent_df["norm_concept"].map(concept2cluster)
    patent_df = patent_df.dropna(subset=["cluster"])

    patent2clusters = (
        patent_df.groupby("source_id")["cluster"]
        .apply(set)
        .to_dict()
    )

    edge_counts = defaultdict(int)
    for patent_id, clusters in patent2clusters.items():
        clusters = list(clusters)
        for i in range(len(clusters)):
            for j in range(len(clusters)):
                if i == j:
                    continue
                edge_counts[(clusters[i], clusters[j])] += 1

    print(f"  Patent edge boosts: {len(edge_counts):,} cluster pairs "
          f"from {len(patent2clusters):,} patents")
    return dict(edge_counts)


def compute_cluster_quality(provenance_df, paper_stats_df, concept2cluster):
    """
    For each cluster, compute mean fwci and mean citations
    across all papers whose concepts map to that cluster.
    Returns DataFrame with columns: cluster, mean_fwci, mean_citations, n_papers
    """

    # paper source_ids only
    paper_prov = provenance_df[
        provenance_df["source_type"] == "paper"][["source_id","norm_concept"]].copy()
    paper_prov["cluster"] = paper_prov["norm_concept"].map(concept2cluster)
    paper_prov = paper_prov.dropna(subset=["cluster"])

    merged = paper_prov.merge(paper_stats_df, on="source_id", how="left")
    
    quality = (
        merged.groupby("cluster")
        .agg(
            mean_fwci      =("fwci",      "mean"),
            mean_citations =("citations", "mean"),
            n_papers       =("source_id", "nunique"),
        )
        .reset_index()
    )
    return quality


# 6.  CITATION GRAPH BUILDER

def build_citation_graph_year(
    references: dict,
    source2clusters: dict,
    patent_node_boost: dict,
    patent_edge_boost: dict,
    alpha_node: float = 0.5,
    alpha_edge: float = 0.3,
) -> tuple[nx.DiGraph, pd.DataFrame]:
    """
    Build yearly directed citation graph at cluster level.

    Also reports how many citing papers and cited references can
    actually be mapped to stable clusters.
    """

    # Normalize patent edge boost.
    max_pat_edge = (
        max(patent_edge_boost.values())
        if patent_edge_boost
        else 1
    )

    norm_pat_edge = {
        key: value / max_pat_edge
        for key, value in patent_edge_boost.items()
    }

    # Citation mapping diagnostics

    n_citing_papers = 0
    n_mapped_citing_papers = 0

    n_all_references = 0
    n_references_from_mapped_sources = 0
    n_mapped_references = 0

    unique_missing_sources = set()
    unique_missing_targets = set()

    edge_weights = defaultdict(float)

    # Build citation edges

    for paper_id, ref_list in references.items():
        n_citing_papers += 1

        src_clusters = source2clusters.get(paper_id)

        if not src_clusters:
            unique_missing_sources.add(paper_id)
            n_all_references += len(ref_list)
            continue

        n_mapped_citing_papers += 1

        for ref_id in ref_list:
            n_all_references += 1
            n_references_from_mapped_sources += 1

            tgt_clusters = source2clusters.get(ref_id)

            if not tgt_clusters:
                unique_missing_targets.add(ref_id)
                continue

            n_mapped_references += 1

            for src_cluster, src_share in src_clusters.items():
                for tgt_cluster, tgt_share in tgt_clusters.items():
                    if src_cluster == tgt_cluster:
                        continue

                    edge_weights[(src_cluster, tgt_cluster)] += (
                        src_share * tgt_share
                    )

    # Print diagnostics

    citing_coverage = (
        n_mapped_citing_papers
        / max(n_citing_papers, 1)
    )

    reference_coverage_all = (
        n_mapped_references
        / max(n_all_references, 1)
    )

    reference_coverage_mapped_sources = (
        n_mapped_references
        / max(n_references_from_mapped_sources, 1)
    )

    print(
        f"  Citation mapping coverage:"
        f"\n    citing papers: "
        f"{n_mapped_citing_papers:,}/{n_citing_papers:,} "
        f"({citing_coverage:.1%})"
        f"\n    references, all citing papers: "
        f"{n_mapped_references:,}/{n_all_references:,} "
        f"({reference_coverage_all:.1%})"
        f"\n    references, mapped citing papers only: "
        f"{n_mapped_references:,}/"
        f"{n_references_from_mapped_sources:,} "
        f"({reference_coverage_mapped_sources:.1%})"
        f"\n    unique missing citing IDs: "
        f"{len(unique_missing_sources):,}"
        f"\n    unique missing cited IDs: "
        f"{len(unique_missing_targets):,}"
    )

    if not edge_weights:
        print("  [warn] No mapped inter-cluster citations for this year.")
        return nx.DiGraph(), pd.DataFrame()

    # Aggregate yearly edges
    edge_df = pd.DataFrame(
        (
            (u, v, w)
            for (u, v), w in edge_weights.items()
        ),
        columns=[
            "src_cluster",
            "tgt_cluster",
            "raw_weight",
        ],
    )

    del edge_weights
    gc.collect()


    # Reduce yearly graph density
    # Can change this according to your memory constraints and graph size.
    MIN_YEARLY_RAW_WEIGHT = 0.15
    MAX_OUT_EDGES_PER_CLUSTER = 150

    before = len(edge_df)

    # 1. Remove weak yearly relationships
    edge_df = edge_df[
        edge_df["raw_weight"] >= MIN_YEARLY_RAW_WEIGHT
    ].copy()

    after_threshold = len(edge_df)

    # 2. Keep only strongest outgoing relationships per source field
    edge_df = (
        edge_df
        .sort_values(
            ["src_cluster", "raw_weight"],
            ascending=[True, False],
        )
        .groupby(
            "src_cluster",
            group_keys=False,
        )
        .head(MAX_OUT_EDGES_PER_CLUSTER)
        .reset_index(drop=True)
    )

    after_topk = len(edge_df)

    print(
        f"  yearly edge pruning: "
        f"{before:,} -> {after_threshold:,} after weight threshold "
        f"-> {after_topk:,} after top-{MAX_OUT_EDGES_PER_CLUSTER}"
    )

    # Apply patent boosts

    edge_df["src_node_boost"] = (
        edge_df["src_cluster"]
        .map(patent_node_boost)
        .fillna(0)
    )

    edge_df["tgt_node_boost"] = (
        edge_df["tgt_cluster"]
        .map(patent_node_boost)
        .fillna(0)
    )

    edge_df["edge_boost"] = edge_df.apply(
        lambda row: norm_pat_edge.get(
            (
                row["src_cluster"],
                row["tgt_cluster"],
            ),
            0,
        ),
        axis=1,
    )

    edge_df["boosted_weight"] = (
        edge_df["raw_weight"]
        * (
            1
            + alpha_node
            * edge_df["src_node_boost"]
        )
        * (
            1
            + alpha_node
            * edge_df["tgt_node_boost"]
        )
        * (
            1
            + alpha_edge
            * edge_df["edge_boost"]
        )
    )

    # Assemble NetworkX graph

    graph = nx.DiGraph()

    all_clusters = (
        set(edge_df["src_cluster"])
        | set(edge_df["tgt_cluster"])
    )

    for cluster in all_clusters:
        graph.add_node(
            cluster,
            patent_boost=patent_node_boost.get(
                cluster,
                0,
            ),
        )

    for row in edge_df.itertuples(index=False):
        graph.add_edge(
            row.src_cluster,
            row.tgt_cluster,
            raw_weight=row.raw_weight,
            weight=row.boosted_weight,
            src_node_boost=row.src_node_boost,
            tgt_node_boost=row.tgt_node_boost,
            edge_boost=row.edge_boost,
        )

    return graph, edge_df



# 7.  ACCUMULATOR 

@dataclass
class EdgeAccum:
    raw_weight: float = 0.0
    src_node_boost: float = 0.0
    tgt_node_boost: float = 0.0
    edge_boost: float = 0.0
    first_seen: int   = 0
    last_seen: int   = 0


@dataclass
class NodeAccum:
    patent_boost: float = 0.0
    first_seen: int   = 0


def accumulate(edge_state: dict,
               node_state: dict,
               yearly_G:   nx.DiGraph,
               year:       int) -> tuple[dict, dict]:
    """
    Merge yearly nx graph into plain dict accumulators.
    Returns updated (edge_state, node_state) dicts.
    """

    for node, attrs in yearly_G.nodes(data=True):
        if node not in node_state:
            node_state[node] = NodeAccum(
                patent_boost=attrs.get("patent_boost", 0),
                first_seen=year)
        else:
            node_state[node].patent_boost = attrs.get(
                "patent_boost", node_state[node].patent_boost)

    for u, v, attrs in yearly_G.edges(data=True):
        key = (u, v)
        rw  = attrs.get("raw_weight", 0)
        if key in edge_state:
            ea = edge_state[key]
            ea.raw_weight      += rw
            ea.src_node_boost   = attrs.get("src_node_boost", ea.src_node_boost)
            ea.tgt_node_boost   = attrs.get("tgt_node_boost", ea.tgt_node_boost)
            ea.edge_boost       = attrs.get("edge_boost",     ea.edge_boost)
            ea.last_seen        = year
        else:
            edge_state[key] = EdgeAccum(
                raw_weight=rw,
                src_node_boost=attrs.get("src_node_boost", 0),
                tgt_node_boost=attrs.get("tgt_node_boost", 0),
                edge_boost=attrs.get("edge_boost", 0),
                first_seen=year,
                last_seen=year)

    return edge_state, node_state


def rebuild_nx(edge_state:  dict,
               node_state:  dict,
               alpha_node:  float = 0.5,
               alpha_edge:  float = 0.3) -> nx.DiGraph:
    """
    Reconstruct nx.DiGraph from accumulators (used for GEXF export only).
    Returns nx.DiGraph with node/edge attributes.
    """

    G = nx.DiGraph()
    for node, ns in node_state.items():
        G.add_node(node,
                   patent_boost=ns.patent_boost,
                   first_seen=ns.first_seen)
    for (u, v), ea in edge_state.items():
        boosted = (
            ea.raw_weight
            * (1 + alpha_node * ea.src_node_boost)
            * (1 + alpha_node * ea.tgt_node_boost)
            * (1 + alpha_edge * ea.edge_boost)
        )
        G.add_edge(u, v,
                   raw_weight=ea.raw_weight,
                   weight=boosted,
                   src_node_boost=ea.src_node_boost,
                   tgt_node_boost=ea.tgt_node_boost,
                   edge_boost=ea.edge_boost,
                   first_seen=ea.first_seen,
                   last_seen=ea.last_seen)
    return G


# 8.  NETWORK METRICS 

def compute_network_metrics_sparse(
    edge_state:  dict,
    node_state:  dict,
    year:        int,
    alpha_node:  float = 0.5,
    alpha_edge:  float = 0.3,
    cluster_df:  pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute node metrics directly from edge_state/node_state.
    Returns (node_metrics_df, edge_metrics_df)
    """

    nodes = list(node_state.keys())
    node_idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    rows, cols, weights = [], [], []
    in_deg = defaultdict(float)
    out_deg = defaultdict(float)

    edge_rows = []
    for (u, v), ea in edge_state.items():
        if u not in node_idx or v not in node_idx:
            continue
        w = (
            ea.raw_weight
            * (1 + alpha_node * ea.src_node_boost)
            * (1 + alpha_node * ea.tgt_node_boost)
            * (1 + alpha_edge * ea.edge_boost)
        )
        rows.append(node_idx[u])
        cols.append(node_idx[v])
        weights.append(w)
        out_deg[u] += 1
        in_deg[v]  += 1

        edge_rows.append({
            "src_cluster":    u,
            "tgt_cluster":    v,
            "raw_weight":     ea.raw_weight,
            "boosted_weight": w,
            "src_node_boost": ea.src_node_boost,
            "tgt_node_boost": ea.tgt_node_boost,
            "edge_boost":     ea.edge_boost,
            "first_seen":     ea.first_seen,
            "last_seen":      ea.last_seen,
            "year":           year,
        })

    # sparse pagerank
    if rows:
        A = sparse.csr_matrix((weights, (rows, cols)), shape=(N, N))
        d = np.array(A.sum(axis=1)).flatten()
        d[d == 0] = 1
        D_inv = sparse.diags(1.0 / d)
        M = (D_inv @ A).T

        pr = np.full(N, 1.0 / N)
        damp = 0.85
        for _ in range(100):
            pr_new = damp * (M @ pr) + (1 - damp) / N
            if np.linalg.norm(pr_new - pr, 1) < 1e-4:
                break
            pr = pr_new
    else:
        pr = np.zeros(N)

    node_df = pd.DataFrame({
        "cluster":     nodes,
        "in_degree":   [in_deg.get(n, 0)         for n in nodes],
        "out_degree":  [out_deg.get(n, 0)         for n in nodes],
        "pagerank":    pr.tolist(),
        "patent_boost":[node_state[n].patent_boost for n in nodes],
        "year":        year,
    })

    if cluster_df is not None:
        cdf  = cluster_df.copy()
        cdf["cluster"] = cdf["cluster"].astype(str)
        keep = [c for c in ["cluster", "norm_concept", "total_freq",
                             "mean_score"] if c in cdf.columns]
        
        # aggregate to cluster level for merge
        agg_cols = {c: "first" for c in keep if c != "cluster"}
        cdf_agg  = cdf.groupby("cluster").agg(
            size=("norm_concept", "count"),
            **{c: (c, "first") for c in agg_cols if c in cdf.columns}
        ).reset_index()
        node_df = node_df.merge(cdf_agg, on="cluster", how="left")

    node_df = node_df.sort_values("pagerank", ascending=False)
    edge_df = pd.DataFrame(edge_rows)

    return node_df, edge_df


# 9.  TEMPORAL CHANGE TRACKING

def compute_temporal_changes(prev_edge_weights: dict,
                             edge_state: dict,
                             year: int) -> dict:
    """
    Compute temporal changes in edge weights.
    Returns dict with metrics:
        year, new_edges, total_edges, total_nodes,
        avg_edge_growth_rate, max_edge_growth_rate, num_edges_growing
    """

    prev_keys = set(prev_edge_weights)
    curr_keys = set(edge_state)
    new_edges = curr_keys - prev_keys

    growth_rates = []
    for k in curr_keys & prev_keys:
        w_prev = prev_edge_weights[k]
        w_curr = edge_state[k].raw_weight
        if w_prev > 0:
            growth_rates.append((w_curr - w_prev) / w_prev)

    return dict(
        year=year,
        new_edges=len(new_edges),
        total_edges=len(curr_keys),
        total_nodes=len(set(u for u, _ in curr_keys) |
                        set(v for _, v in curr_keys)),
        avg_edge_growth_rate=(float(np.mean(growth_rates))
                              if growth_rates else 0),
        max_edge_growth_rate=(float(np.max(growth_rates))
                              if growth_rates else 0),
        num_edges_growing=len(growth_rates),
    )


def compute_patent_signal_coverage(
    year: int,
    yearly_graph: nx.DiGraph,
    yearly_edges: pd.DataFrame,
    boost_df: pd.DataFrame,
) -> dict:
    """
    Compute patent-signal coverage for the final pruned yearly citation graph.

    """

    graph_nodes = pd.Index(
        [str(node) for node in yearly_graph.nodes()],
        name="cluster",
    )

    # Restrict the node denominator to clusters present in the yearly graph.
    node_boost = (
        boost_df[["cluster", "patent_boost", "total_patent_freq"]]
        .copy()
    )
    node_boost["cluster"] = node_boost["cluster"].astype(str)

    node_boost = (
        pd.DataFrame({"cluster": graph_nodes})
        .merge(node_boost, on="cluster", how="left")
        .fillna({
            "patent_boost": 0.0,
            "total_patent_freq": 0.0,
        })
    )

    n_nodes = len(node_boost)
    n_boosted_nodes = int(
        (node_boost["patent_boost"] > 0).sum()
    )

    n_edges = len(yearly_edges)

    if n_edges > 0 and "edge_boost" in yearly_edges.columns:
        n_boosted_edges = int(
            (yearly_edges["edge_boost"] > 0).sum()
        )
    else:
        n_boosted_edges = 0

    node_coverage = (
        n_boosted_nodes / n_nodes
        if n_nodes > 0
        else np.nan
    )

    edge_coverage = (
        n_boosted_edges / n_edges
        if n_edges > 0
        else np.nan
    )

    result = {
        "year": year,
        "n_graph_nodes": n_nodes,
        "n_boosted_nodes": n_boosted_nodes,
        "node_boost_coverage": node_coverage,
        "n_yearly_edges": n_edges,
        "n_boosted_edges": n_boosted_edges,
        "edge_boost_coverage": edge_coverage,
        "total_patent_frequency": float(
            boost_df["total_patent_freq"].sum()
        ),
    }

    print(
        "  Patent-signal coverage:"
        f"\n    boosted nodes: "
        f"{n_boosted_nodes:,}/{n_nodes:,} "
        f"({node_coverage:.2%})"
        f"\n    boosted citation edges: "
        f"{n_boosted_edges:,}/{n_edges:,} "
        f"({edge_coverage:.4%})"
    )

    return result

# 10.  CHECKPOINT

def save_checkpoint(out_dir, year, cumulative_state, all_changes):
    """
    Save checkpoint to disk as pickle file.
    """
    path = Path(out_dir) / f"checkpoint_{year}.pkl"
    for old in Path(out_dir).glob("checkpoint_*.pkl"):
        old.unlink()
    with open(path, "wb") as f:
        pickle.dump({"year": year,
                     "cumulative_G": cumulative_state,
                     "all_changes": all_changes}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"checkpoint saved → {path}")


def load_latest_checkpoint(out_dir):
    """
    Load latest checkpoint from disk if exists, else return None.
    """

    ckpts = sorted(Path(out_dir).glob("checkpoint_*.pkl"))
    if not ckpts:
        return None
    with open(ckpts[-1], "rb") as f:
        return pickle.load(f)


# 11.  FULL PIPELINE

def run_pipeline(
    cluster_dir:    str,
    provenance_dir: str,
    network_dir:    str,
    papers_dir:     str,
    year_start:     int,
    year_end:       int,
    out_dir:        str,
    alpha_node:     float = 0.5,
    alpha_edge:     float = 0.3,
    save_gexf:      bool  = False,
):
    """
    Run the full pipeline to build cumulative cluster-level citation graphs.
    Returns (edge_state, node_state) dicts for the final year.
    """

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    coverage_rows = []


    # resume from checkpoint
    ckpt = load_latest_checkpoint(out_dir)
    if ckpt:
        start_year = ckpt["year"] + 1

        cumulative_state = ckpt["cumulative_G"]

        edge_state = cumulative_state["edge_state"]
        node_state = cumulative_state["node_state"]

        cumulative_source2clusters = cumulative_state.get(
            "source2clusters",
            {},
        )

        all_changes = ckpt["all_changes"]

        print(
            f"Resuming from year {start_year}; "
            f"loaded {len(cumulative_source2clusters):,} "
            "cumulative source mappings"
        )
    else:
        start_year = year_start
        edge_state = {}
        node_state = {}
        cumulative_source2clusters = {}
        all_changes = []

    # year loop 
    for year in range(start_year, year_end + 1):
        print(f"\n── Year {year} ──────────────────────────────")


        # load inputs
        stable_df = load_stable_concepts(cluster_dir, year)
        provenance_df = load_provenance(provenance_dir, year)
        references = load_references(network_dir, year)
        paper_stats = load_paper_stats(papers_dir, year)
        print(f" concepts={len(stable_df):,}  "
              f" provenance={len(provenance_df):,}  "
              f" citing_papers={len(references):,}"
              f" paper_stats={len(paper_stats):,}")

        concept2cluster = build_year_concept2cluster(
            stable_df
        )

        # source→clusters from provenance (papers + patents)
        source2clusters = build_source2clusters(provenance_df, concept2cluster)
        print(f"  source→cluster map: {len(source2clusters):,} sources")

        cumulative_source2clusters.update(
            source2clusters
        )

        print(
            f"  source→cluster map: "
            f"{len(source2clusters):,} yearly sources; "
            f"{len(cumulative_source2clusters):,} cumulative sources"
        )

        cluster_quality = compute_cluster_quality(
            provenance_df, paper_stats, concept2cluster)
        # patent node boost
        boost_df          = compute_cluster_patent_boost(stable_df, year)
        patent_node_boost = dict(zip(boost_df["cluster"],
                                     boost_df["patent_boost"]))
        boost_df.to_parquet(out / f"patent_boost_{year}.parquet", index=False)

        # patent edge boost
        patent_edge_boost = compute_patent_edge_boost(
            provenance_df, concept2cluster)
        pd.DataFrame([
            {"src_cluster": k[0], "tgt_cluster": k[1], "co_count": v}
            for k, v in patent_edge_boost.items()
        ]).to_parquet(out / f"patent_edge_boost_{year}.parquet", index=False)

        # build yearly citation graph
        yearly_G, yearly_edges = build_citation_graph_year(
            references, cumulative_source2clusters,
            patent_node_boost, patent_edge_boost,
            alpha_node=alpha_node, alpha_edge=alpha_edge,
        )
        print(f"  yearly graph  nodes={yearly_G.number_of_nodes():,}"
              f"  edges={yearly_G.number_of_edges():,}")

        coverage = compute_patent_signal_coverage(
            year=year,
            yearly_graph=yearly_G,
            yearly_edges=yearly_edges,
            boost_df=boost_df,
        )

        # Replace an existing row if this year is rerun.
        coverage_rows = [
            row for row in coverage_rows
            if int(row["year"]) != year
        ]
        coverage_rows.append(coverage)

        coverage_df = (
            pd.DataFrame(coverage_rows)
            .sort_values("year")
        )

        coverage_df.to_csv(
            out / "patent_signal_coverage.csv",
            index=False,
        )

        del provenance_df, references, paper_stats
        gc.collect()

        # accumulate into plain dicts
        prev_weights = {k: v.raw_weight for k, v in edge_state.items()}
        edge_state, node_state = accumulate(
            edge_state, node_state, yearly_G, year)
        del source2clusters
        del patent_node_boost
        del patent_edge_boost
        del boost_df
        del concept2cluster
        del yearly_G
        gc.collect()

        print(f" cumulative nodes={len(node_state):,}"
              f" edges={len(edge_state):,}")

        # metrics
        node_metrics, edge_metrics = compute_network_metrics_sparse(
            edge_state, node_state, year,
            alpha_node=alpha_node, alpha_edge=alpha_edge,
            cluster_df=stable_df,
        )

        node_metrics = node_metrics.merge(
            cluster_quality[["cluster", "mean_fwci",
                            "mean_citations", "n_papers"]],
            on="cluster", how="left"
        )
        node_metrics.drop_duplicates(subset=["cluster", "year"], inplace=True)

        node_metrics.to_parquet(out / f"node_metrics_{year}.parquet",  index=False)
        edge_metrics.to_parquet(out / f"edge_metrics_{year}.parquet",  index=False)
        yearly_edges.to_parquet(out / f"yearly_edges_{year}.parquet",  index=False)

        del node_metrics
        del edge_metrics
        del yearly_edges
        del stable_df
        del cluster_quality
        gc.collect()

        # optional GEXF 
        if save_gexf:
            g_tmp = rebuild_nx(edge_state, node_state, alpha_node, alpha_edge)
            nx.write_gexf(g_tmp, out / f"cumulative_graph_{year}.gexf")
            del g_tmp
            gc.collect()

        all_changes.append(
            compute_temporal_changes(prev_weights, edge_state, year))
        save_checkpoint(out_dir, year,
                        {"edge_state": edge_state, "node_state": node_state,"source2clusters": cumulative_source2clusters,},
                        all_changes)

    # final outputs 
    pd.DataFrame(all_changes).to_parquet(
        out / "temporal_changes.parquet", index=False)
    if save_gexf:
        g_final = rebuild_nx(edge_state, node_state, alpha_node, alpha_edge)
        nx.write_gexf(g_final, out / "final_cumulative_graph.gexf")
        del g_final

    print(f"\nPipeline complete, outputs in {out_dir}")
    return edge_state, node_state



# ENTRY POINT

def main():
    """
    Entry point for command-line execution. Parses arguments and runs the pipeline.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_dir",    required=True,
                        help="Dir with stable_clustered_concepts.csv")
    parser.add_argument("--provenance_dir", required=True,
                        help="Dir with concepts provenance files")
    parser.add_argument("--network",        required=True,
                        help="Dir with papers_referenced_works_{year}.parquet")
    parser.add_argument("--papers_dir",     required=True,
                        help="Dir with paper files for node quality")
    parser.add_argument("--out",            default="network_output")
    parser.add_argument("--year_start",     type=int, default=2000)
    parser.add_argument("--year_end",       type=int, default=2024)
    parser.add_argument("--alpha_node",     type=float, default=0.3,
                        help="Patent node boost multiplier")
    parser.add_argument("--alpha_edge",     type=float, default=0.5,
                        help="Patent edge boost multiplier ")
    parser.add_argument("--gexf",           action="store_true",
                        help="Save GEXF files (slower, more disk)")
    args = parser.parse_args()

    run_pipeline(
        cluster_dir=args.cluster_dir,
        provenance_dir=args.provenance_dir,
        network_dir=args.network,
        papers_dir=args.papers_dir,
        year_start=args.year_start,
        year_end=args.year_end,
        out_dir=args.out,
        alpha_node=args.alpha_node,
        alpha_edge=args.alpha_edge,
        save_gexf=args.gexf,
    )


if __name__ == "__main__":
    main()





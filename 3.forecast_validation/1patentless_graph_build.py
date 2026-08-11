"""
Builds cumulative cluster-level citation graphs from papers 

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
    path = f"{cluster_dir}/patentless_stable_clustered_concepts_provenance_{year}_merged.csv" # Change name if needed
    df = pd.read_csv(path)

    # use stable_cluster as canonical ID
    if "stable_cluster" in df.columns:
        df["cluster"] = df["stable_cluster"]
    df["cluster"] = df["cluster"].astype(str)
    df = df[df["cluster"] != "-1"].copy()
    keep = [c for c in ["norm_concept", "cluster", "paper_freq",
                         "shared", "mean_score"] if c in df.columns]
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


# 4.  CITATION GRAPH BUILDER

def build_citation_graph_year(
    references:          dict,
    source2clusters:     dict,
) -> tuple[nx.DiGraph, pd.DataFrame]:
    """
    Build yearly directed citation graph at cluster level.
    """

    # build citation edges
    records = []
    for paper_id, ref_list in references.items():
        src_clusters = source2clusters.get(paper_id)
        if not src_clusters:
            continue
        n_src = len(src_clusters)

        for ref_id in ref_list:
            tgt_clusters = source2clusters.get(ref_id)
            if not tgt_clusters:
                continue
            for sc in src_clusters:
                for tc in tgt_clusters:
                    if sc == tc:
                        continue
                    records.append({
                        "src_cluster":    sc,
                        "tgt_cluster":    tc,
                        "weight_contrib": 1.0 / n_src,
                    })

    if not records:
        return nx.DiGraph(), pd.DataFrame()

    edge_df = (
        pd.DataFrame(records)
        .groupby(["src_cluster", "tgt_cluster"])["weight_contrib"]
        .sum()
        .reset_index(name="raw_weight")
    )

    MIN_YEARLY_RAW_WEIGHT = 0.5
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

    # assemble graph
    G = nx.DiGraph()
    all_clusters = (set(edge_df["src_cluster"]) |
                    set(edge_df["tgt_cluster"]))
    for c in all_clusters:
        G.add_node(c)

    for _, row in edge_df.iterrows():
            G.add_edge(
                row["src_cluster"], row["tgt_cluster"],
                raw_weight=row["raw_weight"],
                weight=row["raw_weight"],       # ← just use raw_weight
            )
    return G, edge_df



# 5.  ACCUMULATOR  


@dataclass
class EdgeAccum:
    raw_weight:     float = 0.0
    first_seen:     int   = 0
    last_seen:      int   = 0


@dataclass
class NodeAccum:
    first_seen:   int   = 0


def accumulate(edge_state: dict,
               node_state: dict,
               yearly_G:   nx.DiGraph,
               year:       int) -> tuple[dict, dict]:
    """Merge yearly nx graph into plain dict accumulators.
    Returns updated (edge_state, node_state) dicts.
    """
    for node, attrs in yearly_G.nodes(data=True):
        if node not in node_state:
            node_state[node] = NodeAccum(first_seen=year)

    for u, v, attrs in yearly_G.edges(data=True):
        key = (u, v)
        rw  = attrs.get("raw_weight", 0)
        if key in edge_state:
            ea = edge_state[key]
            ea.raw_weight      += rw
            ea.last_seen        = year
        else:
            edge_state[key] = EdgeAccum(
                raw_weight=rw,
                first_seen=year,
                last_seen=year)

    return edge_state, node_state


def rebuild_nx(edge_state:  dict,
               node_state:  dict,
    ) -> nx.DiGraph:
    """Reconstruct nx.DiGraph from accumulators (used for GEXF export only)."""
    G = nx.DiGraph()
    for node, ns in node_state.items():
        G.add_node(node,
                   first_seen=ns.first_seen)
    for (u, v), ea in edge_state.items():
        boosted = (
            ea.raw_weight
        )
        G.add_edge(u, v,
                   raw_weight=ea.raw_weight,
                   weight=boosted,
                   first_seen=ea.first_seen,
                   last_seen=ea.last_seen)
    return G


# 6.  NETWORK METRICS


def compute_network_metrics_sparse(
    edge_state:  dict,
    node_state:  dict,
    year:        int,
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
        "patent_boost": 0.0,
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



# 7.  TEMPORAL CHANGE TRACKING


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



# 8.  CHECKPOINT


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


# 9.  FULL PIPELINE

def run_pipeline(
    cluster_dir:    str,
    provenance_dir: str,
    network_dir:    str,
    papers_dir:     str,
    year_start:     int,
    year_end:       int,
    out_dir:        str,
    save_gexf:      bool  = False,
):
    """
    Run the full pipeline to build cumulative cluster-level citation graphs.
    Returns (edge_state, node_state) dicts for the final year.
    """

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)



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

        # build yearly citation graph
        yearly_G, yearly_edges = build_citation_graph_year(
            references, cumulative_source2clusters,
        )
        print(f"  yearly graph  nodes={yearly_G.number_of_nodes():,}"
              f"  edges={yearly_G.number_of_edges():,}")

        del provenance_df, references, paper_stats
        gc.collect()

        # accumulate into plain dicts
        prev_weights = {k: v.raw_weight for k, v in edge_state.items()}
        edge_state, node_state = accumulate(
            edge_state, node_state, yearly_G, year)
        del source2clusters
        del concept2cluster
        del yearly_G
        gc.collect()

        print(f" cumulative nodes={len(node_state):,}"
              f" edges={len(edge_state):,}")

        # metrics
        node_metrics, edge_metrics = compute_network_metrics_sparse(
            edge_state, node_state, year,
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
            g_tmp = rebuild_nx(edge_state, node_state)
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
        g_final = rebuild_nx(edge_state, node_state)
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
        save_gexf=args.gexf,
    )


if __name__ == "__main__":
    main()





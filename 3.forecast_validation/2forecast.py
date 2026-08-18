"""
2forecast.py

Forecast future cluster importance using LightGBM regression on node + edge features.

"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb


# 1.  LOADERS


def load_all_node_metrics(metrics_dir: str) -> pd.DataFrame:
    """
    Load all node-level metrics from parquet files in the given directory.
    Returns a DataFrame
    """

    files = sorted(Path(metrics_dir).glob("node_metrics_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No node_metrics_*.parquet in {metrics_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["cluster"] = df["cluster"].astype(str)
    return df


def load_all_edge_metrics(metrics_dir: str) -> pd.DataFrame:
    """
    Load all edge-level metrics from parquet files in the given directory.
    Returns a DataFrame
    """

    files = sorted(Path(metrics_dir).glob("edge_metrics_*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["src_cluster"] = df["src_cluster"].astype(str)
    df["tgt_cluster"] = df["tgt_cluster"].astype(str)
    return df

def load_edge_metrics_years(
    metrics_dir: str,
    years: list[int],
) -> pd.DataFrame:
    dfs = []

    for year in years:
        path = Path(metrics_dir) / f"edge_metrics_{year}.parquet"

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_parquet(path)
        df["src_cluster"] = df["src_cluster"].astype(str)
        df["tgt_cluster"] = df["tgt_cluster"].astype(str)

        if "year" not in df.columns:
            df["year"] = year

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def load_all_patent_boost(metrics_dir: str) -> pd.DataFrame:
    """
    Load all node-level patent boost metrics from parquet files in the given directory.
    Returns a DataFrame
    """

    files = sorted(Path(metrics_dir).glob("patent_boost_*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["cluster"] = df["cluster"].astype(str)
    return df


def load_all_patent_edge_boost(metrics_dir: str) -> pd.DataFrame:
    """
    Load all edge-level patent co-occurrence boost metrics from parquet files in the given directory.
    Returns a DataFrame
    """

    files = sorted(Path(metrics_dir).glob("patent_edge_boost_*.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        year = int(f.stem.split("_")[-1])
        d = pd.read_parquet(f)
        d["year"] = year
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df["src_cluster"] = df["src_cluster"].astype(str)
    df["tgt_cluster"] = df["tgt_cluster"].astype(str)
    return df


def load_temporal_changes(metrics_dir: str) -> pd.DataFrame:
    """
    Load temporal changes DataFrame from parquet file in the given directory.
    Returns a DataFrame
    """

    path = Path(metrics_dir) / "temporal_changes.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)

def load_stable_concepts_for_labels(concepts_dir: str, year: int) -> pd.DataFrame:
    """
    Load stable clustered concepts for a given year from CSV file.
    """
    path = Path(concepts_dir) / (
        f"stable_clustered_concepts_provenance_{year}_mergedV3.csv") # Adjust to your own path
    df = pd.read_csv(path)
    if "stable_cluster" in df.columns:
        df["cluster"] = df["stable_cluster"]
    df["cluster"] = df["cluster"].astype(str)
    df = df[df["cluster"] != "-1"].copy()
    keep = [c for c in ["norm_concept", "cluster", "paper_freq",
                         "patent_freq"] if c in df.columns]
    return df[keep]



# 2.  EDGE-LEVEL FEATURES → aggregate to node level


def compute_edge_node_features(edge_df: pd.DataFrame,
                               patent_edge_df: pd.DataFrame) -> pd.DataFrame:
    """
    From edge table, derive per-node per-year features.
    Also aggregates patent edge boost signal per node.
    Returns a DataFrame with one row per (cluster, year) and features derived from incoming and outgoing edges.
    """

    if edge_df.empty:
        return pd.DataFrame()

    # incoming edge features → target cluster
    in_feats = (
        edge_df.groupby(["tgt_cluster", "year"])
        .agg(
            in_strength      =("boosted_weight", "sum"),
            in_raw_strength  =("raw_weight",     "sum"),
            n_in_edges       =("src_cluster",    "count"),
            n_new_in_edges   =("first_seen",
                               lambda x: (x == edge_df.loc[x.index, "year"]).sum()),
            boost_received   =("src_node_boost", "mean"),
            # NEW: mean edge_boost on incoming edges
            mean_edge_boost_in=("edge_boost",    "mean"),
            max_edge_boost_in =("edge_boost",    "max"),
        )
        .reset_index()
        .rename(columns={"tgt_cluster": "cluster"})
    )

    # outgoing edge features → source cluster
    out_feats = (
        edge_df.groupby(["src_cluster", "year"])
        .agg(
            out_strength       =("boosted_weight", "sum"),
            out_raw_strength   =("raw_weight",     "sum"),
            n_out_edges        =("tgt_cluster",    "count"),
            boost_sent         =("tgt_node_boost", "mean"),
            mean_edge_boost_out=("edge_boost",     "mean"),
        )
        .reset_index()
        .rename(columns={"src_cluster": "cluster"})
    )

    result = in_feats.merge(out_feats, on=["cluster", "year"], how="outer")

    # patent edge boost aggregated per node per year
    if not patent_edge_df.empty:
        # how many patent co-occurrence pairs involve this cluster
        pat_in = (
            patent_edge_df.groupby(["tgt_cluster", "year"])
            .agg(patent_edge_in_sum  =("co_count", "sum"),
                 patent_edge_in_count=("co_count", "count"))
            .reset_index()
            .rename(columns={"tgt_cluster": "cluster"})
        )
        pat_out = (
            patent_edge_df.groupby(["src_cluster", "year"])
            .agg(patent_edge_out_sum  =("co_count", "sum"),
                 patent_edge_out_count=("co_count", "count"))
            .reset_index()
            .rename(columns={"src_cluster": "cluster"})
        )
        result = result.merge(pat_in,  on=["cluster", "year"], how="left")
        result = result.merge(pat_out, on=["cluster", "year"], how="left")

    return result


# 3.  ROLLING / LAG FEATURES


def add_lag_features(df: pd.DataFrame,
                     cols: list[str],
                     lags: list[int] = [1, 2, 3]) -> pd.DataFrame:
    """
    For each column in cols, add lagged features and rolling mean features.
    Returns a DataFrame with new columns added.
    """

    df = df.sort_values(["cluster", "year"])
    grp = df.groupby("cluster")
    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = grp[col].shift(lag)
        df[f"{col}_roll3"] = grp[col].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        df[f"{col}_vel"]   = grp[col].diff()
        df[f"{col}_accel"] = df.groupby("cluster")[f"{col}_vel"].diff()
    return df


def add_patent_lead_features(df: pd.DataFrame,
                             patent_df: pd.DataFrame,
                             leads: list[int] = [1, 2, 3]) -> pd.DataFrame:
    """
    Patent boost at year T is a LEADING indicator for citation activity
    at T+lead. For a row at year T we look up patent_boost at year T-lead
    Return a DataFrame with new columns added for each lead.
    """

    if patent_df.empty:
        return df

    cols = [c for c in ["patent_boost", "patent_ratio", "shared_ratio",
                        "total_patent_freq"] if c in patent_df.columns]
    boost_slim = patent_df[["cluster", "year"] + cols].copy()

    for lead in leads:
        shifted = boost_slim.copy()
        shifted["year"] = shifted["year"] + lead
        rename = {c: f"{c}_lead{lead}" for c in cols}
        shifted = shifted.rename(columns=rename)
        df = df.merge(shifted, on=["cluster", "year"], how="left")

    return df


# 4.  FULL FEATURE MATRIX

CORE_NODE_FEATURES = [
    "pagerank", "in_degree", "out_degree", "patent_boost",
    "mean_fwci", "mean_citations", "n_papers",
]


def build_feature_matrix(
    node_df:         pd.DataFrame,
    edge_df:         pd.DataFrame,
    patent_df:       pd.DataFrame,
    patent_edge_df:  pd.DataFrame,
    target_col:      str = "pagerank",
    horizon:         int = 3,
) -> pd.DataFrame:
    """
    Assemble the full feature matrix.
    For each (cluster, year) the TARGET is future log growth of target_col at year T+horizon.
    Features are all information available at year T.
    Returns a DataFrame with one row per (cluster, year) and features + target.
    """

    # node features
    available_core = [c for c in CORE_NODE_FEATURES if c in node_df.columns]
    feat = node_df[["cluster", "year"] + available_core].copy()
    feat = feat.fillna(0)

    # edge-derived features
    if not edge_df.empty:
        edge_node = compute_edge_node_features(edge_df, patent_edge_df)
        if not edge_node.empty:
            feat = feat.merge(edge_node, on=["cluster", "year"], how="left")

    # extra patent node features
    if not patent_df.empty:
        extra = [c for c in ["cluster", "year", "total_patent_freq",
                              "n_shared", "patent_ratio", "shared_ratio"]
                 if c in patent_df.columns]
        feat = feat.merge(patent_df[extra], on=["cluster", "year"], how="left")

    feat = feat.fillna(0)

    # lag / velocity features
    lag_cols = [c for c in available_core if c in feat.columns]
    if "in_strength" in feat.columns:
        lag_cols += ["in_strength", "out_strength"]
    if "patent_edge_in_sum" in feat.columns:
        lag_cols += ["patent_edge_in_sum", "patent_edge_out_sum"]
    feat = add_lag_features(feat, lag_cols)

    # patent lead features
    feat = add_patent_lead_features(feat, patent_df)

    feat = feat.fillna(0)


    # Target: future log growth rather than future absolute size.
    future = (
        node_df[["cluster", "year", target_col]]
        .rename(columns={
            target_col: "future_value",
            "year": "future_year",
        })
    )

    feat["future_year"] = feat["year"] + horizon

    feat = feat.merge(
        future,
        on=["cluster", "future_year"],
        how="left",
    )


    # Preserve current value explicitly.
    feat["current_value"] = feat[target_col].clip(lower=0)

    feat["target"] = (
        np.log1p(feat["future_value"].clip(lower=0))
        - np.log1p(feat["current_value"])
    )

    # Optional diagnostic columns.
    feat["target_abs_growth"] = (
        feat["future_value"] - feat["current_value"]
    )

    # sanity check
    if target_col in feat.columns:
        corr = feat[target_col].corr(feat["target"])
        print(f"  [sanity] corr({target_col}_T, target_T+{horizon}) = {corr:.4f}")

    feat = feat.drop(
        columns=[
            "future_value",
            "future_year",
            "current_value",
            "target_abs_growth",
        ],
        errors="ignore",
    )

    return feat

def build_cluster_label_map(concepts_dir: str, cutoff_year: int) -> dict:
    """
    Build a mapping from cluster ID to normalized concept label using stable clustered concepts.
    """
    df = load_stable_concepts_for_labels(concepts_dir, cutoff_year)
    if "paper_freq" in df.columns and "patent_freq" in df.columns:
        df["total_freq"] = df["paper_freq"].fillna(0) + df["patent_freq"].fillna(0)
    elif "paper_freq" in df.columns:
        df["total_freq"] = df["paper_freq"].fillna(0)
    else:
        df["total_freq"] = 1

    df_sorted = df.sort_values(["cluster", "total_freq"], ascending=[True, False])
    top = df_sorted.drop_duplicates(subset=["cluster"], keep="first")
    label_map = dict(zip(top["cluster"], top["norm_concept"]))
    print(f"  Built cluster label map: {len(label_map):,} clusters "
          f"(cutoff year {cutoff_year})")
    return label_map


# 5.  WALK-FORWARD CROSS-VALIDATION


def walk_forward_splits(df: pd.DataFrame,
                        min_train_years: int = 5,
                        test_size: int = 1):
    """
    Generator that yields (train_df, test_df) for walk-forward cross-validation.
    """
    years = sorted(df["year"].unique())
    for i in range(min_train_years, len(years) - test_size + 1):
        train_years = years[:i]
        test_years  = years[i: i + test_size]
        yield (
            df[df["year"].isin(train_years)],
            df[df["year"].isin(test_years)],
        )



# 6.  TRAIN


NON_FEATURE_COLS = {"cluster", "year", "target", "concept", "norm_concept",
                    "size", "total_freq_sum", "label", "future_year"}


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Returns a list of feature columns in the DataFrame, excluding non-feature columns.
    """
    return [c for c in df.columns
            if c not in NON_FEATURE_COLS
            and pd.api.types.is_numeric_dtype(df[c])]


def train_forecaster(feat_df:       pd.DataFrame,
                     n_estimators:  int   = 500,
                     learning_rate: float = 0.05) -> tuple:
    """
    Train LightGBM with walk-forward CV.
    Returns (final_model, scaler, feature_cols, cv_results_df)
    """
    feature_cols = get_feature_cols(feat_df)

    print(f"  [check] target mean={feat_df['target'].mean():.6f}  "
          f"std={feat_df['target'].std():.6f}")

    scaler  = RobustScaler()
    cv_rows = []

    for fold, (train_df, test_df) in enumerate(walk_forward_splits(feat_df)):
        X_train = scaler.fit_transform(train_df[feature_cols].fillna(0))
        y_train = train_df["target"].values
        X_test  = scaler.transform(test_df[feature_cols].fillna(0))
        y_test  = test_df["target"].values

        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=63,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
        preds = model.predict(X_test)

        mae     = mean_absolute_error(y_test, preds)
        rmse    = np.sqrt(mean_squared_error(y_test, preds))


        test_year = test_df["year"].iloc[0]
        cv_rows.append({"fold": fold, "test_year": test_year,
                        "mae": mae, "rmse": rmse,
                        "n_train": len(train_df), "n_test": len(test_df)})

        print(f"  fold {fold}  test_year={test_year}  "
              f"mae={mae:.4f}  rmse={rmse:.4f}")

    # final model on all data
    X_all = scaler.fit_transform(feat_df[feature_cols].fillna(0))
    y_all = feat_df["target"].values
    final_model = lgb.LGBMRegressor(
        n_estimators=n_estimators, learning_rate=learning_rate,
        num_leaves=63, min_child_samples=5,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    final_model.fit(X_all, y_all)

    cv_df = pd.DataFrame(cv_rows)
    print(f"\n  CV summary  mae={cv_df.mae.mean():.4f} ± {cv_df.mae.std():.4f}"
            f"  rmse={cv_df.rmse.mean():.4f} ± {cv_df.rmse.std():.4f}"
        )

    return final_model, scaler, feature_cols, cv_df



# 7.  FEATURE IMPORTANCE


def feature_importance_df(model, feature_cols: list[str]) -> pd.DataFrame:
    """
    Compute feature importance for a trained model.
    Returns a DataFrame with columns "feature" and "importance", sorted by importance descending.
    """

    imp = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    return imp



# 8.  PREDICT FUTURE YEARS


def predict_future(
    model,
    scaler,
    feature_cols: list[str],
    feat_df:      pd.DataFrame,
    horizon:      int,
    last_year:    int,
    target_col,
) -> pd.DataFrame:
    """
    Predict target for all clusters at last_year + horizon.
    Returns a DataFrame with columns: cluster, forecast_year, predicted_target, and other features.
    """

    latest = feat_df[feat_df["year"] == last_year].copy()
    if latest.empty:
        latest = feat_df[feat_df["year"] == feat_df["year"].max()].copy()

    X = scaler.transform(latest[feature_cols].fillna(0))
    latest = latest.copy()

    predicted_log_growth = model.predict(X)

    latest["predicted_log_growth"] = predicted_log_growth

    current = latest[target_col].clip(lower=0)

    # Convert predicted log growth back into a predicted future value.
    latest["predicted_target"] = (
        np.expm1(np.log1p(current)+ latest["predicted_log_growth"])).clip(lower=0)

    latest["predicted_growth"] = (latest["predicted_target"] - current)
    latest["predicted_growth_rel"] = (latest["predicted_growth"] / (current + 1e-9))
    latest["forecast_year"] = last_year + horizon

    keep = [
        "cluster",
        "forecast_year",
        "predicted_target",
        "predicted_growth",
        "predicted_growth_rel",
        "predicted_log_growth",
        target_col,
        "patent_boost",
        "pagerank",
        "in_degree",
    ]

    keep = list(dict.fromkeys(keep))

    for col in ["mean_fwci", "mean_citations", "n_papers",
            "patent_edge_in_sum","norm_concept"]:
        if col in latest.columns:
            keep.append(col)

    out = latest[[c for c in keep if c in latest.columns]]
    return out.sort_values("predicted_target", ascending=False)



# 9.  Detect emerging technologies

def detect_emerging_technologies(
    predictions:   pd.DataFrame,
    feat_df:       pd.DataFrame,
    patent_df:     pd.DataFrame,
    top_n:         int = 30,
    growth_metric: str = "in_degree",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify emerging technologies based on predicted growth and patent momentum.
    Returns two DataFrames: tier1 (rising fields) and tier2 (patent hotspots).
    """

    assert growth_metric in feat_df.columns, (
        f"growth_metric='{growth_metric}' not found in feat_df columns: "
        f"{sorted(feat_df.columns)}"
    )

    last_year = feat_df["year"].max()

    cluster_age = (
        feat_df.groupby("cluster")["year"]
        .min()
        .reset_index()
        .rename(columns={"year": "first_seen"})
    )

    now_col  = f"{growth_metric}_now"
    ago_col  = f"{growth_metric}_3y_ago"

    current = feat_df[feat_df["year"] == last_year][
        ["cluster", growth_metric, "mean_fwci",
         "mean_citations", "n_papers"]].copy()
    current = current.rename(columns={
        growth_metric:    now_col,
        "mean_fwci":      "mean_fwci_now",
        "mean_citations": "mean_citations_now",
        "n_papers":       "n_papers_now",
    })

    current = current.merge(cluster_age, on="cluster", how="left")
    current = current[current["first_seen"] <= last_year - 5].copy()

    past = feat_df[feat_df["year"] == last_year - 3][
        ["cluster", growth_metric]].rename(
        columns={growth_metric: ago_col})
    current = current.merge(past, on="cluster", how="left")
    current = current[current[ago_col] > 0].copy()
    current["abs_growth_3y"] = current[now_col] - current[ago_col]
    current["rel_growth_3y"] = current["abs_growth_3y"] / current[ago_col]

    # patent momentum
    if not patent_df.empty:
        pb = patent_df.sort_values(["cluster", "year"])
        pb["boost_vel"] = pb.groupby("cluster")["patent_boost"].diff()
        recent_boost = pb[pb["year"] == last_year][
            ["cluster", "boost_vel", "patent_boost"]]
    else:
        recent_boost = pd.DataFrame(
            columns=["cluster", "boost_vel", "patent_boost"])

    recent_boost = recent_boost.rename(columns={
        "patent_boost": "patent_boost_recent",
        "boost_vel":    "boost_vel_recent",
    })

    base = (
        predictions
        .merge(current,      on="cluster", how="left")
        .merge(recent_boost, on="cluster", how="left")
    )
    base = base.dropna(subset=["abs_growth_3y", "rel_growth_3y"])

    # patent edge velocity
    if "patent_edge_in_sum" in feat_df.columns:
        pe = feat_df[feat_df["year"] == last_year][
            ["cluster", "patent_edge_in_sum"]].copy()
        pe_prev = feat_df[feat_df["year"] == last_year - 1][
            ["cluster", "patent_edge_in_sum"]].rename(
            columns={"patent_edge_in_sum": "patent_edge_prev"})
        pe = pe.merge(pe_prev, on="cluster", how="left")
        pe["patent_edge_vel"] = (
            pe["patent_edge_in_sum"] - pe["patent_edge_prev"].fillna(0))
        base = base.merge(
            pe[["cluster", "patent_edge_vel"]], on="cluster", how="left")
    else:
        base["patent_edge_vel"] = 0

    quality_mask = pd.Series(True, index=base.index)
    if "mean_fwci_now" in base.columns:
        quality_mask &= (base["mean_fwci_now"] > 0)
    if "mean_citations_now" in base.columns:
        quality_mask &= (base["mean_citations_now"] > 0)
    if "n_papers_now" in base.columns:
        quality_mask &= (base["n_papers_now"] >= 25)
    base = base[quality_mask & (base["abs_growth_3y"] > 0)].copy()

    # Tier 1: Rising fields
    p25 = base[now_col].quantile(0.25)
    p75 = base[now_col].quantile(0.75)

    tier1 = base[
        (base[now_col] >= p25) &
        (base[now_col] <= p75)
    ].copy()

    tier1["score"] = (
        np.log1p(tier1["predicted_growth"].clip(lower=0))
        * np.log1p(tier1["predicted_growth_rel"].clip(lower=0))
        * np.log1p(tier1["abs_growth_3y"].clip(lower=0))
        * np.log1p(tier1["rel_growth_3y"].clip(lower=0))
        * np.log1p(tier1["mean_fwci_now"].clip(lower=0))
        * np.log1p(tier1["mean_citations_now"].clip(lower=0))
    )

    # Tier 2: Patent hotspots

    tier2 = base[
        (base["patent_boost_recent"] > 0)
        & (base["boost_vel_recent"].fillna(0) > 0)
        & (base[now_col] <= p75)
        & (base["predicted_growth"] > 0)
    ].copy()


    tier2["patent_to_academic_ratio"] = (
        tier2["patent_boost_recent"] / (tier2[now_col] + 1)
    )

    tier2["score"] = (
        np.log1p(tier2["predicted_growth"].clip(lower=0))
        * np.log1p(tier2["predicted_growth_rel"].clip(lower=0))
        * np.log1p(tier2["mean_fwci_now"].clip(lower=0))
        * np.log1p(tier2["mean_citations_now"].clip(lower=0))
        * (1 + 0.25* np.log1p(tier2["patent_boost_recent"].clip(lower=0)))
        * (1 +  0.25* np.log1p(tier2["boost_vel_recent"].fillna(0).clip(lower=0)))
        * (1 +  0.1*np.log1p(tier2["patent_edge_vel"].fillna(0).clip(lower=0)))
    )

    wanted_t1 = ["cluster", "forecast_year", "predicted_target", now_col,"predicted_growth","predicted_growth_rel", "predicted_log_growth",
                 "abs_growth_3y", "rel_growth_3y",
                 "mean_fwci_now", "mean_citations_now", "norm_concept", "score"]
    wanted_t2 = ["cluster", "forecast_year", "predicted_target", now_col,
                 "patent_boost_recent", "boost_vel_recent", "patent_to_academic_ratio",
                 "patent_edge_vel", "mean_fwci_now", "mean_citations_now",
                 "norm_concept", "score"]

    t1_cols = [c for c in wanted_t1 if c in tier1.columns]
    t2_cols = [c for c in wanted_t2 if c in tier2.columns]

    return (
        tier1.nlargest(top_n, "score")[t1_cols],
        tier2.nlargest(top_n, "score")[t2_cols],
    )


# 10. Validation of predicted tiers against the global network

def compute_true_global_metrics(tier_df: pd.DataFrame, 
                                growth_df: pd.DataFrame, 
                                top_k_tier: int = 10) -> dict:
    """
    Computes growth metrics of predicted tiers against the GLOBAL network.
    Returns a dictionary with average, median, min global percentiles and the percentage of clusters above the 75th percentile.
    """

    # Calculate global ranks ONCE on the entire network
    growth_df = growth_df.copy()
    growth_df["global_rank"] = growth_df["actual_growth"].rank(ascending=False)
    growth_df["global_percentile"] = growth_df["actual_growth"].rank(pct=True) * 100
    
    # Merge the pre-calculated global ranks into your predictions
    merged = tier_df.merge(
        growth_df[[
            "cluster",
            "actual",
            "actual_growth",
            "actual_growth_rel",
            "global_rank",
            "global_percentile"
        ]],
        on="cluster",
        how="left"
    )
    
    # Evaluate your Top K
    top_k = merged.nlargest(top_k_tier, "score")
    
    return {
        "avg_global_percentile": top_k["global_percentile"].mean(),
        "median_global_percentile": top_k["global_percentile"].median(),
        "min_global_percentile": top_k["global_percentile"].min(),
        "pct_above_75th": (top_k["global_percentile"] >= 75).mean(),
        "merged_df": merged  # Return this so we can still save the parquet
    }

def evaluate_ranked_predictions(
    ranked_df: pd.DataFrame,
    growth_df: pd.DataFrame,
    tier_name: str,
    ks: tuple[int, ...] = (10, 25, 50),
) -> pd.DataFrame:
    """
    Evaluate whether a tier's score correctly ranks future growth.
    Returns a DataFrame with evaluation metrics for each k in ks.
    """

    pred = ranked_df.copy()
    actual = growth_df.copy()

    pred["cluster"] = pred["cluster"].astype(str)
    actual["cluster"] = actual["cluster"].astype(str)

    cols = ["cluster", "actual_growth"]

    if "global_percentile" in actual.columns:
        cols.append("global_percentile")

    merged = pred.merge(
        actual[cols],
        on="cluster",
        how="inner",
    )

    merged["score"] = pd.to_numeric(
        merged["score"],
        errors="coerce",
    )

    merged["actual_growth"] = pd.to_numeric(
        merged["actual_growth"],
        errors="coerce",
    )

    merged = merged.dropna(
        subset=["score", "actual_growth"]
    ).copy()

    if merged.empty:
        print(f"\n  [{tier_name}] No matched rows for ranking evaluation.")
        return pd.DataFrame()

    # Preserve the detector's ranking.
    merged = merged.sort_values(
        "score",
        ascending=False,
    ).reset_index(drop=True)

    # Correlation across all returned candidates.
    if (
        len(merged) >= 3
        and merged["score"].nunique() > 1
        and merged["actual_growth"].nunique() > 1
    ):
        rho_growth, rho_growth_p = spearmanr(
            merged["score"],
            merged["actual_growth"],
        )
    else:
        rho_growth = np.nan
        rho_growth_p = np.nan

    if (
        "global_percentile" in merged.columns
        and merged["global_percentile"].nunique() > 1
    ):
        rho_percentile, rho_percentile_p = spearmanr(
            merged["score"],
            merged["global_percentile"],
        )
    else:
        rho_percentile = np.nan
        rho_percentile_p = np.nan

    print(f"\n  [{tier_name}] Internal ranking quality")
    print(
        f"    Spearman(score, actual growth): "
        f"{rho_growth:.3f}  p={rho_growth_p:.3g}"
    )

    if "global_percentile" in merged.columns:
        print(
            f"    Spearman(score, actual percentile): "
            f"{rho_percentile:.3f}  "
            f"p={rho_percentile_p:.3g}"
        )

    # Candidate-population baseline.
    population_growth = pd.to_numeric(
        actual["actual_growth"],
        errors="coerce",
    ).dropna()

    population_mean = population_growth.mean()
    population_median = population_growth.median()

    rows = []

    for k in ks:
        top = merged.head(k).copy()

        if top.empty:
            continue

        effective_k = len(top)

        mean_growth = top["actual_growth"].mean()
        median_growth = top["actual_growth"].median()

        # Relevance must be nonnegative for NDCG.
        relevance = top["actual_growth"].clip(lower=0).to_numpy()

        predicted_scores = top["score"].to_numpy()

        if (
            effective_k >= 2
            and np.any(relevance > 0)
        ):
            ndcg = ndcg_score(
                relevance.reshape(1, -1),
                predicted_scores.reshape(1, -1),
                k=effective_k,
            )
        else:
            ndcg = np.nan

        precision_top_quartile = np.nan
        mean_percentile = np.nan

        if "global_percentile" in top.columns:
            precision_top_quartile = (
                top["global_percentile"] >= 75
            ).mean()

            mean_percentile = (
                top["global_percentile"].mean()
            )

        lift_mean = (
            mean_growth / population_mean
            if population_mean > 0
            else np.nan
        )

        rows.append({
            "tier": tier_name,
            "k": effective_k,
            "ndcg": ndcg,
            "mean_actual_growth": mean_growth,
            "median_actual_growth": median_growth,
            "mean_global_percentile": mean_percentile,
            "precision_top_quartile": precision_top_quartile,
            "lift_vs_population_mean": lift_mean,
            "population_mean_growth": population_mean,
            "population_median_growth": population_median,
            "spearman_growth": rho_growth,
            "spearman_percentile": rho_percentile,
        })

        print(
            f"    @{effective_k}: "
            f"NDCG={ndcg:.3f}  "
            f"mean growth={mean_growth:.3f}  "
            f"median growth={median_growth:.3f}  "
            f"lift={lift_mean:.2f}x"
        )

        if "global_percentile" in top.columns:
            print(
                f"         mean percentile="
                f"{mean_percentile:.1f}%  "
                f"precision>P75="
                f"{precision_top_quartile:.1%}"
            )

    return pd.DataFrame(rows)

def evaluate_tier_overlap(
    tier1: pd.DataFrame,
    tier2: pd.DataFrame,
    ks: tuple[int, ...] = (10, 25, 50),
) -> pd.DataFrame:

    rows = []

    for k in ks:
        t1_ids = set(
            tier1.head(k)["cluster"].astype(str)
        )
        t2_ids = set(
            tier2.head(k)["cluster"].astype(str)
        )

        intersection = t1_ids & t2_ids
        union = t1_ids | t2_ids

        rows.append({
            "k": k,
            "overlap_count": len(intersection),
            "overlap_fraction_tier2": (
                len(intersection) / max(len(t2_ids), 1)
            ),
            "jaccard": (
                len(intersection) / max(len(union), 1)
            ),
            "tier2_unique_count": len(t2_ids - t1_ids),
        })

    return pd.DataFrame(rows)

def validate_held_out(
    train_metrics_dir: str,
    val_metrics_dir:   str,
    predictions:       pd.DataFrame,
    tier1:             pd.DataFrame,
    tier2:             pd.DataFrame,
    out_dir:           str,
    concepts_path:     str = "concept_provenance/stable_clustered_concepts_provenance_2022_mergedV3.csv",
    target:            str = "in_degree",
    horizon:           int = 3,
) -> dict:
    """
    Validate the forecast against actual data.

    two checks:
      1. Tier 1 rising fields 
      2. Tier 2 patent hotspots 

    Returns dict of result dataframes.
    """
    from scipy.stats import spearmanr

    out = Path(out_dir)

    # load actual data 
    node_1      = load_all_node_metrics(train_metrics_dir)
    node_2      = load_all_node_metrics(val_metrics_dir)

    train_max    = node_1["year"].max()          # 2022
    forecast_year = train_max + horizon            # 2025

    edge = pd.DataFrame()
    patent_edge = pd.DataFrame()

    if target in ("in_strength", "out_strength", "in_raw_strength", "out_raw_strength"):
        edge = load_edge_metrics_years(
            val_metrics_dir,
            years=[train_max, forecast_year],
        )
        patent_edge = load_all_patent_edge_boost(val_metrics_dir)
        if not edge.empty:
            strength = compute_edge_node_features(edge, patent_edge)
            if target in strength.columns:
                node_2 = node_2.merge(
                    strength[["cluster", "year", target]],
                    on=["cluster", "year"], how="left"
                )
                node_2[target] = node_2[target].fillna(0)

    assert forecast_year in node_2["year"].unique(), (
        f"forecast year {forecast_year} not in V5 "
        f"(available: {sorted(node_2['year'].unique())})"
    )
    print(f"\nValidating forecast …")
    print(f"  Train up to {train_max}, forecast year {forecast_year}")


    # load best concept per cluster 
    concepts = pd.read_csv(concepts_path)
    concepts["stable_cluster"] = concepts["stable_cluster"].astype(str)
    best_concept = (
        concepts
        .sort_values("total_freq", ascending=False)
        .drop_duplicates(subset=["stable_cluster"], keep="first")
        [["stable_cluster", "norm_concept"]]
        .rename(columns={"stable_cluster": "cluster"})
    )

    def add_concept(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["cluster"] = df["cluster"].astype(str)
        return df.merge(best_concept, on="cluster", how="left")

    # actual values at forecast year 
    actual_forecast = (
        node_2[node_2["year"] == forecast_year][["cluster", target]]
        .rename(columns={target: "actual"})
    )
    actual_forecast["cluster"] = actual_forecast["cluster"].astype(str)

    # actual values at train_max for growth computation
    actual_base = (
        node_2[node_2["year"] == train_max][["cluster", target]]
        .rename(columns={target: f"actual_{train_max}"})
    )
    actual_base["cluster"] = actual_base["cluster"].astype(str)

    # growth between train_max and forecast_year
    growth = actual_base.merge(actual_forecast, on="cluster", how="inner")
    growth["actual_growth"]     = growth["actual"] - growth[f"actual_{train_max}"]
    growth["actual_growth_rel"] = (
        growth["actual_growth"] / (growth[f"actual_{train_max}"] + 1e-9)
    )


    # SANITY CHECK: exact-zero growth 

    n_exact_zero = (growth["actual_growth"] == 0).sum()
    print(f"\n  [sanity] {n_exact_zero}/{len(growth)} clusters have exactly "
          f"zero growth ({n_exact_zero/len(growth)*100:.1f}%)")

    zero_mask = growth["actual_growth"] == 0
    both_zero = growth.loc[zero_mask, [f"actual_{train_max}", "actual"]].eq(0).all(axis=1)
    # reindex both_zero back to full growth index alignment
    both_zero_full = pd.Series(False, index=growth.index)
    both_zero_full.loc[zero_mask[zero_mask].index] = both_zero.values

    print(f"  [sanity] Zero-growth clusters where BOTH values are 0: "
          f"{both_zero_full.sum()}")
    print(f"  [sanity] Zero-growth clusters where values are EQUAL but "
          f"NONZERO (stale/duplicated snapshot?): "
          f"{(zero_mask & ~both_zero_full).sum()}")

    stale_examples = growth.loc[zero_mask & ~both_zero_full].head(10)
    if not stale_examples.empty:
        print(stale_examples[["cluster", f"actual_{train_max}", "actual"]].to_string(index=False))

    # Case A: missing-data-as-zero check
    missing_clusters = growth.loc[zero_mask & both_zero_full, "cluster"]
    if len(missing_clusters) > 0 and not edge.empty:
        in_edges = edge[
            edge["tgt_cluster"].isin(missing_clusters) &
            (edge["year"] == forecast_year)
        ]
        print(f"  [sanity] {in_edges['tgt_cluster'].nunique()}/{len(missing_clusters)} "
              f"'both-zero' clusters DO appear as tgt_cluster in edge at "
              f"{forecast_year} (if >0, a merge/dtype issue is masking real data as 0)")
        print(f"  [sanity] dtypes — edge.tgt_cluster: {edge['tgt_cluster'].dtype}, "
              f"node_2.cluster: {node_2['cluster'].dtype}")

    # Case B: stale/duplicated snapshot check (raw file-level)
    stale_clusters = growth.loc[zero_mask & ~both_zero_full, "cluster"].astype(str)
    if len(stale_clusters) > 0:
        try:
            e_train = pd.read_parquet(
                Path(train_metrics_dir) / f"edge_metrics_{train_max}.parquet")
            e_val = pd.read_parquet(
                Path(val_metrics_dir) / f"edge_metrics_{forecast_year}.parquet")
            s_train = e_train.groupby("tgt_cluster")["boosted_weight"].sum()
            s_val = e_val.groupby("tgt_cluster")["boosted_weight"].sum()
            comp = pd.DataFrame({
                "train_sum": s_train.reindex(stale_clusters),
                "val_sum": s_val.reindex(stale_clusters),
            }).dropna()
            if len(comp) > 0:
                identical_frac = (comp["train_sum"] == comp["val_sum"]).mean()
                print(f"  [sanity] File-level check: {identical_frac*100:.1f}% of "
                      f"stale clusters have IDENTICAL raw edge sums between "
                      f"{train_max} and {forecast_year} files "
                      f"(if high, val edge file wasn't rebuilt / points to wrong source)")
        except FileNotFoundError as e:
            print(f"  [sanity] Could not run file-level check: {e}")


    # tier 1 rising fields validation
    print(f"\n 1. Tier 1 rising fields: Global Percentile Validation ")

    t1 = add_concept(tier1.copy())
    t1["cluster"] = t1["cluster"].astype(str)
    
    now_col = f"{target}_now"
    if "predicted_growth" not in t1.columns:
        t1["predicted_growth"] = (
            t1["predicted_target"] - t1[now_col]
        )

    t1_metrics_10 = compute_true_global_metrics(t1, growth, top_k_tier=10)
    t1_check = t1_metrics_10["merged_df"]

    
    print(f"  Mean Global Percentile (Top 10): {t1_metrics_10['avg_global_percentile']:.1f}%")
    print(f"  Median Global Percentile:        {t1_metrics_10['median_global_percentile']:.1f}%")
    print(f"  Min Global Percentile:           {t1_metrics_10['min_global_percentile']:.1f}%")
    print(f"  % Clusters above 75th Percentile:{t1_metrics_10['pct_above_75th']:.0%}")

    t1_metrics_25 = compute_true_global_metrics(t1, growth, top_k_tier=25)

    print(f"  Mean Global Percentile (Top 25): {t1_metrics_25['avg_global_percentile']:.1f}%")
    print(f"  Median Global Percentile:        {t1_metrics_25['median_global_percentile']:.1f}%")
    print(f"  Min Global Percentile:           {t1_metrics_25['min_global_percentile']:.1f}%")
    print(f"  % Clusters above 75th Percentile:{t1_metrics_25['pct_above_75th']:.0%}")
    display_cols = [c for c in [
        "cluster",
        "norm_concept",
        "score",
        now_col,
        "predicted_target",
        "predicted_growth",
        f"actual_{train_max}",
        "actual",
        "actual_growth",
        "global_rank",
        "global_percentile"
    ] if c in t1_check.columns]

    print(f"\n  Top 10 Tier-1 details:")
    print(t1_check.nlargest(10, "score")[display_cols].to_string(index=False))

    tier1_rank_metrics = evaluate_ranked_predictions(
        ranked_df=t1,
        growth_df=growth,
        tier_name="Tier 1",
        ks=(10, 25, 50),
    )

    t1_check[display_cols].to_parquet(
        out / "validation_tier1_rising_fields.parquet", index=False)

    # tier 2 patent hotspots validation 
    print(f"\n 2. Tier 2 patent hotspots: Global Percentile Validation ")

    t2 = add_concept(tier2.copy())
    t2["cluster"] = t2["cluster"].astype(str)
    
    

    if "predicted_growth" not in t2.columns:
        t2["predicted_growth"] = (
            t2["predicted_target"] - t2[now_col]
        )
        
    t2_metrics_10 = compute_true_global_metrics(t2, growth, top_k_tier=10)
    t2_check = t2_metrics_10["merged_df"]

    
    print(f"  Mean Global Percentile (Top 10): {t2_metrics_10['avg_global_percentile']:.1f}%")
    print(f"  Median Global Percentile:        {t2_metrics_10['median_global_percentile']:.1f}%")
    print(f"  Min Global Percentile:           {t2_metrics_10['min_global_percentile']:.1f}%")
    print(f"  % Clusters above 75th Percentile:{t2_metrics_10['pct_above_75th']:.0%}")

    t2_metrics_25 = compute_true_global_metrics(t2, growth, top_k_tier=25)

    print(f"  Mean Global Percentile (Top 25): {t2_metrics_25['avg_global_percentile']:.1f}%")
    print(f"  Median Global Percentile:        {t2_metrics_25['median_global_percentile']:.1f}%")
    print(f"  Min Global Percentile:           {t2_metrics_25['min_global_percentile']:.1f}%")
    print(f"  % Clusters above 75th Percentile:{t2_metrics_25['pct_above_75th']:.0%}")

    display_cols = [c for c in [
        "cluster", "norm_concept", "score", now_col,
        "patent_boost_recent", "boost_vel_recent", 
        f"actual_{train_max}", "actual", "actual_growth", "actual_growth_rel",
        "global_rank", "global_percentile"
    ] if c in t2_check.columns]

    tier2_rank_metrics = evaluate_ranked_predictions(
        ranked_df=t2,
        growth_df=growth,
        tier_name="Tier 2",
        ks=(10, 25, 50),
    )

    print(f"\n  Top 10 Tier-2 details:")
    print(t2_check.nlargest(10, "score")[display_cols].to_string(index=False))

    t2_check.to_parquet(out / "validation_tier2_patent_hotspots.parquet", index=False)

    overlap_df = evaluate_tier_overlap(
        t1_check,
        t2_check,
    )

    print("\n  Tier-1/Tier-2 overlap:")
    print(overlap_df.to_string(index=False))

    overlap_df.to_csv(
        out / "tier_overlap.csv",
        index=False,
    )

    ranking_metrics = pd.concat(
        [tier1_rank_metrics, tier2_rank_metrics],
        ignore_index=True,
    )

    ranking_metrics.to_csv(
        out / "ranking_validation_metrics.csv",
        index=False,
    )

    return {
        "growth":         growth,
    }

# 11.  FULL PIPELINE

def add_cluster_labels(df: pd.DataFrame, label_map: dict) -> pd.DataFrame:
    """
    Add human-readable cluster labels to a DataFrame based on a mapping from cluster ID to label.
    """
    df = df.copy()
    df["cluster"] = df["cluster"].astype(str)
    df["cluster_label"] = df["cluster"].map(label_map)
    n_missing = df["cluster_label"].isna().sum()
    if n_missing:
        print(f"  [warn] {n_missing}/{len(df)} clusters had no label match "
              f"-- keeping raw cluster ID for those rows")
        df["cluster_label"] = df["cluster_label"].fillna(df["cluster"])
    cols = df.columns.tolist()
    cols.remove("cluster_label")
    cols.insert(cols.index("cluster") + 1, "cluster_label")
    return df[cols]

def run_forecast_pipeline(
    metrics_dir: str,
    val_metrics_dir: str,
    out_dir:     str,
    concepts_dir: str,
    target:      str = "pagerank",
    horizon:     int = 3,
):
    """
    Run the full forecast pipeline: load data, build features, train model, predict future, and validate against held-out data.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    node_df        = load_all_node_metrics(metrics_dir)
    edge_df        = load_all_edge_metrics(metrics_dir)
    patent_df      = load_all_patent_boost(metrics_dir)
    patent_edge_df = load_all_patent_edge_boost(metrics_dir)

    if not edge_df.empty:
        strength_df = compute_edge_node_features(edge_df, patent_edge_df)
        strength_cols = [c for c in ["in_strength", "out_strength",
                                    "in_raw_strength", "out_raw_strength"]
                        if c in strength_df.columns]
        if strength_cols:
            node_df = node_df.merge(
                strength_df[["cluster", "year"] + strength_cols],
                on=["cluster", "year"], how="left"
            )
            node_df[strength_cols] = node_df[strength_cols].fillna(0)

    print(f"  {node_df['cluster'].nunique():,} clusters  "
          f"{node_df['year'].nunique()} years  "
          f"target={target}  horizon={horizon}")
    print(f"  edge rows={len(edge_df):,}  "
          f"patent_boost rows={len(patent_df):,}  "
          f"patent_edge rows={len(patent_edge_df):,}")

    print("\nBuilding feature matrix …")
    feat_df = build_feature_matrix(
        node_df, edge_df, patent_df, patent_edge_df,
        target_col=target, horizon=horizon,
    )
    train_feat_df = feat_df.dropna(
        subset=["target"]
    ).copy()
    feat_df.to_parquet(out / "feature_matrix.parquet", index=False)
    print(f"  {len(feat_df):,} rows  {len(get_feature_cols(feat_df))} features")

    print("\nTraining …")
    model, scaler, feature_cols, cv_df = train_forecaster(train_feat_df)
    cv_df.to_parquet(out / "cv_results.parquet", index=False)

    # feature importance
    imp = feature_importance_df(model, feature_cols)
    imp.to_parquet(out / "feature_importance.parquet", index=False)
    print("\nTop 20 features:")
    print(imp.head(20).to_string(index=False))

    # future predictions
    last_year = node_df["year"].max()
    print(f"\nForecasting year {last_year + horizon} …")
    predictions = predict_future(
        model, scaler, feature_cols, feat_df, horizon, last_year,target)
    predictions.to_parquet(
        out / f"predictions_horizon{horizon}.parquet", index=False)

    tier1, tier2 = detect_emerging_technologies(
        predictions, feat_df, patent_df, top_n=30, growth_metric=target)
    
    label_map = build_cluster_label_map(concepts_dir, cutoff_year=last_year)
    tier1 = add_cluster_labels(tier1, label_map)
    tier2 = add_cluster_labels(tier2, label_map)

    tier1.to_parquet(out / "emerging_rising_fields.parquet",   index=False)
    tier2.to_parquet(out / "emerging_patent_hotspots.parquet", index=False)

    print("\nTier 1 — Rising fields (will be large, already growing):")
    print(tier1.head(10).to_string(index=False))
    print("\nTier 2 — Patent hotspots (industry leading academia):")
    print(tier2.head(10).to_string(index=False))

    max_year = node_df["year"].max()
    print("\nChecking if rising stars actually grew …")


    validation = validate_held_out(
        train_metrics_dir = metrics_dir,
        val_metrics_dir   = val_metrics_dir,
        predictions       = predictions,
        tier1             = tier1,
        tier2             = tier2,
        out_dir           = out_dir,
        target            = target,
        horizon           = horizon,
    )

    print(f"\nForecast complete — outputs in {out_dir}")
    return predictions, cv_df, imp


# ENTRY POINT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", required=True,
                        help="Directory with node/edge/patent parquet files")
    parser.add_argument("--val_metrics_dir", required=True,
                        help="Directory with V5 metrics for validation")
    parser.add_argument("--out",     default="forecast_output")
    parser.add_argument("--target",  default="pagerank",
                        choices=["pagerank", "in_degree", "in_strength", "in_raw_strength"])
    parser.add_argument("--horizon", type=int, default=3,
                        help="Years ahead to forecast")
    parser.add_argument("--concepts_dir", default="concept_provenance",
                        help="Directory with stable clustered concepts CSV")
    args = parser.parse_args()

    run_forecast_pipeline(
        metrics_dir=args.metrics_dir,
        out_dir=args.out,
        val_metrics_dir=args.val_metrics_dir,
        target=args.target,
        horizon=args.horizon,
        concepts_dir=args.concepts_dir
    )


if __name__ == "__main__":
    main()
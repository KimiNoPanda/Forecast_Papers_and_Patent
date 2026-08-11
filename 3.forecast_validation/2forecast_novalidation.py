"""
2forecast_novalidation.py

Forecast future cluster importance using LightGBM regression on node + edge features.

"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

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
        * (1+0.25*np.log1p(tier2["patent_boost_recent"].clip(lower=0)))
        * (1+ 0.25* np.log1p(tier2["boost_vel_recent"].fillna(0).clip(lower=0)))
        * (1+ 0.10* np.log1p(tier2["patent_edge_vel"].fillna(0).clip(lower=0)))
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


#10. Full Pipeline

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


    print(f"\nForecast complete — outputs in {out_dir}")
    return predictions, cv_df, imp


# ENTRY POINT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", required=True,
                        help="Directory with node/edge/patent parquet files")
    parser.add_argument("--out",     default="forecast_output")
    parser.add_argument("--target",  default="pagerank",
                        choices=["pagerank", "in_degree", "in_strength"])
    parser.add_argument("--horizon", type=int, default=3,
                        help="Years ahead to forecast")
    parser.add_argument("--concepts_dir", default="concept_provenance",
                        help="Directory with stable clustered concepts CSV")
    args = parser.parse_args()

    run_forecast_pipeline(
        metrics_dir=args.metrics_dir,
        out_dir=args.out,
        target=args.target,
        horizon=args.horizon,
        concepts_dir=args.concepts_dir
    )


if __name__ == "__main__":
    main()
import shutil
import subprocess
import re
from pathlib import Path

import pandas as pd


ROOT = Path(".")

results = []


# Convert string metric to float

def to_float(value):
    if value is None:
        return None

    if value.lower() == "nan":
        return None

    return float(value)



# Parse forecast output

def parse_metrics(text):
    tier = None
    last_k = None
    percentile_k = None

    metrics = {
        "global": {},
        "tier1": {},
        "tier2": {},
    }

    for line in text.splitlines():
        line = line.strip()

        # Global CV metrics
        if "CV summary" in line:
            mae_match = re.search(
                r"mae=([-+]?\d*\.?\d+)",
                line,
                re.IGNORECASE,
            )

            rmse_match = re.search(
                r"rmse=([-+]?\d*\.?\d+)",
                line,
                re.IGNORECASE,
            )

            if mae_match:
                metrics["global"]["mae"] = float(mae_match.group(1))

            if rmse_match:
                metrics["global"]["rmse"] = float(rmse_match.group(1))

            continue


        # Detect Tier 1 / Tier 2 sections
        if (
            "Tier 1 rising fields" in line
            or "[Tier 1] Internal ranking quality" in line
        ):
            tier = "tier1"
            last_k = None
            percentile_k = None
            continue

        if (
            "Tier 2 patent hotspots" in line
            or "[Tier 2] Internal ranking quality" in line
        ):
            tier = "tier2"
            last_k = None
            percentile_k = None
            continue

        if tier is None:
            continue


        # Global percentile metrics

        if "Mean Global Percentile" in line:
            match = re.search(
                r"Top\s+(\d+)\).*?:\s*([-+]?\d*\.?\d+)",
                line,
                re.IGNORECASE,
            )

            if match:
                percentile_k = int(match.group(1))
                metrics[tier][f"mean_at_{percentile_k}"] = float(
                    match.group(2)
                )

            continue

        if "Median Global Percentile" in line:
            values = re.findall(r"[-+]?\d*\.?\d+", line)

            if values and percentile_k is not None:
                metrics[tier][f"median_at_{percentile_k}"] = float(
                    values[-1]
                )

            continue

        if "Min Global Percentile" in line:
            values = re.findall(r"[-+]?\d*\.?\d+", line)

            if values and percentile_k is not None:
                metrics[tier][f"min_at_{percentile_k}"] = float(
                    values[-1]
                )

            continue

        if "% Clusters above 75th Percentile" in line:
            values = re.findall(r"[-+]?\d*\.?\d+", line)

            if values and percentile_k is not None:
                metrics[tier][f"above75_at_{percentile_k}"] = float(
                    values[-1]
                )

            continue


        # Spearman correlations

        if "Spearman(score, actual growth)" in line:
            match = re.search(
                r":\s*([-+]?\d*\.?\d+|nan)",
                line,
                re.IGNORECASE,
            )

            if match:
                metrics[tier]["spearman_growth"] = to_float(
                    match.group(1)
                )

            continue

        if "Spearman(score, actual percentile)" in line:
            match = re.search(
                r":\s*([-+]?\d*\.?\d+|nan)",
                line,
                re.IGNORECASE,
            )

            if match:
                metrics[tier]["spearman_percentile"] = to_float(
                    match.group(1)
                )

            continue


        # Ranking metrics

        ranking_match = re.search(
            r"@(\d+):\s*"
            r"NDCG=([-+]?\d*\.?\d+|nan)\s+"
            r"mean growth=([-+]?\d*\.?\d+|nan)\s+"
            r"median growth=([-+]?\d*\.?\d+|nan)\s+"
            r"lift=([-+]?\d*\.?\d+|nan)x",
            line,
            re.IGNORECASE,
        )

        if ranking_match:
            last_k = int(ranking_match.group(1))

            metrics[tier][f"ndcg_at_{last_k}"] = to_float(
                ranking_match.group(2)
            )

            metrics[tier][f"mean_growth_at_{last_k}"] = to_float(
                ranking_match.group(3)
            )

            metrics[tier][f"median_growth_at_{last_k}"] = to_float(
                ranking_match.group(4)
            )

            metrics[tier][f"lift_at_{last_k}"] = to_float(
                ranking_match.group(5)
            )

            continue


        # Ranking continuation line

        percentile_match = re.search(
            r"mean percentile=([-+]?\d*\.?\d+|nan)%\s+"
            r"precision>P75=([-+]?\d*\.?\d+|nan)%",
            line,
            re.IGNORECASE,
        )

        if percentile_match and last_k is not None:
            metrics[tier][f"ranking_mean_percentile_at_{last_k}"] = (
                to_float(percentile_match.group(1))
            )

            metrics[tier][f"precision_p75_at_{last_k}"] = (
                to_float(percentile_match.group(2))
            )

    return metrics



# Iterate over all alpha directories

for alpha_dir in sorted(ROOT.glob("alpha_*")):
    if not alpha_dir.is_dir():
        continue

    print("=" * 80)
    print(alpha_dir.name)

    src = alpha_dir / "2025"

    if not src.exists():
        print("Missing 2025 directory")
        continue

    train_dir = alpha_dir / "2022"
    val_dir = alpha_dir / "2025"


    out_dir = alpha_dir / "forecast_out"

    cmd = [
        "python3",
        "7forecast.py",
        "--metrics_dir", str(train_dir),
        "--val_metrics_dir", str(val_dir),
        "--out", str(out_dir),
        "--target", "in_strength",
        "--horizon", "3",
        "--concepts_dir","concept_provenance/"
    ]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    print(proc.stdout)

    if proc.returncode != 0:
        print(proc.stderr)
        continue

    m = parse_metrics(proc.stdout)

    row = {
        "alpha": alpha_dir.name,

        # Global model metrics
        "mae": m["global"].get("mae"),
        "rmse": m["global"].get("rmse"),

        # Tier 1 percentile metrics
        "tier1_mean_at_10": m["tier1"].get("mean_at_10"),
        "tier1_median_at_10": m["tier1"].get("median_at_10"),
        "tier1_min_at_10": m["tier1"].get("min_at_10"),
        "tier1_above75_at_10": m["tier1"].get("above75_at_10"),

        "tier1_mean_at_25": m["tier1"].get("mean_at_25"),
        "tier1_median_at_25": m["tier1"].get("median_at_25"),
        "tier1_min_at_25": m["tier1"].get("min_at_25"),
        "tier1_above75_at_25": m["tier1"].get("above75_at_25"),

        # Tier 2 percentile metrics
        "tier2_mean_at_10": m["tier2"].get("mean_at_10"),
        "tier2_median_at_10": m["tier2"].get("median_at_10"),
        "tier2_min_at_10": m["tier2"].get("min_at_10"),
        "tier2_above75_at_10": m["tier2"].get("above75_at_10"),

        "tier2_mean_at_25": m["tier2"].get("mean_at_25"),
        "tier2_median_at_25": m["tier2"].get("median_at_25"),
        "tier2_min_at_25": m["tier2"].get("min_at_25"),
        "tier2_above75_at_25": m["tier2"].get("above75_at_25"),

        # Spearman
        "tier1_spearman_growth": (
            m["tier1"].get("spearman_growth")
        ),
        "tier1_spearman_percentile": (
            m["tier1"].get("spearman_percentile")
        ),
        "tier2_spearman_growth": (
            m["tier2"].get("spearman_growth")
        ),
        "tier2_spearman_percentile": (
            m["tier2"].get("spearman_percentile")
        ),

        # Tier 1 ranking metrics
        "tier1_ndcg_at_10": m["tier1"].get("ndcg_at_10"),
        "tier1_ndcg_at_25": m["tier1"].get("ndcg_at_25"),


        "tier1_lift_at_10": m["tier1"].get("lift_at_10"),
        "tier1_lift_at_25": m["tier1"].get("lift_at_25"),


        "tier1_precision_p75_at_10": (
            m["tier1"].get("precision_p75_at_10")
        ),
        "tier1_precision_p75_at_25": (
            m["tier1"].get("precision_p75_at_25")
        ),


        # Tier 2 ranking metrics
        "tier2_ndcg_at_10": m["tier2"].get("ndcg_at_10"),
        "tier2_ndcg_at_25": m["tier2"].get("ndcg_at_25"),

        "tier2_lift_at_10": m["tier2"].get("lift_at_10"),
        "tier2_lift_at_25": m["tier2"].get("lift_at_25"),


        "tier2_precision_p75_at_10": (
            m["tier2"].get("precision_p75_at_10")
        ),
        "tier2_precision_p75_at_25": (
            m["tier2"].get("precision_p75_at_25")
        ),

    }

    results.append(row)



# Ranking

df = pd.DataFrame(results)

if df.empty:
    print("No successful forecast results")
    raise SystemExit(1)


# Fill only for score calculation.
# Keep the original columns unchanged in the CSV.
def value(column, default=0):
    if column not in df.columns:
        return default

    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).fillna(default)


# Simple ranking criterion:
#
# - Higher percentile is better
# - Higher precision is better
# - Higher Spearman is better
# - Higher NDCG is better
# - Lower MAE/RMSE is better
#
# The percentile values are printed from 0 to 100.
# Spearman and NDCG usually range from -1/0 to 1.
df["score"] = (
    # Global percentile quality
    value("tier1_mean_at_10")
    + value("tier1_mean_at_25")
    + value("tier2_mean_at_10")
    + value("tier2_mean_at_25")

    # Precision above P75
    + value("tier1_above75_at_10")
    + value("tier1_above75_at_25")
    + value("tier2_above75_at_10")
    + value("tier2_above75_at_25")

    # Spearman: multiply by 100 to use a similar scale
    + 100 * value("tier1_spearman_growth")
    + 100 * value("tier1_spearman_percentile")
    + 100 * value("tier2_spearman_growth")
    + 100 * value("tier2_spearman_percentile")

    # NDCG
    + 100 * value("tier1_ndcg_at_10")
    + 100 * value("tier1_ndcg_at_25")
    + 100 * value("tier2_ndcg_at_10")
    + 100 * value("tier2_ndcg_at_25")

    # Lower errors are better
    - 10 * value("mae")
    - 10 * value("rmse")
)

df = df.sort_values(
    "score",
    ascending=False,
)

df.to_csv(
    "alpha_comparison.csv",
    index=False,
)

print("\nBest alpha combinations:\n")
print(df.to_string(index=False))
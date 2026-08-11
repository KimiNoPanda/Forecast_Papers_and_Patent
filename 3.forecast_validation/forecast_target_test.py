'''
forecast_target_test.py

Run a sweep of forecast experiments, varying the training cutoff year,
forecast horizon, and target metric (in_degree vs in_strength).
'''

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

import pandas as pd


# 1. FILE SLICING


def discover_available_years(base_dir: Path) -> list[int]:
    """Find every year present via node_metrics_{year}.parquet filenames."""
    years = []
    for f in base_dir.glob("node_metrics_*.parquet"):
        try:
            years.append(int(f.stem.split("_")[-1]))
        except ValueError:
            continue
    return sorted(years)


def copy_years(base_dir: Path, dst_dir: Path, years: list[int],
                clean: bool = True) -> int:
    """
    Copy every per-year parquet file (node_metrics_{y}, edge_metrics_{y},
    patent_boost_{y}, patent_edge_boost_{y}, yearly_edges_{y}, ...) for the
    given years from base_dir into dst_dir. Returns number of files copied.
    """
    if clean and dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for year in years:
        for f in base_dir.glob(f"*_{year}.parquet"):
            shutil.copy2(f, dst_dir / f.name)
            n += 1
    return n



# 2. RUN A SINGLE FORECAST


def run_forecast(forecast_script: str,
                  metrics_dir: Path,
                  val_metrics_dir: Path,
                  out_dir: Path,
                  target: str,
                  horizon: int,
                  cwd: Path) -> bool:
    """Invoke 7forecast.py as a subprocess. Returns True on success."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, forecast_script,
        f"--metrics_dir={metrics_dir}/",
        f"--val_metrics_dir={val_metrics_dir}/",
        f"--out={out_dir}/",
        f"--target={target}",
        f"--horizon={horizon}",
    ]
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)

    log_path = out_dir / "run_log.txt"
    log_path.write_text(result.stdout + "\n\n[STDERR]\n" + result.stderr)

    if result.returncode != 0:
        print(f"  [FAILED] see {log_path}")
        print(result.stderr[-2000:])
        return False

    print(f"  [ok]  log -> {log_path}")
    return True



# 3. AGGREGATE VALIDATION RESULTS


def summarize_tier(parquet_path: Path, top_k: int = 10) -> dict | None:
    """
    Load validation parquet and compute both ranking metrics and
    global-percentile metrics.
    """
    if not parquet_path.exists():
        return None

    df = pd.read_parquet(parquet_path)

    required = {"score", "actual_growth", "global_percentile"}
    if df.empty or not required.issubset(df.columns):
        return None

    # ---------- Ranking quality ----------
    rho, _ = spearmanr(
        df["score"],
        df["actual_growth"],
        nan_policy="omit",
    )

    y_true = np.asarray([df["actual_growth"].to_numpy()])
    y_score = np.asarray([df["score"].to_numpy()])

    ndcg10 = ndcg_score(y_true, y_score, k=10)
    ndcg25 = ndcg_score(y_true, y_score, k=25)
    ndcg50 = ndcg_score(y_true, y_score, k=50)

    # ---------- Percentile summary ----------
    top = df.nlargest(min(top_k, len(df)), "score")

    return {
        "n_scored": len(df),
        "n_top_k": len(top),

        "mean_global_pctile": top["global_percentile"].mean(),
        "median_global_pctile": top["global_percentile"].median(),
        "min_global_pctile": top["global_percentile"].min(),
        "pct_above_75th": (top["global_percentile"] >= 75).mean(),

        "spearman": rho,
        "ndcg10": ndcg10,
        "ndcg25": ndcg25,
        "ndcg50": ndcg50,
    }


def collect_results(out_dir: Path, train_year: int, horizon: int,
                     target: str) -> list[dict]:
    rows = []
    for tier_num, fname in [
        (1, "validation_tier1_rising_fields.parquet"),
        (2, "validation_tier2_patent_hotspots.parquet"),
    ]:
        stats = summarize_tier(out_dir / fname)
        if stats is None:
            continue
        rows.append({
            "train_year": train_year,
            "horizon":    horizon,
            "val_year":   train_year + horizon,
            "target":     target,
            "tier":       tier_num,
            **stats,
        })
    return rows



# 4. FULL SWEEP


def run_sweep(base_dir: str,
              out_root: str,
              forecast_script: str,
              train_years: list[int],
              horizons: list[int],
              targets: list[str],
              cwd: str | None = None):
    base_dir = Path(base_dir).resolve()
    out_root = Path(out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    cwd = Path(cwd).resolve() if cwd else Path(forecast_script).resolve().parent

    available_years = discover_available_years(base_dir)
    if not available_years:
        raise FileNotFoundError(
            f"No node_metrics_*.parquet found under {base_dir}")
    year_min, year_max = available_years[0], available_years[-1]
    print(f"Available years in {base_dir}: {year_min}-{year_max} "
          f"({len(available_years)} years)")

    all_rows = []

    for train_year in train_years:
        for horizon in horizons:
            val_year = train_year + horizon

            if val_year > year_max:
                print(f"\n[skip] train={train_year} horizon={horizon} -> "
                      f"val_year={val_year} exceeds max available "
                      f"year {year_max}")
                continue
            if train_year < year_min + 5:
                # forecaster requires >=5 years history per cluster
                print(f"\n[skip] train={train_year} horizon={horizon} -> "
                      f"fewer than 5 years of history before cutoff")
                continue

            train_dir = out_root / f"_slice_train_{train_year}"
            val_dir   = out_root / f"_slice_val_{train_year}_h{horizon}"

            train_years_needed = [y for y in available_years if y <= train_year]
            val_years_needed   = [y for y in [train_year, val_year]
                                   if y in available_years]

            n_train = copy_years(base_dir, train_dir, train_years_needed)
            n_val   = copy_years(base_dir, val_dir, val_years_needed)
            print(f"\n=== train_year={train_year}  horizon={horizon}  "
                  f"val_year={val_year} ===")
            print(f"  train slice: {n_train} files ({len(train_years_needed)} years) "
                  f"-> {train_dir}")
            print(f"  val slice:   {n_val} files ({len(val_years_needed)} years) "
                  f"-> {val_dir}")

            for target in targets:
                run_out = out_root / f"train{train_year}_h{horizon}_{target}"
                ok = run_forecast(
                    forecast_script=forecast_script,
                    metrics_dir=train_dir,
                    val_metrics_dir=val_dir,
                    out_dir=run_out,
                    target=target,
                    horizon=horizon,
                    cwd=cwd,
                )
                if not ok:
                    continue
                all_rows.extend(
                    collect_results(run_out, train_year, horizon, target))

            # free disk — comment out if you want to keep the raw slices
            shutil.rmtree(train_dir, ignore_errors=True)
            shutil.rmtree(val_dir, ignore_errors=True)

    if not all_rows:
        print("\nNo successful runs produced validation output.")
        return pd.DataFrame()

    results = pd.DataFrame(all_rows)
    results_path = out_root / "sweep_comparison.csv"
    results.to_csv(results_path, index=False)
    print(f"\n✓ Sweep complete. Comparison table -> {results_path}")

    print_comparison(results)
    return results



# 5. COMPARISON REPORT


def print_comparison(results: pd.DataFrame):
    if results.empty:
        return

    print("\n" + "=" * 78)
    print("PER-RUN RESULTS")
    print("=" * 78)
    cols = [
        "train_year",
        "horizon",
        "val_year",
        "target",
        "tier",
        "spearman",
        "ndcg10",
        "ndcg25",
        "mean_global_pctile",
        "median_global_pctile",
        "min_global_pctile",
        "pct_above_75th",
    ]
    print(results[cols].sort_values(
        ["tier", "train_year", "horizon", "target"]).to_string(index=False))

    print("\n" + "=" * 78)
    print("in_degree vs in_strength — averaged across all (train_year, horizon)")
    print("=" * 78)
    summary = (
        results.groupby(["tier", "target"])
        .agg(
            n_runs=("mean_global_pctile", "count"),

            avg_spearman=("spearman", "mean"),
            avg_ndcg10=("ndcg10", "mean"),
            avg_ndcg25=("ndcg25", "mean"),
            avg_ndcg50=("ndcg50", "mean"),

            avg_mean_pctile=("mean_global_pctile", "mean"),
            avg_median_pctile=("median_global_pctile", "mean"),
            avg_min_pctile=("min_global_pctile", "mean"),
            avg_pct_above_75th=("pct_above_75th", "mean"),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))

    print("\n" + "=" * 78)
    print("WIN/LOSS — how often did in_strength beat in_degree "
          "(by mean_global_pctile), per (train_year, horizon, tier)")
    print("=" * 78)
    pivot = results.pivot_table(
        index=["tier", "train_year", "horizon"],
        columns="target", values="mean_global_pctile"
    ).reset_index()
    if "in_degree" in pivot.columns and "in_strength" in pivot.columns:
        pivot = pivot.dropna(subset=["in_degree", "in_strength"])
        pivot["in_strength_wins"] = pivot["in_strength"] > pivot["in_degree"]
        print(pivot.to_string(index=False))
        win_rate = pivot["in_strength_wins"].mean()
        print(f"\nin_strength beat in_degree in "
              f"{pivot['in_strength_wins'].sum()}/{len(pivot)} "
              f"windows ({win_rate:.1%})")
    else:
        print("  (need both targets present to compute win/loss)")



# ENTRY POINT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics_dir", default="alpha_03_05/2026",
                   help="Directory with cumulative node/edge/patent "
                        "parquet files covering the full year range "
                        "(e.g. alpha_03_05/2026, years 2000-2026)")
    p.add_argument("--out_root", default="alpha_03_05/experiments",
                   help="Where per-window train/val slices and forecast "
                        "outputs get written")
    p.add_argument("--forecast_script", default="7forecast.py",
                   help="Path to your forecaster_v2.py entrypoint")
    p.add_argument("--train_years", type=int, nargs="+",
                   default=[2016, 2018, 2020, 2022],
                   help="Cutoff years to train through, one run per year")
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5],
                   help="Forecast horizons (years ahead) to test")
    p.add_argument("--targets", nargs="+",
                   default=["in_degree", "in_strength"],
                   help="Targets to compare")
    p.add_argument("--cwd", default=None,
                   help="Working directory to run 7forecast.py from "
                        "(defaults to the forecast script's own directory, "
                        "so relative paths like concepts_path resolve the "
                        "same way they do when you run it manually)")
    args = p.parse_args()

    run_sweep(
        base_dir=args.metrics_dir,
        out_root=args.out_root,
        forecast_script=args.forecast_script,
        train_years=args.train_years,
        horizons=args.horizons,
        targets=args.targets,
        cwd=args.cwd,
    )


if __name__ == "__main__":
    main()
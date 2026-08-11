""""""
import argparse
import glob
import os
import re

import pandas as pd
import matplotlib.pyplot as plt


def find_year_files(data_dir: str) -> dict[int, str]:
    """Map year -> filepath by parsing the filename."""
    pattern = os.path.join(data_dir, "papers_referenced_works_*.parquet") # adjust to your own path
    files = glob.glob(pattern)
    year_files = {}
    for f in files:
        m = re.search(r"(\d{4})", os.path.basename(f))
        if m:
            year_files[int(m.group(1))] = f
    return dict(sorted(year_files.items()))


def build_known_ids(year_files: dict[int, str]) -> set[str]:
    """collect every paper id present anywhere in the dataset,
    regardless of year, so we can test 'in dataset' membership for any
    referenced_work id."""
    known_ids: set[str] = set()
    for year, path in year_files.items():
        df = pd.read_parquet(path, columns=["id"])
        known_ids.update(df["id"].tolist())
    return known_ids


def count_in_out_per_year(year_files: dict[int, str], known_ids: set[str]) -> pd.DataFrame:
    """for each year's papers, count referenced_works that are
    in `known_ids` (in-dataset) vs not (out-of-dataset)."""
    records = []
    for year, path in year_files.items():
        df = pd.read_parquet(path)

        in_count = 0
        out_count = 0
        n_papers = len(df)

        for refs in df["referenced_works"]:
            if refs is None:
                continue
            # refs may be a numpy array or list depending on parquet reader
            for rid in refs:
                if rid in known_ids:
                    in_count += 1
                else:
                    out_count += 1

        total = in_count + out_count
        records.append({
            "year": year,
            "n_papers": n_papers,
            "refs_in_dataset": in_count,
            "refs_out_of_dataset": out_count,
            "total_refs": total,
            "pct_in_dataset": (in_count / total * 100) if total else 0.0,
        })

    return pd.DataFrame(records).sort_values("year")


def plot_counts(summary: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(14, 7))

    years = summary["year"]
    ax.bar(years, summary["refs_in_dataset"], label="Referenced work IS in dataset", color="#4C72B0")
    ax.bar(years, summary["refs_out_of_dataset"], bottom=summary["refs_in_dataset"],
           label="Referenced work NOT in dataset", color="#DD8452")

    ax.set_xlabel("Publication year of citing paper")
    ax.set_ylabel("Number of references")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved chart to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=".", help="Directory containing the parquet files")
    parser.add_argument("--out", default="references_in_vs_out.png", help="Output chart path")
    parser.add_argument("--csv", default="references_in_vs_out_summary.csv", help="Output summary CSV path")
    args = parser.parse_args()

    year_files = find_year_files(args.dir)
    if not year_files:
        raise SystemExit(f"No files matching papers_referenced_works_YYYY.parquet found in {args.dir}")

    print(f"Found {len(year_files)} year files: {list(year_files.keys())}")

    print("Building set of known ids across all years (first pass)...")
    known_ids = build_known_ids(year_files)
    print(f"Total distinct papers in dataset: {len(known_ids):,}")

    print("Counting in-dataset vs out-of-dataset references per year (second pass)...")
    summary = count_in_out_per_year(year_files, known_ids)
    print(summary.to_string(index=False))

    summary.to_csv(args.csv, index=False)
    print(f"Saved summary table to {args.csv}")

    plot_counts(summary, args.out)


if __name__ == "__main__":
    main()
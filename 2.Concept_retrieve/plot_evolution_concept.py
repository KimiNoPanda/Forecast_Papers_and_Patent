import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

FILENAME_RE = re.compile(r"stable_clustered_concepts_provenance_(\d{4})_mergedV3\.csv$") # adjust to your own path


def find_yearly_files(directory: Path) -> dict[int, Path]:
    """Return {year: path} for every matching file in the directory."""
    files = {}
    for p in directory.glob("stable_clustered_concepts_provenance_*_mergedV3.csv"): # adjust to your own path
        m = FILENAME_RE.match(p.name)
        if m:
            files[int(m.group(1))] = p
    if not files:
        raise FileNotFoundError(
            f"No files matching 'stable_clustered_concepts_provenance_<YEAR>_mergedV3.csv' "
            f"found in {directory}"
        )
    return dict(sorted(files.items()))


def load_year(path: Path) -> pd.DataFrame:
    """Load one year's CSV."""
    df = pd.read_csv(path)

    required = {"norm_concept", "paper_only", "patent_only", "shared"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing expected columns: {missing}")

    cluster_col = "stable_cluster" if "stable_cluster" in df.columns else "cluster"
    if cluster_col not in df.columns:
        raise ValueError(f"{path.name} has neither 'stable_cluster' nor 'cluster' column")


    return df


def build_yearly_counts(files: dict[int, Path]) -> pd.DataFrame:
    """Per-year snapshot counts of paper_only / patent_only / shared."""
    rows = []
    year_frames = {}
    for year, path in files.items():
        df = load_year(path)
        year_frames[year] = df
        rows.append({
            "year": year,
            "paper_only": int(df["paper_only"].sum()),
            "patent_only": int(df["patent_only"].sum()),
            "shared": int(df["shared"].sum()),
        })
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True), year_frames


def build_cumulative_counts(year_frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Running total of *distinct* norm_concept strings ever classified as
    paper_only / patent_only / shared, up through each year.
    """
    seen_category: dict[str, str] = {}  # norm_concept -> current category
    rows = []
    for year in sorted(year_frames):
        df = year_frames[year]
        for _, r in df.iterrows():
            concept = r["norm_concept"]
            if r["shared"]:
                seen_category[concept] = "shared"
            elif r["paper_only"]:
                seen_category[concept] = "paper_only"
            elif r["patent_only"]:
                seen_category[concept] = "patent_only"
            else:
                # Neither flag set (shouldn't normally happen) - skip
                continue

        counts = {"paper_only": 0, "patent_only": 0, "shared": 0}
        for cat in seen_category.values():
            counts[cat] += 1

        rows.append({"year": year, **counts})

    return pd.DataFrame(rows)


def plot_counts(df: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for col, color, marker in [
        ("paper_only", "#4C72B0", "o"),
        ("patent_only", "#DD8452", "s"),
        ("shared", "#55A868", "^"),
    ]:
        ax.plot(df["year"], df[col], marker=marker, color=color, label=col.replace("_", " "))

    ax.set_xlabel("Year")
    ax.set_ylabel("Number of concepts")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, help="Directory containing the yearly CSV files")
    parser.add_argument("--out", default=None, help="Output directory for PNGs (default: --dir)")
    args = parser.parse_args()

    in_dir = Path(args.dir)
    out_dir = Path(args.out) if args.out else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_yearly_files(in_dir)
    print(f"Found {len(files)} yearly files: {list(files.keys())}")

    yearly_df, year_frames = build_yearly_counts(files)
    cumulative_df = build_cumulative_counts(year_frames)

    print("\nPer-year snapshot counts:")
    print(yearly_df.to_string(index=False))
    print("\nCumulative distinct-concept counts:")
    print(cumulative_df.to_string(index=False))

    yearly_df.to_csv(out_dir / "concept_counts_per_year.csv", index=False)
    cumulative_df.to_csv(out_dir / "concept_counts_cumulative.csv", index=False)

    plot_counts(
        yearly_df,
        "Concept counts per year",
        out_dir / "concept_evolution_per_year.png",
    )
    plot_counts(
        cumulative_df,
        "Cumulative distinct concepts by year",
        out_dir / "concept_evolution_cumulative.png",
    )


if __name__ == "__main__":
    sys.exit(main())
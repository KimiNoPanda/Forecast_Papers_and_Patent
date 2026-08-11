
import pandas as pd
import os
import glob


DATA_DIR = "concept_provenance/" # adjust to your own path

results = []


files = sorted(
    glob.glob(
        os.path.join(
            DATA_DIR,
            "stable_clustered_concepts_provenance_*_mergedV3.csv" # adjust to your own path
        )
    )
)


previous_clusters = set()


for file in files:

    filename = os.path.basename(file)
    year = int(filename.split("_")[4])

    print(f"Processing {year}")

    df = pd.read_csv(file)


    if "stable_cluster" not in df.columns:
        raise ValueError(
            f"Missing stable_cluster column in {file}"
        )


    total_concepts = len(df)


    # Remove unstable / unmatched concepts
    stable_df = df[df["stable_cluster"] != -1]


    assigned_concepts = len(stable_df)


    coverage_ratio = (
        assigned_concepts / total_concepts * 100
        if total_concepts > 0
        else 0
    )


    # Number of stable trajectories
    n_stable_clusters = (
        stable_df["stable_cluster"]
        .nunique()
    )


    # Stable cluster sizes
    cluster_sizes = (
        stable_df
        .groupby("stable_cluster")
        .size()
    )


    avg_cluster_size = (
        cluster_sizes.mean()
        if len(cluster_sizes) > 0
        else 0
    )


    largest_cluster_size = (
        cluster_sizes.max()
        if len(cluster_sizes) > 0
        else 0
    )


    current_clusters = set(
        stable_df["stable_cluster"].unique()
    )


    # New stable clusters compared to previous year
    new_clusters = (
        current_clusters - previous_clusters
        if previous_clusters
        else set()
    )


    disappeared_clusters = (
        previous_clusters - current_clusters
        if previous_clusters
        else set()
    )


    results.append({

        "year": year,

        "total_concepts": total_concepts,

        "stable_clusters": n_stable_clusters,

        "assigned_concepts": assigned_concepts,

        "coverage_percent": round(
            coverage_ratio, 2
        ),

        "average_cluster_size": round(
            avg_cluster_size, 2
        ),

        "largest_cluster_size": largest_cluster_size,

        "new_stable_clusters": len(new_clusters),

        "disappeared_stable_clusters": len(
            disappeared_clusters
        )
    })


    previous_clusters = current_clusters



summary = pd.DataFrame(results)

summary = summary.sort_values("year")


summary.to_csv(
    "plot/stable_cluster_statistics_by_year.csv",
    index=False
)


print(summary)
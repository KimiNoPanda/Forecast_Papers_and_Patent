import pandas as pd
import os
import glob


DATA_DIR = "concept_provenance/" # adjust to your own path

results = []


# Find all yearly files
files = sorted(
    glob.glob(
        os.path.join(
            DATA_DIR,
            "stable_clustered_concepts_provenance_*_mergedV3.csv" # adjust to your own path
        )
    )
)


for file in files:

    # Extract year from filename
    filename = os.path.basename(file)
    year = filename.split("_")[4]

    print(f"Processing {year}")

    df = pd.read_csv(file)

    # Check required column
    if "cluster" not in df.columns:
        raise ValueError(f"Missing cluster column in {file}")


    total_concepts = len(df)


    # Noise points in HDBSCAN are labelled -1
    noise_count = (df["cluster"] == -1).sum()

    noise_ratio = (
        noise_count / total_concepts * 100
        if total_concepts > 0
        else 0
    )


    # Remove noise for cluster statistics
    clustered = df[df["cluster"] != -1]


    # Number of clusters
    n_clusters = clustered["cluster"].nunique()


    # Cluster sizes
    cluster_sizes = (
        clustered
        .groupby("cluster")
        .size()
    )


    avg_cluster_size = (
        cluster_sizes.mean()
        if len(cluster_sizes) > 0
        else 0
    )


    max_cluster_size = (
        cluster_sizes.max()
        if len(cluster_sizes) > 0
        else 0
    )


    results.append({
        "year": int(year),
        "total_concepts": total_concepts,
        "number_of_clusters": n_clusters,
        "noise_concepts": noise_count,
        "noise_ratio_percent": round(noise_ratio, 2),
        "average_cluster_size": round(avg_cluster_size, 2),
        "largest_cluster_size": max_cluster_size
    })


# Create dataframe
summary = pd.DataFrame(results)


# Sort by year
summary = summary.sort_values("year")


# Save result
summary.to_csv(
    "plot/hdbscan_cluster_statistics_by_year.csv",
    index=False
)


print("\nSaved:")
print("plot/hdbscan_cluster_statistics_by_year.csv")

print(summary)
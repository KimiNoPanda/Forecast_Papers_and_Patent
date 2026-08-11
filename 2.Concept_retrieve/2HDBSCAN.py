'''
Code was run on Kaggle with GPU T4 accelerator enabled. We use HDBSCAN to cluster concepts extracted from papers and patents.

You might to adapt it to run with your own gpu and datasets. I pointed out the paths to the datasets and output directories, but you may need to adjust them based on your environment.

'''


# %% [code] {"execution":{"iopub.status.busy":"2026-06-16T12:56:19.178948Z","iopub.execute_input":"2026-06-16T12:56:19.179565Z","iopub.status.idle":"2026-06-16T12:57:13.097682Z","shell.execute_reply.started":"2026-06-16T12:56:19.179533Z","shell.execute_reply":"2026-06-16T12:57:13.096780Z"},"jupyter":{"outputs_hidden":false}}
import numpy as np
import pandas as pd
import joblib
import umap
import hdbscan
from pathlib import Path
from sklearn.decomposition import IncrementalPCA
from sentence_transformers import SentenceTransformer

# %% [code] {"execution":{"iopub.status.busy":"2026-06-16T13:08:45.553951Z","iopub.execute_input":"2026-06-16T13:08:45.554471Z","iopub.status.idle":"2026-06-16T13:08:46.193672Z","shell.execute_reply.started":"2026-06-16T13:08:45.554436Z","shell.execute_reply":"2026-06-16T13:08:46.192844Z"},"jupyter":{"outputs_hidden":false}}
!rm -rf /kaggle/working/*/*.pkl
!rm -rf /kaggle/working/*/*.npy
!rm -rf /kaggle/working/*/paperless*

# %% [code] {"execution":{"iopub.status.busy":"2026-06-16T13:09:10.841991Z","iopub.execute_input":"2026-06-16T13:09:10.842723Z","iopub.status.idle":"2026-06-16T13:09:10.859235Z","shell.execute_reply.started":"2026-06-16T13:09:10.842689Z","shell.execute_reply":"2026-06-16T13:09:10.858250Z"},"jupyter":{"outputs_hidden":false}}
def load_embeddings(df, method, out_dir,
                    model_name="allenai/scibert_scivocab_uncased",
                    batch_size=64):
    cache = Path(out_dir) / f"{method}_concept_embeddings.npy"
    if cache.exists():
        print("Loading cached embeddings...")
        return np.load(cache, mmap_mode="r")

    print("Computing embeddings...")
    model = SentenceTransformer(model_name, device="cuda")
    embeddings = model.encode(
        df["norm_concept"].astype(str).tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    np.save(cache, embeddings)
    return embeddings


def summarize_cluster(group):
    group = group.sort_values("total_freq", ascending=False)
    top_concepts = group["norm_concept"].head(10).tolist()
    return pd.Series({
        "cluster":        group["cluster"].iloc[0],
        "size":           len(group),
        "total_freq_sum": group["total_freq"].sum(),
        "top_concepts":   " | ".join(top_concepts),
        "label":          top_concepts[0] if top_concepts else None,
    })
import re

ARTIFACT_PATTERNS = [
    # Patent boilerplate
    r"\bdiscloses?\b",
    r"\bdisclosure\b",
    r"\bembodiment\b",
    r"\b(?:the\s+)?present invention\b",
    r"\binvention\b",
    r"\bwherein\b",
    r"\bclaim(?:ed)?\b",
    r"\bapparatus\b",
    r"\bmethod for\b",
    r"\bsystem for\b",
    r"\bdevice for\b",
    r"\bconfigured to\b",
    r"\boperable to\b",
    r"\bcoupled to\b",
    r"\bplurality of\b",
    r"\bat least one\b",
    r"\bcompris(?:e|es|ing)\b",
    r"\bconsist(?:s|ing)? of\b",
    r"\bcomputer[- ]readable medium\b",
    r"\bnon[- ]transitory\b",
    r"technical field",
    r"field of the invention",
    r"relates.*field",
    r"directed.*field",

    # Patent figures / drawings
    r"\bfig(?:ure)?\.?\s*\d+\b",
    r"\btable\s+\d+\b",
    r"\bflow\s*chart\b",
    r"\bblock diagram\b",
    r"\bdrawing(?:s)?\b",
    r"\breference numeral\b",

    # Journal metadata
    r"^abstract available",
    r"^available content",
    r"^author letter",
    r"^lett author",
    r"^letter offer reply",
    r"^withdrawn",
    r"^citation alert",
    r"^icon view",
    r"^pct[\s_]",
    r"^sec date",
    r"\bcorresponding author\b",
    r"\bopen access\b",
    r"\bpublisher'?s note\b",
    r"\bconflict of interest\b",
    r"\bcompeting interests?\b",
    r"\bsupplementary material\b",
    r"\backnowledg(?:e)?ments?\b",

    # Citation / indexing artifacts
    r"^et al\b",
    r"\bet al\.\b",
    r"^doi\b",
    r"\bdoi:",
    r"^arxiv\b",
    r"\barxiv:",
    r"\bpmid\b",
    r"\bpmcid\b",
    r"\bissn\b",
    r"\beissn\b",
    r"\bcrossref\b",

    # Generic legal / boilerplate
    r"\bherein\b",
    r"\bhereof\b",
    r"\bthereof\b",
    r"\btherein\b",
    r"\bthereby\b",
    r"\bwhereby\b",
    r"\bpursuant to\b",
    r"\bwithout departing from\b",
    r"\bscope of the invention\b",

    # OCR / extraction junk
    r"^\[\d+\]$",
    r"^\(\d+\)$",
    r"^page\s+\d+$",
    r"^\d+$",

    # Very short low-information concepts
    r"^[a-z]$",
    r"^[a-z]{2}$",
]

def is_artifact(concept: str) -> bool:
    if not concept:
        return True

    concept = str(concept).strip().lower()

    return any(re.search(pattern, concept) for pattern in ARTIFACT_PATTERNS)

def run(year_start,year_end):
    for year in range(year_start,year_end):
    
        OUT_DIR = Path(f"{year}/")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    
        df = pd.read_csv(f"survivors_{year}.csv") # adjust to your own path

        mask     = ~df["norm_concept"].apply(is_artifact)
        removed  = (~mask).sum()
        df       = df[mask].reset_index(drop=True)
        print(f"  {year}: removed {removed:,} artifacts → {len(df):,} remain")

    
        # Embeddings
        embeddings = load_embeddings(df, "scibert", OUT_DIR)
        embeddings = np.asarray(embeddings, dtype=np.float32, copy=False)
    
        # PCA
        pca = IncrementalPCA(n_components=50, batch_size=1024)
        embeddings = pca.fit_transform(embeddings)
        joblib.dump(pca, OUT_DIR / "pca_model_provenance.pkl")
    
        # UMAP 
        umap_model = umap.UMAP(
            n_neighbors=15,
            n_components=10,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
            low_memory=True,
            n_jobs=1,
        )
        reduced = umap_model.fit_transform(embeddings)
        joblib.dump(umap_model, OUT_DIR / "umap_model_provenance.pkl")
        del embeddings, umap_model
    
        # ── HDBSCAN ───────────────────────────────────────────────

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=3,
            min_samples=2,
            metric="euclidean",
            cluster_selection_method="leaf",
            prediction_data=True,           
        )
        labels = clusterer.fit_predict(reduced)
        joblib.dump(clusterer, OUT_DIR / "hdbscan_model_provenance.pkl")
        del reduced, clusterer
    
        # ── Save results ──────────────────────────────────────────
        df["cluster"] = labels
    
        noise = df[df["cluster"] == -1].copy()
        if "total_freq" in noise.columns:
            noise = noise.sort_values("total_freq", ascending=False)
        noise.to_csv(OUT_DIR / "paperless_scibert_noise.csv", index=False)
        print(f"Noise: {len(noise):,} concepts")
    
        clusters_df = df[df["cluster"] != -1].copy()
        df.to_csv(OUT_DIR / f"clustered_concepts_provenance_{year}.csv", index=False)
    
        cluster_summary = (
            clusters_df
            .groupby("cluster", as_index=False)
            .apply(summarize_cluster)
            .reset_index(drop=True)
            .sort_values("total_freq_sum", ascending=False)
        )
        cluster_summary.to_csv(OUT_DIR / f"clusters_summary_provenance_{year}.csv", index=False)
        print(f"Clusters: {len(cluster_summary):,}  "
              f"(noise rate: {len(noise)/len(df):.1%})")


# %% [code] {"execution":{"iopub.status.busy":"2026-06-16T13:09:14.413037Z","iopub.execute_input":"2026-06-16T13:09:14.413800Z","iopub.status.idle":"2026-06-16T13:11:58.950367Z","shell.execute_reply.started":"2026-06-16T13:09:14.413768Z","shell.execute_reply":"2026-06-16T13:11:58.949527Z"},"jupyter":{"outputs_hidden":false}}
if __name__ == "__main__": 
    run(2000,2008) # Adjust the year range based on your dataset and needs

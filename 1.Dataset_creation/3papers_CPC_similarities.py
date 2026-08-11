"""
3papers_CPC_similarities.py
This script computes the similarity between paper abstracts and CPC labels using both TF-IDF and sentence embeddings.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterable, List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

def clean_text(text: str) -> str:
    """
    Cleans the input text by removing LaTeX, citations, URLs, and other unwanted characters.
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = text.replace("\n", " ").replace("-", " ")
    text = re.sub(r"\$.*?\$", " ", text)
    text = re.sub(r"\[\d+\]", " ", text)
    text = re.sub(r"\(\d{4}\)", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\b[a-z]\b", " ", text)
    text = re.sub(r"\b\w*\d\w*\b", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Load and clean abstracts
def load_papers_from_year_files(folder: Path) -> pd.DataFrame:
    """
    Loads paper data from yearly parquet files, removes duplicates, and filters for valid abstracts.
    """
    files = []
    for f in folder.glob("papers_*.parquet"):
        match = re.search(r"papers_(\d{4})\.parquet$", f.name)
        if match:
            year = int(match.group(1))
            if 2000 <= year <= 2026:
                files.append(f)

    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No files matching work_*.parquet found in {folder}")

    dfs = [pd.read_parquet(f) for f in files]

    # combine all years
    df_paper = pd.concat(dfs, ignore_index=True)

    # remove duplicate papers
    df_paper = df_paper.drop_duplicates(subset="id")

    # keep only rows with a valid abstract
    df_paper = df_paper[df_paper["abstract"].notna()]

    # remove empty abstracts
    df_paper = df_paper[df_paper["abstract"].str.strip() != ""]

    # keep useful columns
    df_paper = df_paper[["display_name", "publication_year", "abstract"]]

    return df_paper


# TF-IDF matching : Measure how similar the paper abstracts are to the CPC labels using TF-IDF and nearest neighbors. This is a more traditional approach that focuses on keyword overlap and can be more interpretable, but may miss deeper semantic connections.
def match_tfidf(
    df_papers: pd.DataFrame,
    df_cpc: pd.DataFrame,
    top_k: int = 5,
    max_features: int = 100_000,
):
    """
    Compute top-k CPC matches per paper using TF-IDF and nearest neighbors.
    """

    custom_stopwords = {
    "scale",
    "device",
    "system",
    "apparatus",
    "method",
    "means",
    }
    cpc_texts = df_cpc["full_label"].tolist()
    paper_texts = df_papers["abstract"].tolist()

    vectorizer = TfidfVectorizer(
         stop_words=list(custom_stopwords.union({"english"})),
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=max_features,
    )

    X_cpc = vectorizer.fit_transform(cpc_texts)
    X_papers = vectorizer.transform(paper_texts)

    nn = NearestNeighbors(
        n_neighbors=top_k,
        metric="cosine",
        algorithm="brute",
    )
    nn.fit(X_cpc)

    print("Finding nearest neighbors...")
    distances, indices = nn.kneighbors(X_papers, return_distance=True)
    scores = 1.0 - distances

    rows = []
    print("Compiling results...")
    for i in range(len(df_papers)):
        for rank, (cpc_idx, score) in enumerate(zip(indices[i], scores[i]), start=1):
            rows.append({
                "paper_id": df_papers.iloc[i]["display_name"],
                "year": df_papers.iloc[i]["publication_year"],
                "cpc_code": df_cpc.iloc[cpc_idx]["code"],
                "cpc_label": df_cpc.iloc[cpc_idx]["full_label"],
                "tfidf_score": float(score),
                "rank": rank,
            })

    return pd.DataFrame(rows)

# Embedding-based matching : Use a pre-trained sentence transformer model to encode both the paper abstracts and the CPC labels into dense vector representations. 
# Then, use nearest neighbor search in this embedding space to find the most semantically similar CPC codes for each paper. This approach can capture deeper semantic relationships but may be less interpretable than TF-IDF.
def match_embeddings(
    df_papers: pd.DataFrame,
    df_cpc: pd.DataFrame,
    top_k: int = 5,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 256,
) -> pd.DataFrame:
    """
    Compute top-k CPC matches per paper using sentence embeddings.
    Returns long-format dataframe:
        paper_id, year, cpc_code, cpc_label, emb_score, rank
    """
    model = SentenceTransformer(model_name)

    if Path("cpc_embeddings.npy").exists():
        print("Loading pre-computed CPC embeddings...")
        cpc_embeddings = np.load("cpc_embeddings.npy")
    else:
        cpc_embeddings = model.encode(
            df_cpc["full_label"].tolist(),
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        np.save("cpc_embeddings.npy", cpc_embeddings)

    if Path("paper_embeddings.npy").exists():
        print("Loading pre-computed paper embeddings...")
        paper_embeddings = np.load("paper_embeddings.npy")
    else:
        paper_embeddings = model.encode(
            df_papers["abstract"].tolist(),
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        np.save("paper_embeddings.npy", paper_embeddings)

    # cosine similarity on normalized embeddings = dot product
    nn = NearestNeighbors(
        n_neighbors=top_k,
        metric="cosine",
        algorithm="brute",
    )
    nn.fit(cpc_embeddings)

    distances, indices = nn.kneighbors(paper_embeddings, return_distance=True)
    scores = 1.0 - distances

    all_rows = []

    for paper_idx in range(len(df_papers)):
        paper_id = df_papers.iloc[paper_idx]["display_name"]
        year = df_papers.iloc[paper_idx]["publication_year"]

        for rank, (cpc_idx, score) in enumerate(zip(indices[paper_idx], scores[paper_idx]), start=1):
            all_rows.append({
                "paper_id": paper_id,
                "year": year,
                "cpc_code": df_cpc.iloc[cpc_idx]["code"],
                "cpc_label": df_cpc.iloc[cpc_idx]["full_label"],
                "emb_score": float(score),
                "rank": rank,
            })

    return pd.DataFrame(all_rows)

# Hybrid match : combine both TF-IDF and embedding scores
def build_hybrid_scores(
    tfidf_matches: pd.DataFrame,
    emb_matches: pd.DataFrame,
    top_k_final: int = 10,
    weight_emb: float = 0.6,
    weight_tfidf: float = 0.4,
) -> pd.DataFrame:
    """
    Merge TF-IDF and embedding matches and compute hybrid score.
    Keeps top_k_final CPC codes per paper.
    """
    merged = pd.merge(
        tfidf_matches,
        emb_matches,
        on=["paper_id", "year", "cpc_code", "cpc_label"],
        how="outer",
    )

    merged["tfidf_score"] = merged["tfidf_score"].fillna(0.0)
    merged["emb_score"] = merged["emb_score"].fillna(0.0)
    merged["hybrid_score"] = (
        weight_emb * merged["emb_score"] +
        weight_tfidf * merged["tfidf_score"]
    )

    merged = merged.sort_values(
        ["paper_id", "hybrid_score"],
        ascending=[True, False]
    ).copy()

    merged["hybrid_rank"] = merged.groupby("paper_id").cumcount() + 1
    merged = merged[merged["hybrid_rank"] <= top_k_final].copy()

    return merged.reset_index(drop=True)


def main():
    abstract_folder = Path("papers_data") 
    CPC_folder = Path("Patent_retrive")
    OUT_DIR = Path("outputs_cpc_matching")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load and clean CPC data
    df_cpc = pd.read_parquet(CPC_folder / "cpc_enriched.parquet").copy()
    df_cpc["full_label"] = df_cpc["full_label"].fillna("").map(clean_text)
    df_cpc = df_cpc[df_cpc["full_label"].str.len() > 0].copy()
    df_cpc = df_cpc[df_cpc["level_num"].isin([0, 1, 2, 3])].copy()
    df_cpc = df_cpc[df_cpc["code"].str.startswith(("G","H","Y"))].copy()
    df_cpc = df_cpc.drop_duplicates(subset=["code"]).reset_index(drop=True)

    print(f"CPC rows kept: {len(df_cpc):,}")
    
    df: pd.DataFrame = pd.read_parquet(OUT_DIR / f"hybrid_matches.parquet")

    #df: pd.DataFrame = hybrid_matches.copy()
    df = df[df["hybrid_score"] > 0.45]

    #print(df.head(20))
    #print(len(df))
    df_cpc_year = df.groupby(["cpc_code", "year"]).agg({
        "paper_id": "count",
        "hybrid_score": "mean"
    }).rename(columns={
        "paper_id": "paper_count",
        "hybrid_score": "avg_score"
    }).reset_index()

    #print(df_cpc_year.head(20))


    sorted = df_cpc_year.sort_values(["paper_count", "avg_score"], ascending=[False, False])
    #print(sorted.head(100))

    df_cpc_global = df_cpc_year.groupby("cpc_code").agg({
        "paper_count": "sum",
        "avg_score": "mean"
    }).reset_index()

    sorted_global = df_cpc_global.sort_values(["paper_count", "avg_score"], ascending=[False, False])
    #print(sorted_global.head(20))

    df_cpc_global["score"] = (
        df_cpc_global["paper_count"] * df_cpc_global["avg_score"]
    )
    df_cpc_global: pd.DataFrame = df_cpc_global.sort_values("score", ascending=False).nlargest(100, "score").reset_index(drop=True)

    print(len(df_cpc_global))
    print(df_cpc_global.head(20))

    
    df_cpc_global.to_parquet(OUT_DIR / f"cpc_global_scores.parquet", index=False)

    df_cpc_global = pd.read_parquet(OUT_DIR / f"cpc_global_scores.parquet")
    cpc_list = pd.read_parquet(CPC_folder / "cpc_enriched.parquet")
    cpc_list : pd.DataFrame= cpc_list[["code", "label"]].drop_duplicates(subset=["code"]).reset_index(drop=True)
    df_cpc_global : pd.DataFrame= df_cpc_global.merge(cpc_list, left_on="cpc_code", right_on="code", how="left")
    #out : pd.DataFrame = df_cpc_global[["cpc_code", "label", "paper_count", "avg_score", "score"]].copy()
    df_cpc_global.to_parquet(OUT_DIR / f"cpc_global_scores_with_labels.parquet", index=False)

    final = pd.read_parquet(OUT_DIR / f"cpc_global_scores_with_labels.parquet")
    final : pd.DataFrame= final[final["paper_count"] > 3].copy()
    final.to_parquet(OUT_DIR / f"cpc_global_scores_final.parquet", index=False)

if __name__ == "__main__":
    main()
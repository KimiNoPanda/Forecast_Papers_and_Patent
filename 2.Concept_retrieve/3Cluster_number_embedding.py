
'''
Code was run on Kaggle with GPU T4 accelerator enabled. We stabilize the cluster IDs across years by matching clusters based on their centroids. We use a cosine similarity threshold to determine if a cluster should be assigned an existing stable ID or a new one.

You might to adapt it to run with your own gpu and datasets. I pointed out the paths to the datasets and output directories, but you may need to adjust them based on your environment.
'''

# %% [code] {"execution":{"iopub.status.busy":"2026-06-11T08:40:15.854293Z","iopub.execute_input":"2026-06-11T08:40:15.855894Z","iopub.status.idle":"2026-06-11T08:40:46.725188Z","shell.execute_reply.started":"2026-06-11T08:40:15.855858Z","shell.execute_reply":"2026-06-11T08:40:46.724524Z"}}
import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer

# %% [code] {"execution":{"iopub.status.busy":"2026-06-11T08:41:40.772172Z","iopub.execute_input":"2026-06-11T08:41:40.773277Z","iopub.status.idle":"2026-06-11T08:41:40.796216Z","shell.execute_reply.started":"2026-06-11T08:41:40.773226Z","shell.execute_reply":"2026-06-11T08:41:40.795599Z"}}

MODEL_NAME = "allenai/scibert_scivocab_uncased"
STATE_FILE = "stable_cluster_state.json" # adjust the path as needed
MATCH_THRESHOLD = 0.85
EMA_ALPHA = 0.3


def load_state(path):
    if not Path(path).exists():
        return {}, 0

    with open(path) as f:
        data = json.load(f)

    centroids = {
        int(k): np.array(v, dtype=np.float32)
        for k, v in data["centroids"].items()
    }
    return centroids, data["next_id"]


def save_state(path, centroids, next_id):
    with open(path, "w") as f:
        json.dump({
            "centroids": {str(k): v.tolist() for k, v in centroids.items()},
            "next_id": next_id
        }, f)


def embed_concepts(model, concepts):
    """
    Encode ALL concepts at once on GPU.
    """
    embs = model.encode(
        concepts,
        batch_size=256,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )
    return embs.astype(np.float32)


def build_cluster_centroids(df, concept_to_emb):
    cluster_concepts = (
        df[df["cluster"] != -1]
        .groupby("cluster")["norm_concept"]
        .apply(list)
        .to_dict()
    )

    centroids = {}
    for cid, concepts in cluster_concepts.items():
        embs = np.vstack([concept_to_emb[c] for c in concepts])
        centroid = embs.mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        centroids[cid] = centroid.astype(np.float32)

    return centroids, cluster_concepts


def match_clusters(cluster_centroids, global_centroids, used):
    if len(global_centroids) == 0:
        return {}, used, {}

    stable_ids = np.array(list(global_centroids.keys()))
    hist_matrix = np.vstack(list(global_centroids.values()))  # (K, D)

    local_to_stable = {}

    for local_id, centroid in cluster_centroids.items():
        sims = hist_matrix @ centroid  # cosine similarity (already normalized)

        # mask used IDs
        for i, sid in enumerate(stable_ids):
            if sid in used:
                sims[i] = -1

        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        best_id = int(stable_ids[best_idx])

        local_to_stable[local_id] = (best_id, best_score)

    return local_to_stable, used, cluster_centroids


NOISE_ASSIGN_THRESHOLD = 0.78

def assign_stable_ids(year, model, centroids, next_id):

    input_path = Path(f"scibert_clustered_concepts_provenance_{year}.csv") # adjust to your own path
    output_path = Path(f"{year}")
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df["cluster"] = df["cluster"].astype(int)

    # 1. Embed everything ONCE
    all_concepts = df["norm_concept"].unique().tolist()
    embeddings   = embed_concepts(model, all_concepts)
    concept_to_emb = dict(zip(all_concepts, embeddings))

    # 2. Build centroids for non-noise clusters
    cluster_centroids, _ = build_cluster_centroids(df, concept_to_emb)

    # 3. Match clusters → stable IDs (vectorized)
    local_to_stable = {}

    if len(centroids) > 0 and len(cluster_centroids) > 0:
        stable_ids   = np.array(list(centroids.keys()))
        hist_matrix  = np.vstack(list(centroids.values()))
        local_ids    = list(cluster_centroids.keys())
        local_matrix = np.vstack([cluster_centroids[i] for i in local_ids])
        sim_matrix   = local_matrix @ hist_matrix.T  # (n_local, n_history)

        used = set()
        for local_id, sims in zip(local_ids, sim_matrix):
            sims_masked = sims.copy()
            for i, sid in enumerate(stable_ids):
                if sid in used:
                    sims_masked[i] = -1

            best_idx   = int(np.argmax(sims_masked))
            best_score = float(sims_masked[best_idx])
            best_id    = int(stable_ids[best_idx])

            if best_score >= MATCH_THRESHOLD:
                local_to_stable[local_id] = best_id
                used.add(best_id)
                # EMA update
                old     = centroids[best_id]
                updated = (1 - EMA_ALPHA) * old + EMA_ALPHA * cluster_centroids[local_id]
                centroids[best_id] = (updated / np.linalg.norm(updated)).astype(np.float32)
            else:
                local_to_stable[local_id] = next_id
                centroids[next_id] = cluster_centroids[local_id]
                used.add(next_id)
                next_id += 1
    else:
        # first year — assign all clusters as new
        for local_id, centroid in cluster_centroids.items():
            local_to_stable[local_id] = next_id
            centroids[next_id] = centroid
            next_id += 1

    # 4. Apply stable IDs to non-noise rows
    df["stable_cluster"] = df["cluster"].map(local_to_stable)
    df.loc[df["cluster"] == -1, "stable_cluster"] = -1
    df["stable_cluster"] = df["stable_cluster"].fillna(-1).astype(int)

    # 5. Assign noise to nearest stable cluster
    if len(centroids) > 0:
        stable_ids  = np.array(list(centroids.keys()))
        hist_matrix = np.vstack(list(centroids.values()))
    
        noise_mask     = df["cluster"] == -1
        noise_concepts = df.loc[noise_mask, "norm_concept"].tolist()
    
        if noise_concepts:
            noise_embs  = np.vstack([concept_to_emb[c] for c in noise_concepts])
            sims        = noise_embs @ hist_matrix.T
            best_scores = sims.max(axis=1)
            best_ids    = stable_ids[sims.argmax(axis=1)]
    
            # only assign if above threshold
            assigned = np.where(best_scores >= NOISE_ASSIGN_THRESHOLD, best_ids, -1)
    
            # per-cluster noise cap: max 50% of original cluster size
            cluster_sizes = df[df["cluster"] != -1].groupby("stable_cluster").size().to_dict()
            cluster_noise_count = {}
            final_assigned = []
    
            for noise_id, score in zip(assigned, best_scores):
                if noise_id == -1:
                    final_assigned.append(-1)
                    continue
                current_noise = cluster_noise_count.get(noise_id, 0)
                original_size = cluster_sizes.get(noise_id, 0)
                cap = max(3, int(original_size * 0.5))  # max 50% noise addition
    
                if current_noise < cap:
                    final_assigned.append(noise_id)
                    cluster_noise_count[noise_id] = current_noise + 1
                else:
                    final_assigned.append(-1)  # cluster full — keep as noise

            assigned = np.array(final_assigned)
    
            noise_map = dict(zip(noise_concepts, assigned.tolist()))
            df.loc[noise_mask, "stable_cluster"] = (
                df.loc[noise_mask, "norm_concept"]
                .map(noise_map)
                .fillna(-1)
                .astype(int)
            )
    
            n_assigned = (np.array(final_assigned) != -1).sum()
            print(f"  Noise assigned: {n_assigned}/{len(noise_concepts)} "
                  f"({n_assigned/len(noise_concepts):.1%})")
            print(f"  Remaining noise: {(df['stable_cluster']==-1).sum():,}")

    # 6. Save
    df.to_csv(f"stable_clustered_concepts_provenance_{year}.csv", index=False)
    print(f"[{year}] clusters={len(local_to_stable)}  "
          f"remaining_noise={( df['stable_cluster']==-1).sum()}")

    return centroids, next_id


def run(year_start, year_end):

    model = SentenceTransformer(
        MODEL_NAME,
        device="cuda" 
    )

    centroids, next_id = load_state(STATE_FILE)

    print(f"Loaded {len(centroids)} centroids\n")

    for year in range(year_start, year_end + 1):
        print(f"── Year {year} ─────────────────────")
        centroids, next_id = assign_stable_ids(year, model, centroids, next_id)
        save_state("stable_cluster_state.json", centroids, next_id)

    print(f"Done. {len(centroids)} clusters tracked.")




# %% [code] {"execution":{"iopub.status.busy":"2026-06-11T08:41:50.481667Z","iopub.execute_input":"2026-06-11T08:41:50.482482Z","iopub.status.idle":"2026-06-11T08:59:40.474125Z","shell.execute_reply.started":"2026-06-11T08:41:50.482413Z","shell.execute_reply":"2026-06-11T08:59:40.473373Z"}}
if __name__ == "__main__":
    run(2000, 2026)
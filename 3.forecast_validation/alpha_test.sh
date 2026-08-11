#!/usr/bin/env bash
set -euo pipefail

# Grid search over alpha_node and alpha_edge
ALPHA_NODE_VALUES=(0.1 0.3 0.5 0.7 1.0)
ALPHA_EDGE_VALUES=(0.1 0.2 0.3 0.5)

CLUSTER_DIR="concept_provenance/"
PROVENANCE_DIR="concept_provenance/"
NETWORK="referenced/"
PAPERS_DIR="papers/"

YEAR_START=2000
TRAIN_END=2022
VAL_START=2000
VAL_END=2025

for alpha_node in "${ALPHA_NODE_VALUES[@]}"; do
    for alpha_edge in "${ALPHA_EDGE_VALUES[@]}"; do

        node_str="${alpha_node//./}"
        edge_str="${alpha_edge//./}"

        ROOT_DIR="alpha_${node_str}_${edge_str}"
        TRAIN_DIR="${ROOT_DIR}/2022"
        VAL_DIR="${ROOT_DIR}/2025"

        echo
        echo "=================================================="
        echo "alpha_node=${alpha_node}, alpha_edge=${alpha_edge}"
        echo "Training snapshot:   ${TRAIN_DIR}"
        echo "Validation snapshot: ${VAL_DIR}"
        echo "=================================================="

        # ------------------------------------------------------
        # Stage 1: build only through 2022
        # ------------------------------------------------------
        if [[ ! -f "${TRAIN_DIR}/checkpoint_${TRAIN_END}.pkl" ]]; then
            echo "[Stage 1] Building graph from ${YEAR_START} through ${TRAIN_END}"

            mkdir -p "${TRAIN_DIR}"

            python3 6graph_build.py \
                --cluster_dir="${CLUSTER_DIR}" \
                --provenance_dir="${PROVENANCE_DIR}" \
                --network="${NETWORK}" \
                --papers_dir="${PAPERS_DIR}" \
                --out="${TRAIN_DIR}" \
                --year_start="${YEAR_START}" \
                --year_end="${TRAIN_END}" \
                --alpha_node="${alpha_node}" \
                --alpha_edge="${alpha_edge}"

            if [[ ! -f "${TRAIN_DIR}/checkpoint_${TRAIN_END}.pkl" ]]; then
                echo "ERROR: checkpoint_${TRAIN_END}.pkl was not created."
                exit 1
            fi
        else
            echo "[Stage 1] Existing 2022 checkpoint found; skipping rebuild."
        fi

        # ------------------------------------------------------
        # Stage 2: copy the 2022 state into the 2025 directory
        # ------------------------------------------------------
        if ! compgen -G "${VAL_DIR}/checkpoint_*.pkl" > /dev/null; then
            echo "[Stage 2] Copying 2022 snapshot to validation directory"

            rm -rf "${VAL_DIR}"
            mkdir -p "${VAL_DIR}"

            # Copies all metrics, state, and checkpoint files.
            rsync -a "${TRAIN_DIR}/" "${VAL_DIR}/"
        else
            echo "[Stage 2] Validation directory already contains checkpoint."
        fi

        # ------------------------------------------------------
        # Stage 3: resume at 2023 and continue through 2025
        # ------------------------------------------------------
        if [[ ! -f "${VAL_DIR}/checkpoint_${VAL_END}.pkl" ]]; then
            echo "[Stage 3] Resuming graph from ${VAL_START} through ${VAL_END}"

            python3 6graph_build.py \
                --cluster_dir="${CLUSTER_DIR}" \
                --provenance_dir="${PROVENANCE_DIR}" \
                --network="${NETWORK}" \
                --papers_dir="${PAPERS_DIR}" \
                --out="${VAL_DIR}" \
                --year_start="${VAL_START}" \
                --year_end="${VAL_END}" \
                --alpha_node="${alpha_node}" \
                --alpha_edge="${alpha_edge}"

            if [[ ! -f "${VAL_DIR}/checkpoint_${VAL_END}.pkl" ]]; then
                echo "ERROR: checkpoint_${VAL_END}.pkl was not created."
                exit 1
            fi
        else
            echo "[Stage 3] Existing 2025 checkpoint found; skipping continuation."
        fi

        echo "Completed alpha_node=${alpha_node}, alpha_edge=${alpha_edge}"
    done
done

echo
echo "All alpha combinations completed."
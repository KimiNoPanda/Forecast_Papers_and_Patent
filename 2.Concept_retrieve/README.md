# Data Retrieval Pipeline

This folder contains all scripts for retrieving, clustering and stabilizing the different concept retrieved from the paper and patent abstract.

## Prerequisites

- Some of the code was run on kaggle notebook in order to use cuda/gpu in order to speed up the processing and have a bigger memory capacity
- Abstract from paper and patent
- Referenced works information from OpenAlex

## Pipeline Overview

The scripts are numbered and should be run sequentially:

```
1retrieve_concept.py
        ↓
2HDBSCAN.py
        ↓
3Cluster_number_embedding.py
        ↓
4concept_reassigned.py
```

## Scripts

### `1retrieve_concept.py`
Retrieves concepts from paper and patent abstract using SciBERT

**Configuration:**
- Run on kaggle notebook
- Need to add the path to the different file containing paper and patent abstract by year (1 file by year)

```bash
Run on Notebook using cells
```

**Output**
Concept provenance file by year 

---

### `2HDBSCAN.py`
Cluster the different concept by year

**Configuration:**
- Run on kaggle notebook
- Need to add the path to the different file containing concept by year (1 file by year)

```bash
Run on Notebook using cells
```

**Output**
Clustered concepts by year

---

### `3Cluster_number_embedding.py`
Stabilization phase between the different cluster over the years

**Configuration:**
- Run on kaggle notebook
- Need to add the path to the different file containing cluster by year (1 file by year)

```bash
Run on Notebook using cells
```

**Output**
Stabilized clustered concept by year


---

### `4concept_reassigned.py`
Merge substring clusters into the most representative stable cluster


```bash
python3 4concept_reassigned.py
```

**Output**
Merged stabilized clustered concept by year

---

### `plot_cluster.py`
Plots hdbscan cluster statistics by year

```bash
python3 plot_cluster.py
```

---

### `plot_concept_evolution.py`
Plots the evolution over years of the number of concepts that are paper_only, patent_only, and shared (both)

```bash
python plot_concept_evolution.py --dir /path/to/csvs --out /path/to/output_dir
```

---

### `plot_reference.py`
Plots the number of references that are in-dataset vs out-of-dataset, per year

```bash
python plot_reference.py --dir /path/to/csvs --out /path/to/output_dir --csv /path/to/output_summary_csv
```

---

### `plot_stable_cluster.py`
Plots the statistics of stable clusters over the years

```bash
python plot_stable_cluster.py
```


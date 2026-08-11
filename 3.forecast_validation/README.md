# Forecasting Pipeline

This folder contains all scripts for building the graph and forecasting the technologies

## Prerequisites

- Stable clustered concept from previous step
- Additionnal information from OpenAlex


## Pipeline Overview

The scripts are numbered and should be run sequentially:


**With patent run**

```
1graph_build.py
        ↓
2forecast.py
```

**Without patent run**

```
1patentless_graph_build.py
        ↓
2patentless_forecast.py
```

**Forecasting without validation**

```
1graph_build.py
        ↓
2forecast_novalidation.py
```

**Alpha test**

```
alpha_test.sh
        ↓
alpha_test.py
```


## Scripts

### `1graph_build.py`
Builds cumulative cluster-level citation graphs from papers + patent boost.

**Configuration:**
- Might need to adapt some threshold according to the memory capacity of your computer

```bash
python3 1graph_build.py --cluster_dir=Path/to/stable_cluster/ --provenance_dir=Path/to/concept_provenance/ --network=Path/to/referenced_papers/ --papers_dir=Path/to/papers/ --out=Path/to/Output --year_start=2000 --year_end=2022 --alpha_node=0.1 --alpha_edge=0.1
```

**Output**
Output different files with citation graph data by year

---

### `2forecast.py`
Forecast future cluster importance using LightGBM regression on node + edge features.

```bash
python3 7forecast.py --metrics_dir=Path/to/metrics_for_train --val_metrics_dir=Path/to/metrics_for_validation --out=Path/to/output --target=in_strength --horizon=3 --concepts_dir=Path/to/stable_concept
```

**Output**
Forecast and validation result

---

### `1patentless_graph_build.py`
Builds cumulative cluster-level citation graphs from papers 

**Configuration:**
- Need to rerun the entire pipeline without patent but only papers to obtain patentless stable cluster

```bash
python3 1graph_build.py --cluster_dir=Path/to/stable_cluster/ --provenance_dir=Path/to/concept_provenance/ --network=Path/to/referenced_papers/ --papers_dir=Path/to/papers/ --out=Path/to/Output --year_start=2000 --year_end=2022 
```

**Output**
Output different files with citation graph data without patent boost by year


---

### `2patentless_forecast.py`
Forecast future cluster importance using LightGBM regression on node + edge features without patent information


```bash
python3 2patentless_forecast.py --metrics_dir=Path/to/metrics_for_train --val_metrics_dir=Path/to/metrics_for_validation --out=Path/to/output --target=in_strength --horizon=3 --concepts_dir=Path/to/stable_concept
```

**Output**
Forecasted and validation result without patent information

---

### `2forecast_novalidation.py`
Forecast future cluster importance using LightGBM regression on node + edge features without validation step


```bash
python3 2forecast_novalidation.py --metrics_dir=Path/to/metrics_for_train --out=Path/to/output --target=in_strength --horizon=3 --concepts_dir=Path/to/stable_concept
```



---

### `alpha_test.sh`
Build the graph with different alpha node and alpha edge values

```bash
./alpha_test.sh
```

---

### `alpha_test.py`
Run Forecasting with the different alpha graph and rank them

```bash
python alpha_test.py
```

---

### `forecast_target_test.py`
Run a sweep of forecast experiments, varying the training cutoff year, forecast horizon, and target metric (in_degree vs in_strength).

```bash
python3 forecast_target_test.py --metrics_dir=Path/to/metrics --out_root=Path/to/output --forecast_script 7forecast.py --train_years 2016 2018 2020 2022 --horizons 1 3 5 --targets in_degree in_strength
```


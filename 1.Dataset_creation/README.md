# Data Retrieval Pipeline

This folder contains all scripts for retrieving and preprocessing data from OpenAlex (academic papers) and EPO (patents), including CPC code matching between the two sources.

## Prerequisites

- OpenAlex API key
- EPO API key
- Full CPC code list (available from the [EPO website](https://www.epo.org))

## Pipeline Overview

The scripts are numbered and should be run sequentially:

```
1get_papers_abstract.py
        ↓
2CPC_combine.py
        ↓
3papers_CPC_similarities.py
        ↓
4get_patents.py
        ↓
5clean_patent.py
```

## Scripts

### `1get_papers_abstract.py`
Retrieves paper abstracts and metadata from OpenAlex.

**Configuration:**
- Add your OpenAlex API key in the script
- To change the research topic, find the `topic.id` on OpenAlex and update it in the script

```bash
python3 1get_papers_abstract.py
```

**Output**
Paper abstract and other information from OpenAlex by year

---

### `2CPC_combine.py`
Rearranges and combines the raw CPC code data downloaded from EPO into a usable format.

```bash
python3 2CPC_combine.py
```

**Output**
CPC code rearranged in parquet file

---

### `3papers_CPC_similarities.py`
Computes similarity scores between paper abstracts and CPC codes to identify 
the most relevant patent classifications for each paper.

```bash
python3 3papers_CPC_similarities.py
```

**Output**
Different parquet files containing the similarities between papers and CPC code

---

### `4get_patents.py`
Retrieves all patents from EPO matching the CPC codes identified in the previous step.

**Configuration:**
- Add your EPO API key in the script

```bash
python3 4get_patents.py
```

**Output**
Patent abstract from EPO by year

---

### `5clean_patent.py`
Cleans and normalises the raw patent data retrieved from EPO.

```bash
python3 5clean_patent.py
```

**Output**
Cleaned patent abstract
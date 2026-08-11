"""
1get_papers_abstract.py
This script retrieves papers with abstracts from the specified topics using the pyalex library and saves the data
"""

import os
import pandas as pd
import pyalex
import json
from itertools import chain



pyalex.config.api_key = "Add your API key here" # Add your API key here

def extract_work_features(work):
    """
    Extracts relevant features from a work object and appends them to the corresponding lists.
    """

    concepts_l2_l3 = [
        {
            "display_name": c.get("display_name"),
            "score": c.get("score"),
            "level": c.get("level")
        }
        for c in work.get("concepts", [])
        if c.get("level") in [2, 3]
    ]

    work_id = work["id"]

    works_rows.append({
        "id": work_id,
        "display_name": work.get("display_name"),
        "publication_year": work.get("publication_year"),
        "fwci": work.get("fwci"),
        "cited_by_count": work.get("cited_by_count"),
        "primary_topic_display_name": work.get("primary_topic_display_name"),
        "abstract": work["abstract"]
    })

    referenced_works.append({
        "id": work_id,
        "referenced_works": work.get("referenced_works", [])
    })

    for kw in work.get("keywords", []):
        keyword_rows.append({
            "work_id": work_id,
            "keyword_display_name": kw.get("display_name"),
            "keyword_score": kw.get("score"),
        })

    for c in concepts_l2_l3:
        concept_rows.append({
            "work_id": work_id,
            "concept_display_name": c.get("display_name"),
            "concept_score": c.get("score"),
            "concept_level": c.get("level"),
        })

    for cy in work.get("counts_by_year", []):
        counts_rows.append({
            "work_id": work_id,
            "year": cy.get("year"),
            "cited_by_count": cy.get("cited_by_count"),
        })


j = 0

OUT_DIR = "papers_data"
os.makedirs(OUT_DIR, exist_ok=True)

for i in range(2000,2027):
    works_rows = []
    keyword_rows = []
    concept_rows = []
    counts_rows = []
    referenced_works = []
    query =pyalex.Works().filter(**{
            "topics.id": "T10020|T10682|T11804", # Topic IDs that need to be retrieved
            "has_abstract": True,
            "publication_year": i
        })
    
    print(f"Processing year {i} with {query.count()} works")
    for work in chain.from_iterable(query.paginate(per_page=200, n_max=None)):
        print(j)
        extract_work_features(work)
        # print(works_rows)
        # print(keyword_rows)
        # print(concept_rows) 
        # print(counts_rows)
        j+=1
    df_works = pd.DataFrame(works_rows)
    df_keywords = pd.DataFrame(keyword_rows)
    df_concepts = pd.DataFrame(concept_rows)
    df_counts = pd.DataFrame(counts_rows)
    df_referenced_works = pd.DataFrame(referenced_works)
    df_works.to_parquet(f"{OUT_DIR}/papers_{i}.parquet", index=False)
    df_keywords.to_parquet(f"{OUT_DIR}/papers_keywords_{i}.parquet", index=False)
    df_concepts.to_parquet(f"{OUT_DIR}/papers_concepts_{i}.parquet", index=False)
    df_counts.to_parquet(f"{OUT_DIR}/papers_counts_by_year_{i}.parquet", index=False)
    df_referenced_works.to_parquet(f"{OUT_DIR}/papers_referenced_works_{i}.parquet", index=False)

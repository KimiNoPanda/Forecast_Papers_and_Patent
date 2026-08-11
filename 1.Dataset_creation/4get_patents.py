"""
4get_patents.py
Fetch paper abstracts from the European Patent Office (EPO) API based on CPC codes and publication years.

"""

import re
import epo_ops
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import time
import json
from lxml import etree

OUT_DIR = Path("Patent_Paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_abstracts(xml_bytes):
    """
    Extract abstracts from EPO API XML response.
    """

    ns = {
        "exchange": "http://www.epo.org/exchange",
    }

    root = etree.fromstring(xml_bytes)
    rows = []

    for doc in root.xpath(".//exchange:exchange-document", namespaces=ns):
        country = doc.get("country")
        doc_number = doc.get("doc-number")
        kind = doc.get("kind")

        # --- publication date ---
        date_node = doc.xpath(
            ".//exchange:publication-reference//exchange:date/text()",
            namespaces=ns
        )

        year = None
        if date_node:
            year = date_node[0][:4]

        # --- English abstract only ---
        abstract_nodes = doc.xpath(
            './/exchange:abstract[@lang="en"]//exchange:p//text()',
            namespaces=ns
        )

        if not abstract_nodes:
            continue

        abstract_text = " ".join(t.strip() for t in abstract_nodes if t.strip())

        rows.append({
            "publication_number": f"{country}{doc_number}{kind}",
            "year": int(year) if year else None,
            "abstract": abstract_text
        })

    return rows

def append_rows(OUTPUT_FILE, rows):
    with open(OUTPUT_FILE, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def fetch_abstracts_by_cpc(
    client: epo_ops.Client,
    year_start: int = 2000,
    year_end: int = 2006,
    batch_size: int = 50,
    sleep_seconds: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Retrieve abstracts for all publications matching a CPC code.
    """

    cpc_code = pd.read_parquet("outputs_cpc_matching/cpc_global_scores_final.parquet").copy()
    cpc_code = cpc_code["cpc_code"]
    # Common OPS CQL form for CPC search:
    check = False
    for year in range(year_start, year_end + 1):
        for cpc in cpc_code:
            cql = f'cpc="{cpc}" AND pd within "{year}0101,{year}1231"'

            all_rows: List[Dict[str, Any]] = []
            start = 1
            counter = 0

            while True:
                end = start + batch_size - 1

                try:

                    response = client.published_data_search(
                        cql=cql,
                        range_begin=start,
                        range_end=end,
                        constituents=["abstract"],
                    )

                except Exception as e:
                    if counter > 2:
                        print(f"Finished CPC {cpc} for year {year}, total rows: {len(all_rows)}")
                        break
                    counter += 1
                    start += batch_size
                    continue

                # response.content is XML

                rows = extract_abstracts(response.content)
                append_rows(OUT_DIR / f"abstracts_{year}.jsonl", rows)
                #print(rows[:1])  # debug: show first row of current batch
                if not rows:
                    if counter > 3:
                        print(f"Finished CPC {cpc} for year {year}, total rows: {len(all_rows)}")
                        break
                    counter += 1
                    start += batch_size
                    continue

                all_rows.extend(rows)
                #print(f"Fetched {len(rows)} rows for range {start}-{end} (total={len(all_rows)})")
                print(f"{cpc} {end} : {len(rows)}")

                start += batch_size
                counter = 0  # reset counter on successful fetch
                time.sleep(sleep_seconds)  # be polite with OPS quotas
            


# Some lines are not json object (Not Found)
def load_jsonl_objects(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(obj, dict):
                continue

            rows.append(obj)

    return pd.DataFrame(rows)

def append_rows(output_file: Path, rows: list[dict]) -> None:
    with open(output_file, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")




def main():
    client = epo_ops.Client(key="add your OPS key and secret here", secret="add your OPS key and secret here") # add your OPS key and secret here

    rows = fetch_abstracts_by_cpc(client, 2000,2026) # might bug if run with many years, run small batches of years to be safe


if __name__ == "__main__":
    main()






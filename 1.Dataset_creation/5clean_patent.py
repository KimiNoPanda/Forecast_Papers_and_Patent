"""

Clean patent abstracts fetched from the EPO API.

"""

import pandas as pd
from pathlib import Path
import json

OUT_DIR = Path("Patent_Paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)


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

def main():
    for year in range(2014, 2027): # Change date
        path = OUT_DIR / f"abstracts_{year}.jsonl"
        if not path.exists():
            continue

        df = load_jsonl_objects(path)

        if df.empty:
            continue

        if "abstract" not in df.columns:
            continue

        df = df[
            df["abstract"].notna()
            & df["abstract"].astype(str).str.len().gt(50)
            & ~df["abstract"].astype(str).str.contains(r"not found|404", case=False, na=False)
        ].copy()

        df.drop_duplicates(subset="abstract", inplace=True)
        df.drop_duplicates(subset="publication_number", inplace=True)
        df.to_parquet(OUT_DIR / f"patents_{year}.parquet", index=False)
            
if __name__ == "__main__":
    main()
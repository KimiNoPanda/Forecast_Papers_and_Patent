"""
2CPC_combine.py
This script combines the CPC codes from multiple text files into a single DataFrame.
"""

import pandas as pd
import re
from pathlib import Path


folder = Path("Patent_retrive/CPC_list") 
rows = []
files = list(folder.glob("cpc-section*.txt"))
print(f"Found {len(list(files))} files in {folder}")
for fi in files:
    print(f"Processing file: {fi}")
    with open(fi, "r", encoding="utf-8") as f:
        for line in f:
            parts = re.split(r"\t+", line.strip())
            parts = [p.strip() for p in parts if p.strip() != ""]
            if not parts:
                continue

            # Common patterns:
            # [code, label]
            # [code, level, label]
            if len(parts) == 2:
                code, label = parts
                level = None
            else:
                code, level, label = parts[0], parts[1], parts[2]

            #print(f"Code: {code}, Level: {level}, Label: {label}")

            rows.append({
                "code": code,
                "level": level,
                "label": label
            })

df = pd.DataFrame(rows)

# Convert level to numeric if possible
df["level"] = pd.to_numeric(df["level"], errors="coerce")
df["label"] = (
    df["label"]
    .str.lower()
    .str.replace(r"\{|\}", "", regex=True)
    .str.replace(r"\(.*?\)", "", regex=True)
)
print("\nTotal CPC entries:", len(df))
#print(df.head())

# Save result
df.to_parquet("Patent_retrive/cpc_codes.parquet", index=False)


# normalize level
df["level_num"] = pd.to_numeric(df["level"], errors="coerce")

parent_codes = []
stack = []  # stores tuples (effective_level, code)

for _, row in df.iterrows():
    code = row["code"]
    lvl = row["level_num"]

    # top hierarchy without numeric level
    if pd.isna(lvl):
        if len(code) == 1:         
            effective_level = -2
        elif len(code) == 3:       
            effective_level = -1
        elif len(code) == 4:       
            effective_level = 0
        else:
            effective_level = 0
    else:
        effective_level = int(lvl) + 1

    while stack and stack[-1][0] >= effective_level:
        stack.pop()

    parent_code = stack[-1][1] if stack else None
    parent_codes.append(parent_code)
    stack.append((effective_level, code))

df["parent_code"] = parent_codes

# map code -> label / parent
label_map = dict(zip(df["code"], df["label"]))
parent_map = dict(zip(df["code"], df["parent_code"]))

def build_full_label(code, max_depth=4):
    labels = []
    current = code
    depth = 0
    while current is not None and depth < max_depth:
        lbl = label_map.get(current)
        if pd.notna(lbl):
            labels.append(lbl)
        current = parent_map.get(current)
        depth += 1
    return " | ".join(labels)

df["full_label"] = df["code"].apply(build_full_label)

df.to_parquet("Patent_retrive/cpc_enriched.parquet", index=False)
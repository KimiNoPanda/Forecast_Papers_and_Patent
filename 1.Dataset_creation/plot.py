import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path



years = range(2000, 2027)

abstracts_list = []
papers_list = []

# ---- Load all yearly parquet files ----
for y in years:
    abs_file = f"Patent_Paper/patents_{y}.parquet"
    pap_file = f"papers_data/papers_{y}.parquet"


    df_abs = pd.read_parquet(abs_file)
    df_abs["year"] = y
    abstracts_list.append(df_abs)

    df_pap = pd.read_parquet(pap_file)
    df_pap["publication_year"] = y
    papers_list.append(df_pap)

# ---- Concatenate ----
abstracts = pd.concat(abstracts_list, ignore_index=True)
papers = pd.concat(papers_list, ignore_index=True)

# ---- Group by year ----
abs_by_year = abstracts.groupby("year").size()
pap_by_year = papers.groupby("publication_year").size()

years_all = range(2000, 2027)

abs_counts = [abs_by_year.get(y, 0) for y in years_all]
pap_counts = [pap_by_year.get(y, 0) for y in years_all]


total_abstracts = abs_by_year.sum()
total_papers = pap_by_year.sum()

print("Total abstracts/patents:", total_abstracts)
print("Total papers:", total_papers)
print("Total documents:", total_abstracts + total_papers)
# ---- Plot ----
plt.figure()

plt.plot(years_all, pap_counts, label="Papers", linewidth=2)
plt.plot(years_all, abs_counts, label="Abstracts / Patents", linewidth=2)

plt.xlabel("Year")
plt.ylabel("Number of documents")
plt.title("Evolution of Publications Over Time (2000–2026)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)

plt.show()
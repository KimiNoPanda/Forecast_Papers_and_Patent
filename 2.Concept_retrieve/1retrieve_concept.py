'''
Code was run on Kaggle with GPU T4 accelerator enabled. It extracts concepts from papers and patents using KeyBERT with SciBERT embeddings, and saves the results in parquet and CSV formats. The code includes functions for text cleaning, phrase validation, batch processing, and aggregation of results. It also supports checkpointing to resume processing in case of interruptions.

You might to adapt it to run with your own gpu and datasets. I pointed out the paths to the datasets and output directories, but you may need to adjust them based on your environment. The code is designed to handle both papers and patents, extracting keywords from abstracts and aggregating them into a final survivors table.

'''




# %% [code] {"execution":{"iopub.status.busy":"2026-06-16T13:23:01.168045Z","iopub.execute_input":"2026-06-16T13:23:01.168456Z","iopub.status.idle":"2026-06-16T13:23:06.201111Z","shell.execute_reply.started":"2026-06-16T13:23:01.168426Z","shell.execute_reply":"2026-06-16T13:23:06.200428Z"},"jupyter":{"outputs_hidden":false}}
!pip install keybert sentence-transformers scikit-learn pyarrow torch torchvision

# %% [code] {"execution":{"iopub.status.busy":"2026-06-16T13:23:22.746835Z","iopub.execute_input":"2026-06-16T13:23:22.747743Z","iopub.status.idle":"2026-06-16T13:23:53.406020Z","shell.execute_reply.started":"2026-06-16T13:23:22.747707Z","shell.execute_reply":"2026-06-16T13:23:53.405422Z"},"jupyter":{"outputs_hidden":false}}
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from collections import Counter
import re
from pathlib import Path
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
import pickle
from transformers import AutoTokenizer, AutoModel
import torch

# %% [code] {"execution":{"iopub.status.busy":"2026-06-16T13:24:06.721571Z","iopub.execute_input":"2026-06-16T13:24:06.722149Z","iopub.status.idle":"2026-06-16T13:24:07.068939Z","shell.execute_reply.started":"2026-06-16T13:24:06.722117Z","shell.execute_reply":"2026-06-16T13:24:07.068036Z"},"jupyter":{"outputs_hidden":false}}
# Stop words
for resource, path in [
    ("stopwords", "corpora/stopwords"),
    ("wordnet", "corpora/wordnet"),
    ("omw-1.4", "corpora/omw-1.4"),
    ("punkt_tab", "tokenizers/punkt_tab/english"),
]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource)

wnl = WordNetLemmatizer()

PERSONAL_STOP = [
    "presents","show","shows","paper","approach","method","methods","new","many",
    "introduces","based","using","results","result","also","whose","prove","take",
    "use","used","however","thus","can","may","proposed","novel","consider",
    "considered","study","present","evaluate","including","involving","typically",
    "recently","different","various","general","allows",
]

BLACKLIST_PHRASES = {
    "international audience",
    "proposed algorithm",
    "proposed scheme",
    "lower bound",
    "upper bound",
    "data owner",
    "experimental result",
    "numerical result",
    "simulation result",
    "performance evaluation",
}

BAD_EDGE_WORDS = {
    "rather", "like", "well", "thus", "also", "however",
    "among", "within", "without", "based", "using", "used",
    "involving", "including", "furthermore", "typically",
    "recently", "particular", "general", "different",
    "various", "proposed", "novel", "new", "many"
}

full_stop_list = nltk.corpus.stopwords.words("english") + PERSONAL_STOP

# %% [code] {"execution":{"iopub.status.busy":"2026-06-16T13:24:09.189307Z","iopub.execute_input":"2026-06-16T13:24:09.189994Z","iopub.status.idle":"2026-06-16T13:24:09.201034Z","shell.execute_reply.started":"2026-06-16T13:24:09.189954Z","shell.execute_reply":"2026-06-16T13:24:09.200215Z"},"jupyter":{"outputs_hidden":false}}
# Text cleaning
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = text.replace("\n", " ").replace("-", " ")
    text = re.sub(r"\$.*?\$", " ", text)
    text = re.sub(r"\[\d+\]", " ", text)
    text = re.sub(r"\(\d{4}\)", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\b[a-z]\b", " ", text)
    text = re.sub(r"\b\w*\d\w*\b", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_phrase(phrase: str) -> str:
    if not isinstance(phrase, str):
        return ""

    tokens = phrase.lower().strip().split()
    tokens = [wnl.lemmatize(token) for token in tokens]

    return " ".join(tokens)


def is_good_phrase(phrase: str) -> bool:
    if not phrase:
        return False

    tokens = phrase.split()

    if len(tokens) < 2 or len(tokens) > 4:
        return False

    if phrase in BLACKLIST_PHRASES:
        return False

    if tokens[0] in BAD_EDGE_WORDS or tokens[-1] in BAD_EDGE_WORDS:
        return False

    if all(token.isdigit() for token in tokens):
        return False

    if sum(len(t) == 1 for t in tokens) >= 2:
        return False

    return True

def remove_contained_phrases(scored_phrases):
    kept = []

    for score, phrase in sorted(scored_phrases, key=lambda x: (len(x[1].split()), -x[0])):
        add_phrase = True
        for _, existing in kept:
            if phrase == existing:
                add_phrase = False
                break
            if phrase in existing or existing in phrase:
                if len(phrase.split()) >= len(existing.split()):
                    add_phrase = False
                    break
        if add_phrase:
            kept.append((score, phrase))

    return kept

# %% [code] {"execution":{"iopub.status.busy":"2026-06-16T13:24:14.561785Z","iopub.execute_input":"2026-06-16T13:24:14.562227Z","iopub.status.idle":"2026-06-16T13:24:14.569569Z","shell.execute_reply.started":"2026-06-16T13:24:14.562177Z","shell.execute_reply":"2026-06-16T13:24:14.568625Z"},"jupyter":{"outputs_hidden":false}}
class SciBERTEmbedder:
    '''
    default embedder for KeyBERT using SciBERT embeddings. It uses the "allenai/scibert_scivocab_uncased" model from Hugging Face Transformers. The encode method takes a list of texts and returns their embeddings as a numpy array.
    '''
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "allenai/scibert_scivocab_uncased"
        )
        self.model = AutoModel.from_pretrained(
            "allenai/scibert_scivocab_uncased"
        ).to(self.device)
        self.model.eval()
    
    @torch.inference_mode()
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model(**inputs)
        
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        
        # Masked mean pooling
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        embeddings = summed / counts
        
        return embeddings.cpu().numpy()

# %% [code] {"execution":{"iopub.status.busy":"2026-06-16T13:24:18.794793Z","iopub.execute_input":"2026-06-16T13:24:18.795521Z","iopub.status.idle":"2026-06-16T13:24:18.848577Z","shell.execute_reply.started":"2026-06-16T13:24:18.795482Z","shell.execute_reply":"2026-06-16T13:24:18.847966Z"},"jupyter":{"outputs_hidden":false}}
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# %% [code] {"execution":{"iopub.status.busy":"2026-06-16T13:24:20.815985Z","iopub.execute_input":"2026-06-16T13:24:20.816692Z","iopub.status.idle":"2026-06-16T13:24:20.827254Z","shell.execute_reply.started":"2026-06-16T13:24:20.816654Z","shell.execute_reply":"2026-06-16T13:24:20.826439Z"},"jupyter":{"outputs_hidden":false}}
def process_batch(
    batch_texts:    list[str],
    batch_ids:      list[str],     
    embedder:       SciBERTEmbedder,
    kw_model:       KeyBERT,
    vectorizer:     CountVectorizer,
    source_type:    str,            
) -> list[dict]:
    """
    Extract concepts for a batch and return records with source_id.
 
    Returns
    list of dicts:
        norm_concept, source_id, source_type, score
    """
    records = []
    try:
        keywords_batch = kw_model.extract_keywords(
            batch_texts,
            vectorizer=vectorizer,
            top_n=5,
            use_mmr=False,
        )
        for doc_id, keywords in zip(batch_ids, keywords_batch):
            for phrase, score in keywords:
                norm = normalize_phrase(phrase)
                if is_good_phrase(norm) and len(norm) > 2:
                    records.append({
                        "norm_concept": norm,
                        "source_id":    doc_id,       # ← kept!
                        "source_type":  source_type,
                        "score":        round(float(score), 4),
                    })
    except Exception as e:
        print(f"  [warn] batch error for ids {batch_ids[:3]}…: {e}")
 
    return records

def aggregate_to_survivors(records_df: pd.DataFrame) -> pd.DataFrame:
    """
    From the raw (norm_concept, source_id, source_type, score) table,
    produce the survivors.csv-style aggregation with paper_freq,
    patent_freq, shared, etc.
 
    This is a lossless aggregation, the raw parquet keeps full provenance.
    """
    grp = records_df.groupby(["norm_concept", "source_type"]).agg(
        freq      = ("source_id", "nunique"),   # unique documents
        mean_score= ("score",     "mean"),
    ).reset_index()
 
    # pivot source_type → paper_freq / patent_freq
    paper  = grp[grp["source_type"] == "paper" ][["norm_concept","freq","mean_score"]].rename(
        columns={"freq":"paper_freq","mean_score":"paper_score"})
    patent = grp[grp["source_type"] == "patent"][["norm_concept","freq","mean_score"]].rename(
        columns={"freq":"patent_freq","mean_score":"patent_score"})
 
    agg = (
        pd.DataFrame({"norm_concept": records_df["norm_concept"].unique()})
        .merge(paper,  on="norm_concept", how="left")
        .merge(patent, on="norm_concept", how="left")
        .fillna({"paper_freq": 0, "patent_freq": 0,
                 "paper_score": 0.0, "patent_score": 0.0})
    )
    agg["paper_freq"]   = agg["paper_freq"].astype(int)
    agg["patent_freq"]  = agg["patent_freq"].astype(int)
    agg["total_freq"]   = agg["paper_freq"] + agg["patent_freq"]
    agg["mean_score"]   = (
        (agg["paper_score"]  * agg["paper_freq"] +
         agg["patent_score"] * agg["patent_freq"])
        / agg["total_freq"].clip(lower=1)
    )
    agg["paper_only"]  = (agg["paper_freq"]  > 0) & (agg["patent_freq"] == 0)
    agg["patent_only"] = (agg["patent_freq"] > 0) & (agg["paper_freq"]  == 0)
    agg["shared"]      = (agg["paper_freq"]  > 0) & (agg["patent_freq"] >  0)
 
    return agg.sort_values("total_freq", ascending=False).reset_index(drop=True)

# %% [code] {"execution":{"iopub.status.busy":"2026-06-16T13:24:26.851329Z","iopub.execute_input":"2026-06-16T13:24:26.852172Z","iopub.status.idle":"2026-06-16T13:24:26.862423Z","shell.execute_reply.started":"2026-06-16T13:24:26.852136Z","shell.execute_reply":"2026-06-16T13:24:26.861643Z"},"jupyter":{"outputs_hidden":false}}
def run_extraction(
    df:           pd.DataFrame,
    source_type:  str,           # "patent" or "paper"
    id_col:       str,           # column in df that holds the document ID
    year:         int,
    out_dir:      Path,
    batch_size:   int = 128,
    checkpoint_every: int = 10,  # save checkpoint every N batches
) -> pd.DataFrame:
    """
    Run KeyBERT extraction over df, tracking source IDs.
 
    Returns the raw records DataFrame.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"checkpoint_{source_type}_{year}.pkl"
 
    # resume from checkpoint 
    if ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            state = pickle.load(f)
        all_records  = state["records"]
        start_index  = state["next_index"]
        print(f"  Resumed from checkpoint: {len(all_records):,} records, "
              f"starting at index {start_index}")
    else:
        all_records = []
        start_index = 0
 
    # setup
    embedder  = SciBERTEmbedder()
    kw_model  = KeyBERT(model=embedder)
    full_stop = nltk.corpus.stopwords.words("english") + PERSONAL_STOP
    vectorizer = CountVectorizer(
        ngram_range=(1, 4),
        stop_words=full_stop,
        min_df=1,
        max_df=0.95,
    )
 
    # filter to rows with usable abstracts 
    rows = df.dropna(subset=["abstract"])[[id_col, "abstract"]].reset_index(drop=True)
    rows = rows.iloc[start_index:].copy()
 
    batch_texts, batch_ids = [], []
    batches_since_ckpt = 0
 
    for i, row in rows.iterrows():
        text = clean_text(row["abstract"])
        if len(text.split()) < 3:
            continue
 
        batch_texts.append(text)
        batch_ids.append(str(row[id_col]))   # ← real document ID
 
        if len(batch_texts) >= batch_size:
            records = process_batch(
                batch_texts, batch_ids, embedder, kw_model, vectorizer, source_type
            )
            all_records.extend(records)
            batch_texts, batch_ids = [], []
            batches_since_ckpt += 1
 
            if batches_since_ckpt >= checkpoint_every:
                with open(ckpt_path, "wb") as f:
                    pickle.dump({"records": all_records,
                                 "next_index": start_index + i + 1}, f,
                                protocol=pickle.HIGHEST_PROTOCOL)
                print(f"  checkpoint @ row {start_index + i + 1:,}  "
                      f"concepts so far: {len(all_records):,}")
                batches_since_ckpt = 0
 
    # final partial batch
    if batch_texts:
        records = process_batch(
            batch_texts, batch_ids, embedder, kw_model, vectorizer, source_type
        )
        all_records.extend(records)
 
    # final save 
    records_df = pd.DataFrame(all_records)
    records_df.to_parquet(out_dir / f"concepts_raw_{source_type}_{year}.parquet", index=False)
 
    # clean up checkpoint
    if ckpt_path.exists():
        ckpt_path.unlink()
 
    print(f"  {source_type} {year}: {len(records_df):,} concept-source pairs "
          f"from {records_df['source_id'].nunique():,} documents")
    return records_df

# %% [code] {"execution":{"iopub.status.busy":"2026-06-16T13:24:33.156231Z","iopub.execute_input":"2026-06-16T13:24:33.156527Z","iopub.status.idle":"2026-06-16T13:24:33.163106Z","shell.execute_reply.started":"2026-06-16T13:24:33.156497Z","shell.execute_reply":"2026-06-16T13:24:33.162285Z"},"jupyter":{"outputs_hidden":false}}
def combine_year(
    papers_raw:  pd.DataFrame | None,
    patents_raw: pd.DataFrame | None,
    out_dir:     Path,
    year:        int,
) -> pd.DataFrame:
    """
    Merge paper and patent raw records into one provenance table and
    produce the survivors aggregation.
    """
    parts = [p for p in [papers_raw, patents_raw] if p is not None and not p.empty]
    if not parts:
        raise ValueError("No records to combine")
 
    combined = pd.concat(parts, ignore_index=True)
 
    # full provenance, one row per (concept, document)
    combined.to_parquet(out_dir / f"concepts_provenance_{year}.parquet", index=False)
 
    # aggregated survivors
    survivors = aggregate_to_survivors(combined)
    survivors.to_csv(out_dir / f"survivors_{year}.csv", index=False)
 
    print(f"  {year}: {len(survivors):,} unique concepts  "
          f"({survivors['shared'].sum()} shared, "
          f"{survivors['paper_only'].sum()} paper-only, "
          f"{survivors['patent_only'].sum()} patent-only)")
    return survivors

# %% [code] {"execution":{"iopub.status.busy":"2026-06-16T12:41:24.051096Z","iopub.execute_input":"2026-06-16T12:41:24.051689Z","iopub.status.idle":"2026-06-16T12:41:24.201771Z","shell.execute_reply.started":"2026-06-16T12:41:24.051563Z","shell.execute_reply":"2026-06-16T12:41:24.200393Z"},"jupyter":{"outputs_hidden":false}}
!rm -rf /kaggle/working/checkpoint*.pkl

# %% [code] {"execution":{"iopub.status.busy":"2026-06-16T13:24:38.379452Z","iopub.execute_input":"2026-06-16T13:24:38.380010Z","iopub.status.idle":"2026-06-16T13:24:38.387254Z","shell.execute_reply.started":"2026-06-16T13:24:38.379952Z","shell.execute_reply":"2026-06-16T13:24:38.386416Z"},"jupyter":{"outputs_hidden":false}}
def main(year_start,year_end):
    papers_dir = "Papers_dir"  # adjust to your own path
    patents_dir = "Patents_dir" # adjust to your own path
    out_dir = Path("concept_with_source_no_papers")
    
    out_dir.mkdir(parents=True, exist_ok=True)
 
    for year in range(year_start,year_end + 1):
        print(f"\n══ Year {year} ══════════════════════════════")
 
        papers_raw  = None
        patents_raw = None
 

        papers_path = Path(papers_dir) / f"papers_{year}.parquet"
        if papers_path.exists():
            df_papers = pd.read_parquet(papers_path)
            # OpenAlex uses 'id' column; adjust if different
            id_col = "id" if "id" in df_papers.columns else df_papers.columns[0]
            papers_raw = run_extraction(
                df_papers, source_type="paper", id_col=id_col,
                year=year, out_dir=out_dir, batch_size=128,
            )
        else:
            print(f"  [skip] no papers file for {year}")


        patents_path = Path(patents_dir) / f"patents_{year}.parquet"
        if patents_path.exists():
            df_patents = pd.read_parquet(patents_path)
            # adjust id column name to match your patent data
            id_col = "patent_id" if "patent_id" in df_patents.columns else df_patents.columns[0]
            patents_raw = run_extraction(
                df_patents, source_type="patent", id_col=id_col,
                year=year, out_dir=out_dir, batch_size=128,
            )
        else:
            print(f"  [skip] no patents file for {year}")
 
        #combine and produce survivors
        if papers_raw is not None or patents_raw is not None:
            combine_year(papers_raw, patents_raw, out_dir, year)

# %% [code] {"execution":{"iopub.status.busy":"2026-06-16T13:24:41.358318Z","iopub.execute_input":"2026-06-16T13:24:41.358633Z"},"jupyter":{"outputs_hidden":false}}
main(2000,2026)
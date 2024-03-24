# %%
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# %%
import pandas as pd

pd.set_option("display.max_columns", None)

# %% [markdown]
# http://www.iedb.org/downloader.php?file_name=doc/bcell_full_v3_single_file.zip

# %%
dfa = pd.read_csv("bcell_full_v3.csv", header=[0, 1], nrows=10)
dfa.columns.to_list()

# %%
dfa = pd.read_csv(
    "bcell_full_v3.csv",
    header=1,
    usecols=[
        "Epitope IRI",
        "Object Type",
        "Description",
        "Organism Name",
        "Qualitative Measure",
        "Heavy Chain Type",
    ],
)
len(dfa)

# %%
dfa = dfa[dfa["Object Type"] == "Linear peptide"]
dfa = dfa[dfa["Description"].str.fullmatch(r"[A-Z]+")]
len(dfa)

# %%
dfa["Epitope ID"] = dfa["Epitope IRI"].str.extract(r"epitope/(\d+)").astype("int")
dfa["Assay Result"] = dfa["Qualitative Measure"].str[:8]
dfa["Isotype"] = dfa["Heavy Chain Type"].str.extract(r"(Ig\w)")

# %%
dfe = (
    dfa[["Epitope ID", "Description"]]
    .drop_duplicates()
    .set_index("Epitope ID")
    .sort_index()
)
dfe["Length"] = dfe["Description"].str.len()

# %%
dfr = dfa.groupby(["Epitope ID", "Assay Result"]).size().unstack(fill_value=0)

# %%
dfg = (
    dfa[dfa["Assay Result"] == "Positive"]
    .groupby(["Epitope ID", "Isotype"])
    .size()
    .unstack(fill_value=0)
)

# %%
dfb = pd.concat([dfe, dfr, dfg], axis=1)
dfb

# %%
ids_sars2 = set(
    dfa.loc[
        dfa["Organism Name"] == "Severe acute respiratory syndrome coronavirus 2",
        "Epitope ID",
    ]
)
dfb["SARS_CoV2"] = 0
dfb.loc[dfb.index.isin(ids_sars2), "SARS_CoV2"] = 1
dfb["SARS_CoV2"].value_counts()

# %%
df0 = dfb[(dfb["Negative"] >= 1) & (dfb["Positive"] == 0)]
df0 = df0[(df0["Length"] >= 1) & (df0["Length"] <= 25)]
print(len(df0))

# %%
df1 = dfb[dfb["Positive"] >= 1]
df1 = df1[(df1["Length"] >= 1) & (df1["Length"] <= 25)]
print(len(df1))

# %%
# 0 for negative
df0["Label"] = 0
# 1, 2, 3 for Ig
ig_lst = ["IgA", "IgE", "IgM"]
# 4 for other
df1["Label"] = len(ig_lst) + 1
for i, ig in enumerate(ig_lst):
    df1.loc[
        (df1[ig] > 0) & (df1[[ig_ for ig_ in ig_lst if ig_ != ig]] == 0).all(axis=1),
        "Label",
    ] = (
        i + 1
    )

print(df1["Label"].value_counts())

# %%
df0[["Description", "Label", "SARS_CoV2"]].to_csv("0.csv")
df1[["Description", "Label", "SARS_CoV2"]].to_csv("123.csv")

# %%
df0 = pd.read_csv("0.csv", index_col=0)
df123 = pd.read_csv("123.csv", index_col=0)


# %%
def df2fasta(df: pd.DataFrame, name):
    with open(f"{name}.fasta", "w") as f:
        for row in df.itertuples():
            f.write(f">{row.Index}_{row.Label}\n")
            f.write(f"{row.Description}\n")


# %%
df2fasta(df0, "0")
df2fasta(df123, "123")

# %%
df = pd.concat([df0, df123], sort=True)
df = df.fillna(0)
df["Label"] = df["Label"].astype("int64")
df["Label"].value_counts()

# %% [markdown]
# install cd-hit V4.8.1
#
# ~/cd-hit/cd-hit -i ./data/0.fasta -o ./data/0r.fasta -c 0.8
#
# ~/cd-hit/cd-hit -i ./data/123.fasta -o ./data/123r.fasta -c 0.8
#
# ~/cd-hit/cd-hit-2d -i ./data/123r.fasta -i2 ./data/0r.fasta -o ./data/0rr.fasta -c 0.8


# %%
def fasta2id(name):
    ids = set()
    with open(f"{name}.fasta") as f:
        for line in f:
            if line.startswith(">"):
                ids.add(int(line[1:].split("_")[0]))
    return ids


# %%
ids_0r = fasta2id("0rr")
ids_123r = fasta2id("123r")
print(len(ids_0r.union(ids_123r)))

# %%
dfc = df[df.index.isin(ids_0r.union(ids_123r))]
print(dfc["Label"].value_counts())

# %%
dfc[dfc["Label"] == 0].to_csv("0r.csv")
dfc[dfc["Label"] > 0].to_csv("123r.csv")

# %%
df2fasta(dfc[dfc["SARS_CoV2"] == 1], "sars")

# %%
from sklearn.model_selection import StratifiedShuffleSplit

size = 5000
splits = ("train", "valid", "test")

df_neg = pd.read_csv("0r.csv", index_col=0)
df_neg = df_neg[df_neg["SARS_CoV2"] == 0]
df_neg = df_neg.sample(frac=1, random_state=42)
if "Label" not in df_neg.columns:
    df_neg["Label"] = 0
d_neg = dict.fromkeys(splits)
d_neg["train"] = df_neg.iloc[: -2 * size]
d_neg["valid"] = df_neg.iloc[-2 * size : -size]
d_neg["test"] = df_neg.iloc[-size:]

df_pos = pd.read_csv("123r.csv", index_col=0)
df_pos = df_pos[df_pos["SARS_CoV2"] == 0]
y = df_pos["Label"]
sss = StratifiedShuffleSplit(n_splits=1, test_size=size, random_state=42)
train_index, test_index = next(sss.split(y, y))
d_pos = dict.fromkeys(splits)
d_pos["test"] = df_pos.iloc[test_index]
df_pos = df_pos.iloc[train_index]
y = df_pos["Label"]
sss = StratifiedShuffleSplit(n_splits=1, test_size=size, random_state=42)
train_index, valid_index = next(sss.split(y, y))
d_pos["valid"] = df_pos.iloc[valid_index]
d_pos["train"] = df_pos.iloc[train_index]

for split in splits:
    pd.concat([d_neg[split], d_pos[split]]).to_csv(f"{split}.csv")

# %%

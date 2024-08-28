import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
import streamlit as st
from torch.utils.data import DataLoader

from data import Vocab
from model import *
from utils import Peptides, fasta2df, predict, dict2df

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(Path(ROOT / "results" / "model.pt"), map_location=device)

st.header("BeeTLe: A Framework for Linear B-Cell Epitope Prediction and Classification")
uploaded_file = st.file_uploader("Upload a FASTA file")
if uploaded_file is not None:
    temp_dir = tempfile.TemporaryDirectory()
    temp_file = Path(temp_dir.name) / uploaded_file.name
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getvalue())
    df_in = fasta2df(temp_file)
    vocab = Vocab(max_len=25)
    dataset = Peptides(df_in, vocab)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=dataset.collate_fn)
    output_file = Path(temp_file).with_suffix(".csv").name
    lst = predict(dataloader, model)
    df_out = dict2df(lst)
    df = pd.concat([df_in, df_out], axis=1)
    csv = df.to_csv().encode("utf-8")
    st.download_button(
        label="Download result as a CSV file",
        data=csv,
        file_name=output_file,
        mime="text/csv",
    )

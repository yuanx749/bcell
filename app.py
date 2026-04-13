import inspect
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


def load_streamlit_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    model = torch.load(Path(ROOT / "results" / "model.pt"), **kwargs)
    for layer in model.transformer_enc.transformer_encoder.layers:
        if hasattr(layer, "activation_relu_or_gelu"):
            continue
        name = getattr(layer.activation, "__name__", "")
        if name == "relu":
            layer.activation_relu_or_gelu = 1
        elif name == "gelu":
            layer.activation_relu_or_gelu = 2
        else:
            layer.activation_relu_or_gelu = 0
    return model


@st.cache_resource
def get_model():
    return load_streamlit_model()


def run_prediction(file_name, file_bytes):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / file_name
        temp_file.write_bytes(file_bytes)
        df_in = fasta2df(temp_file)
    vocab = Vocab(max_len=25)
    dataset = Peptides(df_in, vocab)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=dataset.collate_fn)
    lst = predict(dataloader, get_model())
    df_out = dict2df(lst)
    df = pd.concat([df_in, df_out], axis=1)
    return df.to_csv().encode("utf-8"), Path(file_name).with_suffix(".csv").name


st.header("BeeTLe: A Framework for Linear B-Cell Epitope Prediction and Classification")
uploaded_file = st.file_uploader("Upload a FASTA file")
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if uploaded_file is not None and st.button("Predict"):
    st.session_state.prediction = run_prediction(
        uploaded_file.name,
        uploaded_file.getvalue(),
    )
if st.session_state.prediction is not None:
    csv, output_file = st.session_state.prediction
    st.download_button(
        label="Download result as a CSV file",
        data=csv,
        file_name=output_file,
        mime="text/csv",
    )

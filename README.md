# BeeTLe

A deep learning framework for linear **B**-cell **e**pitope prediction and antibody type-specific **e**pitope classification using **T**ransformer and **L**STM **e**ncoders.

## Installation

Linux is preferred. GPU is not required.

-   Clone this repo and navigate to the repo folder.
-   Install with pip, preferably in a virtual environment:

    ```bash
    pip install -r requirements.txt
    ```

-   Alternatively, to be more specific, use conda in Linux:

    ```bash
    conda env create -p ./envs -f environment.yml
    ```

## Usage

Run command like below. It takes a few seconds to predict 10000 peptides.

```bash
python cli.py -i input.fasta -o output.csv
```

To show help, run `python cli.py -h`. The input is a FASTA file of peptides. The output is a table with following columns:

-   identifier: FASTA header.
-   sequence: FASTA sequence.
-   score: Probability of being epitope.
-   epitope: {0, 1}. 1 for epitope (score > 0.5).
-   Ig: {A, E, M}. The antibody most probably binds to in these three types.

## Data

Follow the notebook `data/dataset.py` to generate datasets, in which redundancy and false negatives are reduced. The raw data is on [figshare](https://doi.org/10.6084/m9.figshare.22139777).

## Development

The code is designed to be reusable and extensible. It may be adopted in other peptide classification tasks. Some useful components are:

-   Loss functions: logit-adjusted, focal; sigmoid, softmax.
-   LSTM (packed variable length input), Transformer encoder, attention.
-   Amino acid encoder.

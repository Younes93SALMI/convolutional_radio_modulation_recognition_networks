# Datasets

The RadioML datasets are **not included** in this repository due to their large size.

Before running the notebook, please download the required dataset and place the files in this directory.

## Required files

```
datasets/
├── RML2016.pkl
└── RML2016_metadata.pkl
```

## Dataset

This implementation uses the **RadioML2016.10A** dataset introduced in:

> T. J. O'Shea, J. Corgan, and T. C. Clancy,
> *Convolutional Radio Modulation Recognition Networks*,
> arXiv:1602.04105, 2016.

The dataset can be obtained from the official RadioML repository or other publicly available mirrors.
e.g., https://www.kaggle.com/datasets/zaslee/rml2016-10a

## Directory structure

After downloading the dataset, your directory should look like

```text
datasets/
├── README.md
├── RML2016.pkl
└── RML2016_metadata.pkl
```

## Notes

- Do **not** rename the dataset files.
- The notebook expects the dataset to be located inside the `datasets/` directory.
- The metadata file contains the modulation labels and SNR information required for preprocessing.

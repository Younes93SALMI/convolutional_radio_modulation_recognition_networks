# Convolutional Radio Modulation Recognition Networks

PyTorch/TensorFlow implementation of the DL architectures proposed in

> Timothy J. O'Shea, Johnathan Corgan, T. Charles Clancy,
> *Convolutional Radio Modulation Recognition Networks*,
> arXiv:1602.04105, 2016.

This repository reproduces the well-known VT-CNN2 model and others for **Automatic Modulation Classification (AMC)** using the RadioML 2016.10A dataset.

---

# Overview

Automatic Modulation Classification (AMC) is an important task in modern wireless communications, cognitive radio, and spectrum sensing.

VT-CNN2 was one of the first deep learning architectures demonstrating that raw IQ samples can outperform expert features for modulation recognition, particularly at low SNR.

The original paper proposed a shallow convolutional neural network operating directly on complex-valued IQ samples represented as two real-valued channels (I and Q).

---

# Architecture

The implemented network follows the VT-CNN2 architecture:

Input
```
(2 × 128)
```

↓

Conv2D

↓

ReLU

↓

Dropout

↓

Conv2D

↓

ReLU

↓

Dropout

↓

Flatten

↓

Dense

↓

Dropout

↓

Softmax

---

# Supported Datasets

The implementation supports

- RadioML2016.10A

Each IQ sample is represented as

```
128 × 2
```

---

# Features

- Reproduction of the VT-CNN2 architecture
- Automatic dataset loading
- IQ preprocessing
- One-hot label generation
- Train / Validation / Test splitting
- Accuracy vs SNR evaluation
- Confusion matrices
- Model checkpointing
- GPU support
- Reproducible training

---

# Repository Structure

```
.
├── data/
│   ├── RML2016.10A.pkl
│   └── RML2018.01A.h5
│
├── models/
│   └── vtcnn2.py
│
├── notebooks/
│   └── VT_CNN2.ipynb
│
├── results/
│   ├── confusion_matrix/
│   ├── accuracy_vs_snr/
│   └── checkpoints/
│
├── utils/
│   ├── dataset.py
│   ├── preprocessing.py
│   └── metrics.py
│
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/Younes93SALMI/convolutional_radio_modulation_recognition_networks.git

cd VT-CNN2
```

Create a virtual environment

```bash
python -m venv venv
```

Activate it

Linux/macOS

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Training

Open

```
VT_CNN2.ipynb
```

or execute

```bash
python train.py
```

Training parameters can be modified, including

- epochs
- batch size
- optimizer
- learning rate
- dropout
- dataset selection

---

# Evaluation

The implementation provides

- Overall accuracy
- Accuracy versus SNR
- Per-class accuracy
- Confusion matrix
- Training history
- Validation curves

Example output:

```
Accuracy vs SNR

-20 dB    8.7%
-18 dB    8.5%
...
0 dB     71%
...
18 dB    94%
```

---

# Requirements

- Python ≥ 3.11
- TensorFlow / Keras
- NumPy
- SciPy
- h5py
- matplotlib
- scikit-learn
- pandas

---

# Reference

If you use this repository, please cite the original paper:

```bibtex
@article{oshea2016cnn,
  title={Convolutional Radio Modulation Recognition Networks},
  author={O'Shea, Timothy J. and Corgan, Johnathan and Clancy, T. Charles},
  journal={arXiv preprint arXiv:1602.04105},
  year={2016}
}
```

---

# Acknowledgements

This repository reproduces the architecture proposed by O'Shea et al. for educational and research purposes. The original paper introduced one of the earliest convolutional neural network architectures for end-to-end radio modulation classification using raw IQ samples and demonstrated significant performance improvements over expert feature-based classifiers at low SNR. :contentReference[oaicite:1]{index=1}

---

# License

This project is released under the MIT License.

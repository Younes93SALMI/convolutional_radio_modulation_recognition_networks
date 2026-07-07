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

Each IQ example is represented as

```
128 × 2
```

---

---

# Dataset Requirement ⚠️

Due to GitHub's file size limitations, the **RadioML2016.10A dataset is not included** in this repository.

Before running any notebook or training a model, you **must** download the dataset and place the required files inside the `datasets/` directory.

The expected directory structure is

```text
datasets/
├── README.md
├── RML2016.pkl
└── RML2016_metadata.pkl
```

Instructions for obtaining and placing the dataset are provided in

```text
datasets/README.md
```

The implementation expects these filenames and locations. Renaming or relocating the files may result in dataset loading errors.

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
├── .vscode/
│
├── datasets/
│   ├── RML2016.pkl                 # RadioML 2016.10A dataset
│   └── RML2016_metadata.pkl        # Dataset metadata
│
├── DeepLearning/
│   ├── __init__.py
│   ├── cnn1.py                     # CNN1 model
│   ├── cnn2.py                     # VT-CNN2 model
│   └── dnn.py                      # Fully-connected DNN baseline
│
├── results/
│   ├── dnn/                        # DNN training results
│   ├── 
.
.
.
│   └── cnn/                        # CNN training results
│
└── automatic_modulation_classification.ipynb
                                 # Main notebook for training,
                                 # evaluation, and visualization
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

# References

If you use this repository, please have a look at my papers:

```bibtex
@article{salmi2026xai,
  title={XAI-Guided Physical Adversarial Machine Learning Attacks on Automatic Modulation Classification},
  author={Salmi, Younes and Bogucka, Hanna},
  journal={IEEE Journal on Selected Areas in Communications},
  year={2026},
  publisher={IEEE}
}
```
```bibtex
@inproceedings{salmi2025physical,
  title={Physical backdoor attack against deep learning-based modulation classification},
  author={Salmi, Younes and Bogucka, Hanna},
  booktitle={2025 IEEE International Mediterranean Conference on Communications and Networking (MeditCom)},
  pages={1--6},
  year={2025},
  organization={IEEE}
}
```
```bibtex
@inproceedings{salmi2025mitigating,
  title={Mitigating evasion attacks in fog computing resource provisioning through proactive hardening},
  author={Salmi, Younes and Bogucka, Hanna},
  booktitle={2025 Joint European Conference on Networks and Communications \& 6G Summit (EuCNC/6G Summit)},
  pages={769--774},
  year={2025},
  organization={IEEE}
}
```
```bibtex
@inproceedings{salmi2026vulnerability,
  title={On the Vulnerability of Deep Automatic Modulation Classifiers to Explainable Backdoor Threats},
  author={Salmi, Younes and Bogucka, Hanna},
  booktitle={2026 IEEE 23rd Consumer Communications \& Networking Conference (CCNC)},
  pages={1--4},
  year={2026},
  organization={IEEE}
}
```
---

# Acknowledgements

This repository reproduces the architecture proposed by O'Shea et al. for educational and research purposes. The original paper introduced one of the earliest convolutional neural network architectures for end-to-end radio modulation classification using raw IQ samples and demonstrated significant performance improvements over expert feature-based classifiers at low SNR. :contentReference[oaicite:1]{index=1}

---

# License

This project is released under the MIT License.

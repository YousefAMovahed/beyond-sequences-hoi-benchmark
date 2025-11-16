# Beyond Sequences: A Benchmark for Atomic Hand-Object Interaction
Official Code Repository for the paper: *Beyond Sequences: A Benchmark for Atomic Hand-Object Interaction Using a Static RNN Encoder*

[![Paper](https://img.shields.io/badge/Paper-LINK%20(TBD)-blue)](https://LINK_TO_YOUR_PAPER_PDF)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides the dataset and code necessary to reproduce the key findings of our paper. We investigate the classification of atomic hand-object interaction states (e.g., *approaching*, *grabbing*, *holding*) using statistical-kinematic features.

Our central, counter-intuitive finding is that for richly engineered features, a **Bidirectional RNN used as a static encoder (with `seq_length=1`)** dramatically outperforms both standard MLPs and conventional temporal RNNs, achieving **97.60% accuracy** and resolving the most ambiguous transitional classes.

## The Benchmark Dataset

The core of this benchmark is located in:
* `data/MANIAC_benchmark_dataset.csv`

This file contains the final, processed dataset of **27,476 statistical-kinematic feature vectors**. Each row is one sample, derived from a 10-frame window, and labeled with one of five atomic states. The full data preparation pipeline (from raw MANIAC videos) is archived in `notebooks/Data_Preparation_Archive.ipynb` for transparency.

## Repository Structure

This repository is structured to follow the "evolutionary path" of experiments detailed in the paper:

* `1_Baseline_MLP.py`: Reproduces the **Optimized MLP Baseline** (Model 3 in the paper).
* `2_Temporal_BiRNN.py`: Reproduces the **Temporal Hypothesis** using a Bi-RNN with `seq_length=5` (Model 6).
* `3_Champion_Model_Static_RNN.py`: Reproduces the **Breakthrough Finding** and the **Final Champion Model** using a Bi-RNN with `seq_length=1` and Optuna hyperparameter tuning (Model 8).

## How to Reproduce Results

To reproduce our findings, follow these steps:

**1. Clone the repository:**

git clone [https://github.com/YOUR_USERNAME/beyond-sequences-hoi-benchmark.git](https://github.com/YOUR_USERNAME/beyond-sequences-hoi-benchmark.git)
cd beyond-sequences-hoi-benchmark
(Replace YOUR_USERNAME with your GitHub username)

**2. Install dependencies: (We will add the requirements.txt file in the next step)


pip install -r requirements.txt
**3. Run the experiments:

To reproduce the Optimized MLP baseline (Model 3):


python 1_Baseline_MLP.py
(This script will run K-Fold cross-validation and print the average baseline metrics)

To reproduce the Temporal Bi-RNN (Model 6):


python 2_Temporal_BiRNN.py
(This script will test the temporal hypothesis and print the average metrics for the seq_length=5 model)

To reproduce the Champion Model (Model 8):



python 3_Champion_Model_Static_RNN.py
(This script will run the full Optuna search to find the best hyperparameters for the seq_length=1 static encoder and will train, evaluate, and save the final champion model, printing its detailed classification report.)

Citation
If you find this work useful in your research, please consider citing our paper:


@inproceedings{azizi2024beyond,
  title={Beyond Sequences: A Benchmark for Atomic Hand-Object Interaction Using a Static RNN Encoder},
  author={Azizi Movahed, Yousef and Ziaeetabar, Fatemeh},
  booktitle={Conference (TBD)},
  year={2024 (TBD)}
}

# Beyond Sequences: A Benchmark for Atomic Hand-Object Interaction
Official Code Repository for the paper: *Beyond Sequences: A Benchmark for Atomic Hand-Object Interaction Using a Static RNN Encoder*

Paper License: MIT

This repository provides the dataset and code necessary to reproduce the key findings of our paper. We investigate the classification of atomic hand-object interaction states (e.g., *approaching*, *grabbing*, *holding*) using statistical-kinematic features.

Our central, counter-intuitive finding is that for richly engineered features, a **Bidirectional RNN used as a static encoder (with `seq_length=1`)** dramatically outperforms both standard MLPs and conventional temporal RNNs, achieving **97.60% accuracy** (in random split settings) and resolving the most ambiguous transitional classes.

## The Benchmark Dataset

The core of this benchmark is located in:
* `data/MANIAC_benchmark_dataset.csv`

This file contains the final, processed dataset of **27,476 statistical-kinematic feature vectors**. Each row is one sample, derived from a 10-frame window, and labeled with one of five atomic states. The full data preparation pipeline (from raw MANIAC videos) is archived in `notebooks/Data_Preparation_Archive.ipynb` for transparency.

## Repository Structure

This repository is structured to follow the "evolutionary path" of experiments detailed in the paper. It is divided into two phases: **Phase 1 (Initial Findings)** and **Phase 2 (Rigorous Benchmarking)**.

### Phase 1: Initial Explorations (Random Split)
* `1_Baseline_MLP.py`: Reproduces the **Optimized MLP Baseline** (Model 3 in the paper).
* `2_Temporal_BiRNN.py`: Reproduces the **Temporal Hypothesis** using a Bi-RNN with `seq_length=5` (Model 6).
* `3_Champion_Model_Static_RNN.py`: Reproduces the **Breakthrough Finding** and the **Final Champion Model** using a Bi-RNN with `seq_length=1` and Optuna hyperparameter tuning (Model 8).

### Phase 2: Rigorous Evaluation (Group K-Fold & Ablation) [NEW]
* `4_Rigorous_Group_Benchmark.py`: **(Crucial)** Runs the comprehensive benchmark comparing SVM, RF, MLP, and Bi-RNN (Static vs. Temporal) using a strict **Leave-One-Video-Out (Group K-Fold)** protocol. This script reproduces the final "Truth Table" presented in the revised paper, addressing data leakage concerns and testing true generalization.
* `5_Ablation_Study.py`: Reproduces the **Ablation Study**, quantifying the contribution of Interaction vs. Kinematic features to the model's performance.

## How to Reproduce Results

To reproduce our findings, follow these steps:

**1. Clone the repository:**

```bash
git clone [https://github.com/YousefAMovahed/beyond-sequences-hoi-benchmark.git](https://github.com/YousefAMovahed/beyond-sequences-hoi-benchmark.git)
cd beyond-sequences-hoi-benchmark

```

**2. Install dependencies:**

```bash
pip install -r requirements.txt

```

**3. Run Phase 1 Experiments (Feature Discriminability):**

To reproduce the Optimized MLP baseline (Model 3):

```bash
python 1_Baseline_MLP.py

```

*(This script will run K-Fold cross-validation and print the average baseline metrics)*

To reproduce the Temporal Bi-RNN (Model 6):

```bash
python 2_Temporal_BiRNN.py

```

*(This script will test the temporal hypothesis and print the average metrics for the seq_length=5 model)*

To reproduce the Champion Model (Model 8):

```bash
python 3_Champion_Model_Static_RNN.py

```

*(This script will run the full Optuna search to find the best hyperparameters for the seq_length=1 static encoder and will train, evaluate, and save the final champion model, printing its detailed classification report.)*

**4. Run Phase 2 Experiments (Rigorous Generalization):**

To reproduce the final rigorous benchmark results (Table 2 in the revised paper):

```bash
python 4_Rigorous_Group_Benchmark.py

```

*Output: A comparative table of Overall Accuracy, Weighted F1, and Grabbing F1 for all models under the strict Group Split protocol. This validates that the Static Bi-RNN maintains superior robustness (84.03%) and Grabbing detection (F1: 0.36) even when tested on unseen videos.*

To verify feature importance (Table 3):

```bash
python 5_Ablation_Study.py

```

## Note on Evaluation Protocols

The repository supports two evaluation protocols to demonstrate different aspects of the model:

1. **Random Split (Phase 1):** Used to assess the *instantaneous kinematic discriminability* of the features. High accuracy here (97.60%) confirms that the feature vectors are rich enough to describe the state locally.
2. **Group K-Fold (Phase 2):** Used to assess *internal generalization*. By testing on unseen videos, we provide a rigorous, lower-bound performance estimate (84.03%) that reflects real-world challenges, confirming the model does not rely on background memorization.

## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{azizi2025beyond,
  title={Beyond Sequences: A Benchmark for Atomic Hand-Object Interaction Using a Static RNN Encoder},
  author={Azizi Movahed, Yousef and Ziaeetabar, Fatemeh},
  booktitle={Conference (TBD)},
  year={2025 (TBD)}
}

```


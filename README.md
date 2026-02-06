---

# Beyond Sequences: A Benchmark for Atomic Hand-Object Interaction

**Official Code Repository for the paper:** *Beyond Sequences: A Benchmark for Atomic Hand-Object Interaction Using a Static RNN Encoder*

This repository provides the dataset, code, and rigorous evaluation protocols necessary to reproduce the findings of our research. We address the fine-grained classification of **atomic hand-object interaction states** (e.g., *approaching*, *grabbing*, *holding*) using interpretable statistical-kinematic features.

## Key Contributions & Findings

1. **The "Static Encoder" Discovery:** We demonstrate that a **Bidirectional RNN configured as a static encoder (`seq_length=1`)** functions as a high-capacity Gated MLP, significantly outperforming temporal variants (`seq_length=5`) in capturing transient states like 'grabbing'.
2. **Rigorous Evaluation Protocol:** Unlike prior works that may rely on randomized splits, we establish a strict **Leave-One-Group-Out (Group K-Fold)** benchmark. This ensures no video-leakage occurs between training and testing, providing a realistic assessment of generalization.
3. **Benchmark Results:**
* **Baseline (Random Split):** Our model achieves **97.60% accuracy**, demonstrating high instantaneous feature discriminability.
* **Rigorous (Group Split):** Under strict video-independent testing, our model maintains **84.03% accuracy** and achieves the highest robustness in the critical 'grabbing' class (**F1: 0.36**), outperforming SVM, Random Forest, and MLP baselines.



---

## The Benchmark Dataset

The core of this benchmark is located in:

* `data/MANIAC_benchmark_dataset.csv`

This file contains the final, processed dataset of **27,476 statistical-kinematic feature vectors**. Each row represents a sample derived from a 10-frame sliding window, labeled with one of five atomic states. The raw data originates from the MANIAC dataset, processed through our custom feature engineering pipeline.

---

## Repository Structure

This repository is organized into two phases, reflecting the evolutionary path of our experiments:

### Phase 1: Initial Explorations (Random Split)

*These scripts reproduce the initial findings regarding feature discriminability (as detailed in the early sections of the paper).*

* `1_Baseline_MLP.py`: Reproduces the **Optimized MLP Baseline** (Model 3).
* `2_Temporal_BiRNN.py`: Reproduces the **Temporal Hypothesis** using a Bi-RNN with `seq_length=5` (Model 6).
* `3_Champion_Model_Static_RNN.py`: Reproduces the initial **Breakthrough Finding** using a Bi-RNN with `seq_length=1` (Model 8).

### Phase 2: Rigorous Benchmarking (Group K-Fold) [MAIN]

*These scripts represent the rigorous evaluation protocol and the final reported benchmark results.*

* `4_Rigorous_Group_Benchmark.py`: **(Crucial)** Runs the comprehensive benchmark comparing **SVM, RF, MLP, Bi-RNN (Static), and Bi-RNN (Temporal)** using the strict **Group K-Fold** protocol. This script reproduces the final "Truth Table" in the paper.
* `5_Ablation_Study.py`: Reproduces the **Ablation Study**, quantifying the contribution of Interaction vs. Kinematic features to the model's performance.

---

## How to Reproduce Results

To reproduce our findings, follow these steps:

### 1. Setup

```bash
git clone https://github.com/YousefAMovahed/beyond-sequences-hoi-benchmark.git
cd beyond-sequences-hoi-benchmark
pip install -r requirements.txt

```

### 2. Run the Rigorous Benchmark (The "Truth Table")

This is the most important script, reproducing the final comparative results (Table 2 in the paper):

```bash
python 4_Rigorous_Group_Benchmark.py

```

*Output: A comparative table of Overall Accuracy, Weighted F1, and Grabbing F1 for all models under the strict Group Split protocol.*

### 3. Run Feature Analysis

To verify the importance of interaction features (Table 3 in the paper):

```bash
python 5_Ablation_Study.py

```

### 4. Reproduce Initial Phase (Optional)

To see the high discriminability in the random split setting (Table 1 in the paper):

```bash
python 3_Champion_Model_Static_RNN.py

```

---

## Citation

If you find this benchmark or code useful in your research, please cite our paper:

```bibtex
@inproceedings{azizi2025beyond,
  title={Beyond Sequences: A Benchmark for Atomic Hand-Object Interaction Using a Static RNN Encoder},
  author={Azizi Movahed, Yousef and Ziaeetabar, Fatemeh},
  booktitle={Proceedings of the IEEE International Conference on [Conference Name]},
  year={2025},
  note={Under Review}
}

```

# Seq2Seq

# Seq2Seq Code Generation: RNN vs. LSTM vs. Attention

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)

This repository contains a comprehensive comparative analysis of Sequence-to-Sequence (Seq2Seq) models for **Text-to-Code Generation**. It evaluates three distinct architectures—**Vanilla RNN**, **LSTM**, and **Attention-based LSTM**—on the task of generating Python code from natural language docstrings.

The project demonstrates why simple RNNs fail at structured tasks and how Attention mechanisms significantly improve semantic understanding and syntax generation.

---

## 📌 Table of Contents
- [Introduction](#-introduction)
- [Dataset](#-dataset)
- [Model Architectures](#-model-architectures)
- [Installation & Usage](#-installation--usage)
- [Results & Analysis](#-results--analysis)
- [Visualization](#-visualization)
- [Contributing](#-contributing)

---

## 📖 Introduction

Automatic code generation requires a model to understand user intent (semantics) and produce valid executable code (syntax). This project explores the evolution of Seq2Seq models for this task:

1.  **Vanilla RNN:** A baseline model that struggles with long-term dependencies.
2.  **LSTM:** Improves upon RNNs by preserving gradient flow, allowing for better syntactic structure.
3.  **Attention Mechanism (Bahdanau):** Allows the decoder to "focus" on specific parts of the input docstring, solving the information bottleneck problem.

---

## 📊 Dataset

We use the **CodeSearchNet (Python)** dataset, specifically the subset hosted by `Nan-Do` on Hugging Face.
Dataset link: https://huggingface.co/datasets/Nan-Do/code-search-net-python

* **Source:** Function Docstrings (Natural Language)
* **Target:** Python Function Bodies
* **Vocabulary:** ~8,000 source tokens, ~10,000 target tokens (Frequency threshold > 10)
* **Filtering:** Max source length = 50, Max target length = 80

---

## 🏗 Model Architectures

All models share the same hyperparameters for fair comparison:
* **Embedding Dim:** 128
* **Hidden Dim:** 256
* **Layers:** 2
* **Dropout:** 0.5

### 1. Vanilla RNN
* **Encoder:** Standard `nn.RNN`
* **Decoder:** Standard `nn.RNN`
* **Limitation:** Suffers heavily from the vanishing gradient problem.

### 2. LSTM (Long Short-Term Memory)
* **Encoder/Decoder:** `nn.LSTM`
* **Advantage:** Cell states preserve information over longer sequences, improving syntax generation (indentation, matching brackets).

### 3. Attention-Based LSTM
* **Encoder:** Bidirectional LSTM
* **Attention:** Bahdanau (Additive) Attention
* **Decoder:** LSTM with concatenated context vectors
* **Advantage:** Dynmaically aligns code tokens (e.g., `args`) with docstring keywords (e.g., "arguments").

---

## ⚙️ Installation & Usage

### Prerequisites
* Python 3.8+
* PyTorch
* Matplotlib, Seaborn, NLTK, Datasets

### Installation
```bash
git clone [https://github.com/swadhinpal/Seq2Seq.git](https://github.com/swadhinpal/Seq2Seq.git)
cd Seq2Seq

### Running the Notebook
The core logic is contained in the Jupyter Notebooks.

1. Open `traing_bluescore.ipynb`.
2. Run the cells sequentially to:
   * Download and preprocess data.
   * Train the three models.
   * Visualize loss curves and attention maps.


### 📈 Results & Analysis

Detailed analysis can be found in `attention-error-weight-heatmap.ipynb`.

**Performance Metrics (Test Set)**

| Model | BLEU Score | Token Accuracy | Exact Match | Training Stability |
| :--- | :--- | :--- | :--- | :--- |
| **Vanilla RNN** | 11.16 | 19.13% | 0.00% | Poor (Exploding Gradients) |
| **LSTM** | 19.64 | 18.67% | 0.00% | Stable |
| **Attention** | **21.26** | **21.06%** | **0.00%** | **Best Convergence** |

### 📏 Performance by Docstring Length (Token Accuracy)

| Docstring Length | Token Count | Vanilla RNN | LSTM | Attention |
| :--- | :--- | :--- | :--- | :--- |
| **0-10** | 30,577 | 19.47% | 20.34% | **23.00%** |
| **10-20** | 21,726 | 18.20% | 17.83% | **20.41%** |
| **20-30** | 7,060 | 19.52% | 16.69% | **19.82%** |
| **30+** | 12,645 | 18.19% | 15.72% | **18.78%** |

### Qualitative Findings

    RNN: Produced repetitive nonsense (e.g., def def def self self).

    LSTM: Learned structure (def func(): return) but hallucinated variable names.

    Attention: Correctly copied variable names from the docstring to the code body.

### 🖼 Visualization
Training Dynamics

The loss curves clearly show the Attention model converging faster and achieving a lower validation loss compared to the standard LSTM and RNN.
Attention Heatmaps

We visualize the alignment between docstring words and generated code tokens.

    Diagonal Pattern: Indicates the model is correctly processing the sentence sequentially.

    Vertical Bars: Indicates the model is "stuck" on a specific word (common in failed generations).

(Note: Add an actual image to an assets folder in your repo for this link to work)

### 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements.

    Fork the Project

    Create your Feature Branch (git checkout -b feature/AmazingFeature)

    Commit your Changes (git commit -m 'Add some AmazingFeature')

    Push to the Branch (git push origin feature/AmazingFeature)

    Open a Pull Request
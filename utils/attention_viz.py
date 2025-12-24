import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_attention(attn_weights, input_tokens, output_tokens, filename):
    """
    Plot a heatmap of attention weights.
    attn_weights: numpy array of shape (output_len, input_len).
    input_tokens: list of input tokens (docstring).
    output_tokens: list of output tokens (generated code).
    """
    plt.figure(figsize=(6,5))
    sns.heatmap(attn_weights, xticklabels=input_tokens, yticklabels=output_tokens, cmap='Blues')
    plt.xlabel("Input (docstring tokens)")
    plt.ylabel("Output (code tokens)")
    plt.title("Attention Heatmap")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

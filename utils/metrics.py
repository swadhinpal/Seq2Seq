import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def compute_bleu(references, hypotheses):
    """
    Compute BLEU score. 
    references: list of reference token lists (each a list of tokens).
    hypotheses: list of hypothesis token lists.
    """
    # NLTK expects list of list of reference lists and list of hypotheses
    refs = [[ref] for ref in references]
    bleu = corpus_bleu(refs, hypotheses, smoothing_function=SmoothingFunction().method1)
    return bleu

def compute_token_accuracy(predicted, target):
    """Token-level accuracy (ignoring PAD)."""
    total, correct = 0, 0
    for p, t in zip(predicted, target):
        for pi, ti in zip(p, t):
            if ti == 0:  # assuming PAD index is 0
                continue
            if pi == ti:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0

def compute_exact_match(predicted, target):
    """Exact sequence match accuracy."""
    match = 0
    for p, t in zip(predicted, target):
        # Trim padding and compare whole sequence
        if list(p) == list(t):
            match += 1
    return match / len(target)

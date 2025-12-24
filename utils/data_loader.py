import random
import torch
from datasets import load_dataset
from utils.tokenizer import tokenize_docstring, tokenize_code, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN
from collections import Counter

DOC_MAX_LEN = 50
CODE_MAX_LEN = 80

def build_vocab(sequences, min_freq=1):
    """Build token-to-index dictionary with special tokens."""
    counter = Counter([tok for seq in sequences for tok in seq])
    tokens = [tok for tok, freq in counter.items() if freq >= min_freq]
    # Reserve indices for special tokens
    itos = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + sorted(tokens)
    stoi = {tok: i for i, tok in enumerate(itos)}
    return stoi, itos

def encode_sequence(seq, stoi, max_len):
    """Convert token list to indices, adding EOS and padding."""
    idxs = [stoi.get(tok, stoi[UNK_TOKEN]) for tok in seq]
    idxs = idxs[:max_len-1]  # reserve space for EOS
    idxs.append(stoi[EOS_TOKEN])
    if len(idxs) < max_len:
        idxs += [stoi[PAD_TOKEN]] * (max_len - len(idxs))
    return idxs

def prepare_data():
    """Load CodeSearchNet data, filter, tokenize, and prepare train/val/test splits."""
    dataset = load_dataset("Nan-Do/code-search-net-python", split="train")
    # Filter by token length
    examples = []
    for ex in dataset:
        doc = ex["docstring"] if ex["docstring"] is not None else ""
        code = ex["code"] if ex["code"] is not None else ""
        doc_toks = tokenize_docstring(doc)
        code_toks = tokenize_code(code)
        if 0 < len(doc_toks) <= DOC_MAX_LEN and 0 < len(code_toks) <= CODE_MAX_LEN:
            examples.append((doc_toks, code_toks))
    random.shuffle(examples)
    examples = examples[:9000]  # use 9000 for training (approx)
    train_data = examples[:8000]
    val_data = examples[8000:8500]
    test_data = examples[8500:9000]

    # Build vocabularies on training data
    doc_seqs = [ex[0] for ex in train_data]
    code_seqs = [ex[1] for ex in train_data]
    doc_stoi, doc_itos = build_vocab(doc_seqs)
    code_stoi, code_itos = build_vocab(code_seqs)

    # Encode and save data
    def encode_list(data, doc_stoi, code_stoi):
        enc = []
        for doc_toks, code_toks in data:
            doc_idx = encode_sequence(doc_toks, doc_stoi, DOC_MAX_LEN)
            code_idx = encode_sequence(code_toks, code_stoi, CODE_MAX_LEN)
            enc.append((doc_idx, code_idx))
        return enc

    train_enc = encode_list(train_data, doc_stoi, code_stoi)
    val_enc = encode_list(val_data, doc_stoi, code_stoi)
    test_enc = encode_list(test_data, doc_stoi, code_stoi)
    torch.save({
        'train': train_enc, 'val': val_enc, 'test': test_enc,
        'doc_stoi': doc_stoi, 'doc_itos': doc_itos,
        'code_stoi': code_stoi, 'code_itos': code_itos
    }, 'checkpoints/dataset.pt')
    print("Data prepared: {} train, {} val, {} test examples.".format(
        len(train_enc), len(val_enc), len(test_enc)))

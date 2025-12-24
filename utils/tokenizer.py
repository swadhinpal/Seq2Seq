import re
import io
import tokenize

# Special tokens
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

def tokenize_docstring(text):
    """Simple whitespace and punctuation tokenizer for docstrings."""
    tokens = re.findall(r"\b\w+\b|[^\s\w]", text)
    return tokens

def tokenize_code(code):
    """Tokenize Python code using the standard library tokenize module."""
    tokens = []
    try:
        reader = io.StringIO(code).readline
        for toknum, tokval, _, _, _ in tokenize.generate_tokens(reader):
            if toknum == tokenize.ENDMARKER:
                break
            if toknum == tokenize.INDENT or toknum == tokenize.DEDENT or toknum == tokenize.NEWLINE or toknum == tokenize.NL:
                continue
            tokens.append(tokval)
    except tokenize.TokenError:
        # Fallback: simple split
        tokens = code.strip().split()
    return tokens

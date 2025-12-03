import os
import json
import torch
import math
import pickle
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# ---------- I/O helpers ----------
def save_checkpoint(state, filename="checkpoint.pth"):
    torch.save(state, filename)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# ---------- Masks ----------
def make_pad_mask(tgt, pad_idx=0):
    # tgt: [batch, seq_len] -> returns bool mask where True = PAD
    return (tgt == pad_idx)  # dtype: bool

def subsequent_mask(size, device=None):
    # causal mask for transformer: shape [size, size], float mask with -inf on upper triangle
    mask = torch.triu(torch.ones(size, size, dtype=torch.float32, device=device) * float("-inf"), diagonal=1)
    return mask

# ---------- Decode helpers ----------
def tokens_to_sentence(tokens, idx2word, stop_token=None):
    """
    tokens: list of ints (without start token usually)
    idx2word: dict mapping str(index)->word OR int->word
    """
    words = []
    for t in tokens:
        # idx2word keys might be strings if saved as json
        w = idx2word.get(str(t), idx2word.get(int(t), "<unk>"))
        if stop_token is not None and w == stop_token:
            break
        words.append(w)
    return " ".join(words)

# ---------- Greedy decode (step-by-step) ----------
@torch.no_grad()
def greedy_decode(model, images, idx2word, start_idx, end_idx, max_len=20, device="cpu"):
    """
    model: EncoderDecoderCaptionModel
    images: tensor [batch, 3, 224, 224]
    returns: list of token lists (predictions) and list of strings
    """
    model.eval()
    images = images.to(device)
    batch_size = images.size(0)

    # Encode once
    encoder_out = model.encoder(images)  # [batch, embed_dim]

    # initialize sequences with <start>
    seqs = torch.full((batch_size, 1), start_idx, dtype=torch.long, device=device)  # [batch, 1]

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for t in range(max_len):
        # forward decoder on current sequences
        logits = model.decoder(tgt_sequences=seqs, encoder_out=encoder_out, tgt_pad_mask=None)  # [batch, seq_len, vocab]
        next_logits = logits[:, -1, :]  # [batch, vocab]
        next_tokens = torch.argmax(next_logits, dim=-1).unsqueeze(1)  # [batch, 1]

        seqs = torch.cat([seqs, next_tokens], dim=1)  # append

        # mark finished sequences
        finished = finished | (next_tokens.squeeze(1) == end_idx)
        if finished.all():
            break

    results = seqs[:, 1:].cpu().tolist()  # drop initial <start>
    sentences = [ tokens_to_sentence(r, idx2word, stop_token=None) for r in results ]
    return results, sentences

# ---------- BLEU ----------
def compute_bleu(references, hypotheses):
    """
    references: list of list of reference token lists (i.e., [[ref1_tokens],[ref2_tokens],...])
    hypotheses: list of hypothesis token lists
    Returns BLEU-4 corpus score.
    """
    # smoothing
    chencherry = SmoothingFunction()
    # corpus_bleu expects references as list of list of tokens (words)
    score = corpus_bleu(references, hypotheses, smoothing_function=chencherry.method4)
    return score

import time
from tqdm import tqdm
import torch
import torch.nn as nn
from utils import save_checkpoint, greedy_decode, compute_bleu, tokens_to_sentence
from collections import defaultdict
import numpy as np

def train_one_epoch(model, dataloader, optimizer, device, epoch,
                    pad_idx=0, clip_grad=1.0, scheduler=None, print_every=200):
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    progress = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch} [Train]")

    for batch_idx, (images, captions) in progress:
        images = images.to(device)
        captions = captions.to(device)

        inputs = captions[:, :-1]
        targets = captions[:, 1:]

        optimizer.zero_grad()
        outputs = model(images, inputs)

        vocab_size = outputs.size(-1)
        loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)

        progress.set_postfix({"loss": avg_loss})

    return total_loss / len(dataloader)

def validate(model, dataloader, device, idx2word, start_idx, end_idx,
             pad_idx=0, max_len=20, num_samples=None):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    total_loss = 0.0

    references = []
    hypotheses = []

    progress = tqdm(enumerate(dataloader), total=len(dataloader), desc="Validating")

    with torch.no_grad():
        for batch_idx, (images, captions) in progress:
            images = images.to(device)
            captions = captions.to(device)

            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            outputs = model(images, inputs)
            vocab_size = outputs.size(-1)

            loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
            total_loss += loss.item()

            preds_tokens, _ = greedy_decode(
                model, images, idx2word, start_idx, end_idx,
                max_len=max_len, device=device
            )

            caps_list = captions.cpu().tolist()
            for i in range(images.size(0)):
                ref = [[str(tok) for tok in caps_list[i][1:-1] if tok != pad_idx]]
                hyp = [str(tok) for tok in preds_tokens[i] if tok != end_idx]
                references.append(ref)
                hypotheses.append(hyp)

            avg_loss = total_loss / (batch_idx + 1)
            progress.set_postfix({"val_loss": avg_loss})

            if num_samples is not None and len(hypotheses) >= num_samples:
                break

    bleu_score = compute_bleu(references, hypotheses)
    return total_loss / len(dataloader), bleu_score

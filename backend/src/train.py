import os
import json
import time
import torch
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from model import EncoderDecoderCaptionModel
from dataset import FlickrDataset, collate_fn
from utils import load_json, load_pkl, save_checkpoint
from engine import train_one_epoch, validate

def main(args):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    print("Device:", device)

    # Load vocab
    word2idx = load_json(args.word2idx)
    idx2word = load_json(args.idx2word)
    vocab_size = len(word2idx)
    pad_idx = word2idx.get("<pad>", 0)
    start_idx = word2idx.get("<start>", 1)
    end_idx = word2idx.get("<end>", 2)

    # Dataset & Dataloader
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = FlickrDataset(
        img_folder=args.img_folder,
        captions_file=args.captions_pkl,
        transform=transform
    )
    # split into train/val (simple random split)
    total = len(dataset.data)
    indices = list(range(total))
    split = int(0.9 * total)
    train_idx = indices[:split]
    val_idx = indices[split:]

    from torch.utils.data import Subset
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # Model
    model = EncoderDecoderCaptionModel(
        embed_dim=args.embed_dim,
        vocab_size=vocab_size,
        decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        train_cnn=args.train_cnn
    ).to(device)

    # Optimizer & Scheduler
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    best_bleu = 0.0
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    start_epoch = 1
    if args.resume is not None:
        print(f"Loading checkpoint from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch,
                                     pad_idx=pad_idx, clip_grad=args.clip_grad, scheduler=None, print_every=200)
        val_loss, val_bleu = validate(model, val_loader, device, idx2word, start_idx, end_idx,
                                      pad_idx=pad_idx, max_len=args.max_len, num_samples=args.num_val_samples)

        elapsed = time.time() - start
        print(f"Epoch {epoch} done in {elapsed:.1f}s — train_loss: {train_loss:.4f} — val_loss: {val_loss:.4f} — val_bleu: {val_bleu:.4f}")

        # save checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "word2idx": word2idx,
            "idx2word": idx2word,
            "vocab_size": vocab_size
        }
        save_checkpoint(ckpt, os.path.join(args.checkpoint_dir, f"ckpt_epoch{epoch}.pth"))

        # track best by BLEU
        if val_bleu > best_bleu:
            best_bleu = val_bleu
            save_checkpoint(ckpt, os.path.join(args.checkpoint_dir, "best_model.pth"))
            print("New best BLEU — model saved.")

    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_folder", type=str, default="data/Images", help="path to images")
    parser.add_argument("--captions_pkl", type=str, default="data/numericalized_captions.pkl")
    parser.add_argument("--word2idx", type=str, default="data/word2idx.json")
    parser.add_argument("--idx2word", type=str, default="data/idx2word.json")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--decoder_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--ff_dim", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--num_val_samples", type=int, default=1000)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--train_cnn", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--resume", type=str, default="checkpoints/best_model.pth", help="Path to checkpoint to resume from")
    args = parser.parse_args()
    main(args)

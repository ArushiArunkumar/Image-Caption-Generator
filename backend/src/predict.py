# src/predict.py
import argparse
import torch
import json
from torchvision import transforms
from PIL import Image
from model import EncoderDecoderCaptionModel
from utils import load_json, greedy_decode
from utils import greedy_decode, beam_search_decode

def load_model(checkpoint, device, vocab_size, embed_dim, decoder_layers=3, num_heads=8, ff_dim=2048, train_cnn=False):
    model = EncoderDecoderCaptionModel(
        embed_dim=embed_dim,
        vocab_size=vocab_size,
        decoder_layers=decoder_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        train_cnn=train_cnn
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--img", required=True, help="single image file or folder")
    parser.add_argument("--word2idx", default="data/word2idx.json")
    parser.add_argument("--idx2word", default="data/idx2word.json")
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument("--beam_size", type=int, default=1, help="Beam width for beam search. 1 = greedy.")
    args = parser.parse_args()

    # device selection
    if args.device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    word2idx = json.load(open(args.word2idx))
    idx2word = json.load(open(args.idx2word))
    vocab_size = len(word2idx)
    start_idx = word2idx["<start>"]
    end_idx = word2idx["<end>"]

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    model = load_model(args.checkpoint, device, vocab_size=vocab_size, embed_dim=args.embed_dim)

    # single image or folder
    import os
    imgs = []
    if os.path.isdir(args.img):
        for f in sorted(os.listdir(args.img)):
            if f.lower().endswith((".jpg",".jpeg",".png")):
                imgs.append(os.path.join(args.img,f))
    else:
        imgs = [args.img]

    for path in imgs:
        im = Image.open(path).convert("RGB")
        tensor = transform(im).unsqueeze(0).to(device)

        if args.beam_size == 1:
            # GREEDY decoding
            tokens, sentences = greedy_decode(
                model, tensor, idx2word, start_idx, end_idx,
                max_len=args.max_len, device=device
            )
            print("IMAGE:", path)
            print("DECODE (GREEDY):", sentences[0])
            print("TOKENS:", tokens[0][:30])

        else:
            # BEAM SEARCH decoding
            tokens, sentences = beam_search_decode(
                model, tensor, idx2word, start_idx, end_idx,
                beam_size=args.beam_size,
                max_len=args.max_len,
                device=device
            )
            print("IMAGE:", path)
            print(f"DECODE (BEAM={args.beam_size}):", sentences[0])
            print("TOKENS:", tokens[0][:30])

        print("-----")


    

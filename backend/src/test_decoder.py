import torch
from model_decoder import TransformerCaptionDecoder

vocab_size = 3000
embed_dim = 512
batch = 4
seq_len = 10

decoder = TransformerCaptionDecoder(
    embed_dim=embed_dim,
    vocab_size=vocab_size,
    num_heads=8,
    num_layers=3
)

# Dummy inputs
tgt = torch.randint(0, vocab_size, (batch, seq_len))
encoder_out = torch.randn(batch, embed_dim)

logits = decoder(tgt, encoder_out)

print("Decoder output shape:", logits.shape)

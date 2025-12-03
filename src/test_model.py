import torch
from model import EncoderDecoderCaptionModel

batch = 4
seq_len = 12
vocab_size = 3000
embed_dim = 512

model = EncoderDecoderCaptionModel(
    embed_dim=embed_dim,
    vocab_size=vocab_size
)

# Dummy image batch
images = torch.randn(batch, 3, 224, 224)

# Dummy caption batch
captions = torch.randint(0, vocab_size, (batch, seq_len))

# Forward pass
out = model(images, captions)

print("Model output shape:", out.shape)

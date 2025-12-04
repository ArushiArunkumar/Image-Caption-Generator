import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # shape: [1, max_len, embed_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch, seq_len, embed_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerCaptionDecoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        vocab_size,
        num_heads=8,
        num_layers=3,
        ff_dim=2048,
        dropout=0.1
    ):
        super(TransformerCaptionDecoder, self).__init__()

        # Word embeddings
        self.word_embed = nn.Embedding(vocab_size, embed_dim)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True  # IMPORTANT: allows use of [batch, seq, embed]
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        # Final linear layer → vocab distribution
        self.fc_out = nn.Linear(embed_dim, vocab_size)

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

    def generate_square_subsequent_mask(self, size):
        """
        Create a causal mask (prevent decoder from peeking ahead).
        Shape: [size, size]
        """
        mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
        return mask

    def forward(self, tgt_sequences, encoder_out, tgt_pad_mask=None):
        """
        tgt_sequences: [batch, tgt_len]
        encoder_out:   [batch, embed_dim]
        """

        batch_size, tgt_len = tgt_sequences.size()

        # Embed words
        tgt_emb = self.word_embed(tgt_sequences)  # [batch, tgt_len, embed_dim]

        # Add positional encoding
        tgt_emb = self.positional_encoding(tgt_emb)

        # Expand encoder_out to sequence length 1
        encoder_out = encoder_out.unsqueeze(1)  # [batch, 1, embed_dim]

        # Masks
        causal_mask = self.generate_square_subsequent_mask(tgt_len).to(tgt_sequences.device)
        # Padding mask: True = ignore token
        # tgt_pad_mask shape = [batch, tgt_len]

        # Decoder output
        out = self.transformer_decoder(
            tgt=tgt_emb,
            memory=encoder_out,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_pad_mask
        )

        # Output logits
        logits = self.fc_out(out)  # [batch, tgt_len, vocab_size]

        return logits

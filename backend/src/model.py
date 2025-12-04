import torch
import torch.nn as nn

from src.model_encoder import EncoderCNN
from src.model_decoder import TransformerCaptionDecoder


class EncoderDecoderCaptionModel(nn.Module):
    def __init__(
        self,
        embed_dim,
        vocab_size,
        decoder_layers=3,
        num_heads=8,
        ff_dim=2048,
        train_cnn=False
    ):
        super(EncoderDecoderCaptionModel, self).__init__()

        self.encoder = EncoderCNN(embed_dim=embed_dim, train_cnn=train_cnn)

        self.decoder = TransformerCaptionDecoder(
            embed_dim=embed_dim,
            vocab_size=vocab_size,
            num_heads=num_heads,
            num_layers=decoder_layers,
            ff_dim=ff_dim
        )

    def create_padding_mask(self, tgt, pad_idx=0):
        """
        Returns mask where padded tokens are True (ignored by attention)
        tgt: [batch, seq_len]
        """
        return (tgt == pad_idx)

    def forward(self, images, captions):
        """
        images:   [batch, 3, 224, 224]
        captions: [batch, seq_len]  (includes <start> token)
        """

        # Encoder output
        encoder_out = self.encoder(images)   # [batch, embed_dim]

        # Create padding mask
        tgt_pad_mask = self.create_padding_mask(captions)  # [batch, seq_len]

        # Decoder forward
        outputs = self.decoder(
            tgt_sequences=captions,
            encoder_out=encoder_out,
            tgt_pad_mask=tgt_pad_mask
        )

        return outputs

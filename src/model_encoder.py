import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_dim=512, train_cnn=False):
        super(EncoderCNN, self).__init__()

        # Load pretrained ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Remove final classification layer
        modules = list(resnet.children())[:-1]  # remove fc
        self.resnet = nn.Sequential(*modules)

        # Freeze CNN layers unless training is desired
        for param in self.resnet.parameters():
            param.requires_grad = train_cnn

        # Projection layer: 2048 -> embed_dim
        self.embed = nn.Linear(2048, embed_dim)

    def forward(self, images):
        """
        images: [batch, 3, 224, 224]
        returns: [batch, embed_dim]
        """
        with torch.set_grad_enabled(self.resnet[0].weight.requires_grad):
            features = self.resnet(images)

        features = features.reshape(features.size(0), -1)  # [batch, 2048]
        features = self.embed(features)  # [batch, embed_dim]

        return features

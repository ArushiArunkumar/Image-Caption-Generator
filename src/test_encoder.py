import torch
from model_encoder import EncoderCNN
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load one test image
img = Image.open("data/Images/1000268201_693b08cb0e.jpg").convert("RGB")
img = transform(img).unsqueeze(0)

encoder = EncoderCNN(embed_dim=512)

with torch.no_grad():
    features = encoder(img)

print("Encoder output shape:", features.shape)

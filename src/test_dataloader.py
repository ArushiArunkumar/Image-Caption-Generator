from torch.utils.data import DataLoader
from dataset import FlickrDataset, collate_fn
from preprocessing import Vocabulary
import torchvision.transforms as T

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = FlickrDataset(
    img_folder="data/Images",
    captions_file="data/numericalized_captions.pkl",
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)

for images, captions in loader:
    print("Images shape:", images.shape)
    print("Captions shape:", captions.shape)
    break

import os
import torch
import json
import pickle
from PIL import Image
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class FlickrDataset(Dataset):
    def __init__(self, img_folder, captions_file, transform=None):
        """
        img_folder: path to Images/
        captions_file: numericalized_captions.pkl
        """
        self.img_folder = img_folder
        self.transform = transform

        # Load numericalized captions
        with open(captions_file, "rb") as f:
            self.captions = pickle.load(f)

        # Create list of (image_name, caption) pairs
        self.data = []
        for img, caps in self.captions.items():
            for c in caps:
                self.data.append((img, c))

        print(f"Total pairs: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, caption = self.data[idx]

        img_path = os.path.join(self.img_folder, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        caption = torch.tensor(caption, dtype=torch.long)

        return image, caption

def collate_fn(batch):
    images = []
    captions = []

    for img, cap in batch:
        images.append(img)
        captions.append(cap)

    images = torch.stack(images)
    captions = pad_sequence(captions, batch_first=True, padding_value=0)  # <pad> token = 0

    return images, captions

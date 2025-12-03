import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from model import EncoderDecoderCaptionModel
from utils import load_json, greedy_decode, beam_search_decode

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load vocab
word2idx = load_json("data/word2idx.json")
idx2word = load_json("data/idx2word.json")
start_idx = word2idx["<start>"]
end_idx = word2idx["<end>"]
vocab_size = len(word2idx)

# Load model
ckpt = torch.load("checkpoints/best_model.pth", map_location=device)
model = EncoderDecoderCaptionModel(embed_dim=512, vocab_size=vocab_size)
model.load_state_dict(ckpt["model_state"])
model.eval().to(device)

# Transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])


def annotate(image_path, output_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    greedy_tokens, greedy_caption = greedy_decode(
        model, tensor, idx2word, start_idx, end_idx)

    beam_tokens, beam_caption = beam_search_decode(
        model, tensor, idx2word, start_idx, end_idx, beam_size=5)

    # Draw
    result = img.copy()
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype("Arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    y = result.height - 140
    draw.rectangle([(0, y-10), (result.width, result.height)], fill=(0,0,0,150))

    draw.text((20, y), f"Greedy: {greedy_caption[0]}", fill="white", font=font)
    draw.text((20, y+45), f"Beam Search: {beam_caption[0]}", fill="white", font=font)

    result.save(output_path)
    print(f"Saved {output_path}")


# Examples (place your chosen sample images in docs/assets/)
annotate("docs/assets/sample1.jpg", "docs/assets/sample1_output.jpg")
annotate("docs/assets/sample2.jpg", "docs/assets/sample2_output.jpg")
annotate("docs/assets/sample3.jpg", "docs/assets/sample3_output.jpg")

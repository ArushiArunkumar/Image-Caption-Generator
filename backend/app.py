from flask import Flask, request, jsonify
import torch
from PIL import Image
from torchvision import transforms
import io, json

from model import EncoderDecoderCaptionModel
from utils import load_json, greedy_decode, beam_search_decode

app = Flask(__name__)

device = torch.device("cpu")

# Load vocab
word2idx = json.load(open("model/word2idx.json"))
idx2word = json.load(open("model/idx2word.json"))
start_idx = word2idx["<start>"]
end_idx = word2idx["<end>"]
vocab_size = len(word2idx)

# Load model
ckpt = torch.load("model/best_model.pth", map_location=device)
model = EncoderDecoderCaptionModel(embed_dim=512, vocab_size=vocab_size)
model.load_state_dict(ckpt["model_state"])
model.eval().to(device)

# Image transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

@app.route("/caption", methods=["POST"])
def caption():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    img_bytes = request.files["image"].read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    # Beam search by default
    tokens, sentences = beam_search_decode(
        model,
        tensor,
        idx2word,
        start_idx,
        end_idx,
        beam_size=5,
        max_len=20,
        device=device
    )

    return jsonify({"caption": sentences[0]})

@app.route("/")
def home():
    return "Image Captioning API is running."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

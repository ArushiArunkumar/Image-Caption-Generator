from PIL import Image, ImageDraw, ImageFont

def create_diagram(filename, title, components):
    img = Image.new("RGB", (1200, 700), "white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("Arial.ttf", 28)
        small_font = ImageFont.truetype("Arial.ttf", 22)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    draw.text((20,20), title, fill="black", font=font)

    x = 50
    y = 120

    for comp in components:
        draw.rectangle([x, y, x+300, y+120], outline="black", width=3)
        draw.text((x+20, y+40), comp, fill="black", font=small_font)
        x += 350

    img.save(filename)
    print(f"Saved {filename}")

# ====== Encoder Diagram ======
create_diagram(
    "docs/assets/encoder.png",
    "Encoder: ResNet-50 Architecture",
    ["Input Image", "ResNet-50", "2048-D Vector", "Linear Projection (512-D)"]
)

# ====== Decoder Diagram ======
create_diagram(
    "docs/assets/decoder.png",
    "Transformer Decoder",
    ["Token Embedding", "Positional Encoding", "Multi-Head Self-Attention",
     "Cross-Attention", "Feed Forward Network", "Linear → Softmax"]
)

# ====== Full Pipeline ======
create_diagram(
    "docs/assets/pipeline.png",
    "End-to-End Image Captioning Pipeline",
    ["Image", "CNN Encoder (ResNet50)", "Embedding Vector",
     "Transformer Decoder", "Generated Caption"]
)

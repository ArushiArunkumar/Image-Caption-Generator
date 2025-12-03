import matplotlib.pyplot as plt

# ====== INSERT YOUR COLLECTED VALUES HERE ======
# Example values based on your logs

train_loss = [3.48, 2.81, 2.56, 2.38, 2.25, 2.19, 2.08, 1.99, 1.89, 1.81]
val_loss =   [0.73, 0.68, 0.67, 0.66, 0.66, 0.67, 0.67, 0.67, 0.68, 0.69]
val_bleu =   [0.049, 0.053, 0.047, 0.056, 0.050, 0.058, 0.057, 0.056, 0.056, 0.051]

epochs = range(1, len(train_loss) + 1)

# ====== Training Loss ======
plt.figure(figsize=(8,5))
plt.plot(epochs, train_loss, marker='o')
plt.title("Training Loss Across Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("docs/assets/train_loss.png", dpi=150)
plt.close()

# ====== Validation Loss ======
plt.figure(figsize=(8,5))
plt.plot(epochs, val_loss, marker='o', color='orange')
plt.title("Validation Loss Across Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("docs/assets/val_loss.png", dpi=150)
plt.close()

# ====== BLEU Score ======
plt.figure(figsize=(8,5))
plt.plot(epochs, val_bleu, marker='o', color='green')
plt.title("BLEU Score Across Epochs")
plt.xlabel("Epoch")
plt.ylabel("BLEU Score")
plt.grid(True)
plt.savefig("docs/assets/bleu_scores.png", dpi=150)
plt.close()

print("Plots saved as train_loss.png, val_loss.png, bleu_scores.png")

import os
import re
import json
import csv
import nltk
import pickle
from collections import Counter
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        
        self.word2idx = {
            "<pad>": 0,
            "<start>": 1,
            "<end>": 2,
            "<unk>": 3
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_freq = Counter()

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z]+", " ", text)
        text = text.strip()
        return text

    def build_vocabulary(self, sentence_list):
        for sentence in sentence_list:
            cleaned = self.clean_text(sentence)
            tokens = word_tokenize(cleaned)
            self.word_freq.update(tokens)

        # Only keep frequent words
        for word, freq in self.word_freq.items():
            if freq >= self.freq_threshold:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def numericalize(self, text):
        cleaned = self.clean_text(text)
        tokens = word_tokenize(cleaned)

        numericalized = []
        for token in tokens:
            if token in self.word2idx:
                numericalized.append(self.word2idx[token])
            else:
                numericalized.append(self.word2idx["<unk>"])
        return numericalized


def load_captions(caption_file):
    image_to_captions = {}

    with open(caption_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header: image,caption

        for row in reader:
            if len(row) < 2:
                continue

            img_id = row[0].strip()
            caption = row[1].strip()

            if img_id not in image_to_captions:
                image_to_captions[img_id] = []

            image_to_captions[img_id].append(caption)

    return image_to_captions


def preprocess_and_save(caption_file, output_dir="data/"):

    # Load dataset
    img_caps = load_captions(caption_file)

    all_captions = []
    for caps in img_caps.values():
        all_captions.extend(caps)

    # Build vocabulary
    vocab = Vocabulary(freq_threshold=5)
    vocab.build_vocabulary(all_captions)

    # Numericalize captions
    numericalized_data = {}

    for img, caps in img_caps.items():
        num_caps = []
        for c in caps:
            tokens = [vocab.word2idx["<start>"]]
            tokens += vocab.numericalize(c)
            tokens.append(vocab.word2idx["<end>"])
            num_caps.append(tokens)

        numericalized_data[img] = num_caps

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "numericalized_captions.pkl"), "wb") as f:
        pickle.dump(numericalized_data, f)

    with open(os.path.join(output_dir, "word2idx.json"), "w") as f:
        json.dump(vocab.word2idx, f)

    with open(os.path.join(output_dir, "idx2word.json"), "w") as f:
        json.dump(vocab.idx2word, f)

    print("Preprocessing Complete!")
    print(f"Vocabulary size: {len(vocab.word2idx)}")


if __name__ == "__main__":
    preprocess_and_save("data/captions.txt")

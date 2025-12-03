# Image Caption Generation Using CNN–Transformer Architecture

## Overview

This project implements an end-to-end **image captioning system** that generates natural language descriptions from input images. The model uses a **ResNet-50 convolutional encoder** and a **Transformer-based decoder**, combining visual feature extraction with advanced sequence modeling. The system is trained and evaluated on the **Flickr8k** dataset.

The project demonstrates core competencies in:

* Computer vision and visual feature extraction
* Neural sequence modeling
* Transformer architectures
* Data preprocessing and vocabulary construction
* Training and evaluation of deep learning models
* Quantitative and qualitative analysis
* Efficient GPU-based model development (including Apple Silicon / MPS)

This repository is suitable for academic research use and demonstrates strong preparedness for research roles in deep learning, vision, and NLP.

---

## Motivation

Image caption generation is a challenging multimodal problem requiring a model to jointly understand:

* High-dimensional visual information
* Linguistic structure
* Temporal dependencies in natural language
* Cross-attention between image features and generated tokens

The Transformer decoder has proven highly effective for sequence tasks, while CNN encoders provide strong visual representations. Combining these components offers a high-performing architecture capable of generating coherent captions.

---

## Architecture

### Encoder: ResNet-50

* Pretrained on ImageNet (using `ResNet50_Weights.IMAGENET1K_V2`)
* Final classification layer removed
* Global average pooled 2048-dimensional feature vector
* Linear projection to a 512-dimensional embedding space
* Optionally trainable (frozen by default for stability)

### Decoder: Transformer

* 3-layer Transformer decoder
* 8 attention heads
* Feed-forward dimension: 2048
* Learned word embeddings (512-dimensional)
* Sinusoidal positional encodings
* Causal attention mask to enforce autoregressive generation
* Cross-attention over encoder image embedding

### Caption Generation

Two decoding strategies are supported:

1. **Greedy Decoding**
2. **Beam Search Decoding** (configurable beam width)

Beam search significantly improves caption quality and is recommended for evaluation.

---

## Dataset

### Flickr8k Dataset

* ~8,000 images
* 5 human-annotated captions per image
* Standard benchmark for compact captioning experiments
* File structure:

  * `Images/` (image files)
  * `captions.txt` (image,caption CSV format)

### Preprocessing

* Text cleaning (lowercasing, removal of non-alphabetic characters)
* Tokenization using NLTK
* Vocabulary pruning using frequency threshold
* Numericalization of captions with `<start>`, `<end>`, `<pad>`, and `<unk>` tokens
* Saved artifacts:

  * `word2idx.json`
  * `idx2word.json`
  * `numericalized_captions.pkl`

---

## Repository Structure

```
Image-Caption-Generator/
│
├── src/
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── model_encoder.py
│   ├── model_decoder.py
│   ├── model.py
│   ├── engine.py
│   ├── utils.py
│   ├── train.py
│   └── predict.py
│
├── checkpoints/
├── data/
└── README.md
```

### Key Components

* **preprocessing.py**: Vocabulary construction, caption numericalization
* **dataset.py**: Dataset class and collate function for variable-length sequences
* **model_encoder.py**: ResNet-50 encoder
* **model_decoder.py**: Transformer decoder with positional encodings
* **model.py**: Combined encoder–decoder model
* **engine.py**: Training and validation loops
* **utils.py**: Masking utilities, greedy/beam decoding, BLEU computation
* **predict.py**: Inference script supporting greedy and beam search decoding

This modular design ensures reproducibility, extensibility, and clear separation between data, modeling, and evaluation.

---

## Training

### Requirements

* Python 3.10+
* PyTorch with MPS/CUDA support
* Torchvision
* NLTK
* TQDM

### Training Command

```
python3 src/train.py --batch_size 16 --epochs 10
```

### Resuming Training

The training pipeline supports resumption from saved checkpoints:

```
python3 src/train.py --resume checkpoints/best_model.pth
```

---

## Evaluation

### Quantitative Metrics

The model is evaluated using the standard **BLEU** score (corpus BLEU with smoothing).
Two evaluation modes are supported:

1. **Greedy decoding**
2. **Beam search decoding**

Due to Flickr8k having five captions per image, multi-reference BLEU is recommended for research-grade evaluation.

### Qualitative Evaluation

Generated captions can be inspected using:

```
python3 src/predict.py --checkpoint checkpoints/best_model.pth --img path/to/image.jpg --beam_size 5
```

The system produces coherent, semantically grounded captions such as:

* “a girl in a pink dress is standing in front of a wooden fence”

These outputs demonstrate strong alignment between image content and linguistic structure.

---

## Experimental Results

### Training Dynamics

* Training loss consistently decreases across epochs
* Validation loss remains stable due to teacher forcing
* BLEU scores show incremental improvements
* Beam search yields significantly higher-quality captions compared to greedy decoding

### Model Behavior

The model successfully learns:

* Object recognition (people, animals, objects)
* Colors and clothing attributes
* Spatial relations (“in front of”, “on”, “next to”)
* Basic actions (“standing”, “playing”, “running”)

---

## Beam Search Implementation

Beam search is integrated in `utils.py` and can be invoked via:

```
--beam_size k
```

A typical beam width of 3–5 provides substantial qualitative improvement without significant computational overhead.

---

## Strengths of This Implementation

* End-to-end CNN–Transformer pipeline
* Scalable and extensible architecture
* Resume training capability
* Efficient use of MPS acceleration for Apple Silicon
* Clean, well-structured code suitable for further research
* Supports both greedy and beam decoding strategies
* Training, validation, and inference are fully modular

This makes the repository suitable for:

* Research Assistant roles
* Graduate coursework submissions
* Demonstrations of deep learning capability
* Extensions into attention visualization, fine-tuning, and larger datasets (e.g., MS COCO)

---

## Future Work

Several extensions are natural directions for research:

### Vision Enhancements

* Spatial feature extraction (use of encoder feature maps instead of global pooling)
* Multi-head cross-attention over 7×7 CNN activations
* Incorporation of ViT or Swin Transformer encoders

### Language Modeling

* Larger Transformer decoders
* Better subword tokenization (Byte-Pair Encoding or SentencePiece)
* Length normalization and coverage penalties in beam search

### Optimization

* Mixed precision training
* Differential learning rates for encoder and decoder
* Scheduled sampling

### Dataset Scaling

* Training on MS COCO for higher linguistic variance
* Zero-shot captioning via pretrained vision–language models

These improvements can significantly increase model performance and research value.

---

## Conclusion

This project presents a robust, research-grade implementation of an image captioning system based on a CNN–Transformer architecture. It demonstrates key competencies required for research in multimodal deep learning, including model design, preprocessing, training engineering, evaluation, and decoding algorithms.

The repository provides a solid foundation for further academic work in computer vision, natural language processing, and multimodal AI.

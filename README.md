# Emotion Detection in Tweets using NLP and TensorFlow

This project focuses on building a machine learning model to detect emotions in short text sentences (tweets) using Natural Language Processing (NLP) techniques and TensorFlow/Keras.

The dataset used is the publicly available [`emotion` dataset](https://huggingface.co/datasets/dair-ai/emotion) from Hugging Face, which contains labeled tweets across six emotion categories.

---

## 🚀 Project Overview

- **Goal:** Classify tweets into one of six emotions:
  - `sadness`, `joy`, `love`, `anger`, `fear`, `surprise`
- **Tech Stack:** 
  - Python, TensorFlow, Keras, Hugging Face `datasets`, scikit-learn, matplotlib
- **Main Steps:**
  1. Load and preprocess the dataset
  2. Tokenize and pad text sequences
  3. Build and train a neural network model
  4. Evaluate performance and visualize results (confusion matrix)
  5. Predict emotions for new sentences

---

## 📂 Dataset

- **Source:** Hugging Face `emotion` dataset
- **Structure:** 
  - Short texts (tweets) labeled with one of six emotions
  - Split into `train`, `validation`, and `test` sets

```python
from datasets import load_dataset
dataset = load_dataset('emotion')

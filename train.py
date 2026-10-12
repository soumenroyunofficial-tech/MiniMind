
import json
import re
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "intents.json"
MODEL_FILE = BASE_DIR / "model.npz"
META_FILE = BASE_DIR / "model_metadata.json"

# Reproducible random initialization
np.random.seed(42)


import unicodedata
import re

def tokenize(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    return re.findall(r"[a-z0-9']+", text)


def make_bow(tokens, vocabulary):
    """Create a binary bag-of-words vector."""
    vector = np.zeros(len(vocabulary), dtype=np.float64)
    for token in set(tokens):
        if token in vocabulary:
            vector[vocabulary[token]] = 1.0
    return vector


def softmax(logits):
    """Convert output scores into probabilities."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def main():
    with open(DATA_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    intents = data["intents"]
    labels = [item["tag"] for item in intents]
    label_to_id = {label: i for i, label in enumerate(labels)}

    # Build vocabulary and training examples
    all_tokens = set()
    samples = []

    for intent in intents:
        for pattern in intent["patterns"]:
            tokens = tokenize(pattern)
            all_tokens.update(tokens)
            samples.append((tokens, label_to_id[intent["tag"]]))

    vocabulary = {
        word: i for i, word in enumerate(sorted(all_tokens))
    }

    X = np.array([
        make_bow(tokens, vocabulary)
        for tokens, _ in samples
    ])

    y = np.array([label for _, label in samples], dtype=np.int64)

    # One-hot encoded target classes
    Y = np.eye(len(labels))[y]

    sample_count, input_size = X.shape
    hidden_size = 16
    output_size = len(labels)

    # Xavier-style random weight initialization
    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(
        2.0 / (input_size + hidden_size)
    )
    b1 = np.zeros((1, hidden_size))

    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(
        2.0 / (hidden_size + output_size)
    )
    b2 = np.zeros((1, output_size))

    learning_rate = 0.05
    epochs = 3000

    print("Training MiniMind AI from scratch...")
    print(f"Training examples: {sample_count}")
    print(f"Vocabulary size: {input_size}")
    print(f"Hidden neurons: {hidden_size}")
    print(f"Intent classes: {output_size}")

    for epoch in range(epochs):
        # Forward propagation
        z1 = X @ W1 + b1
        a1 = np.tanh(z1)

        logits = a1 @ W2 + b2
        probabilities = softmax(logits)

        # Cross-entropy loss
        eps = 1e-12
        loss = -np.mean(
            np.sum(Y * np.log(probabilities + eps), axis=1)
        )

        # Backpropagation through softmax and cross-entropy
        d_logits = (probabilities - Y) / sample_count

        dW2 = a1.T @ d_logits
        db2 = np.sum(d_logits, axis=0, keepdims=True)

        d_a1 = d_logits @ W2.T
        d_z1 = d_a1 * (1.0 - a1 ** 2)

        dW1 = X.T @ d_z1
        db1 = np.sum(d_z1, axis=0, keepdims=True)

        # Gradient descent update
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        if (epoch + 1) % 300 == 0:
            predictions = np.argmax(probabilities, axis=1)
            accuracy = np.mean(predictions == y) * 100
            print(
                f"Epoch {epoch + 1:4d}/{epochs} | "
                f"Loss: {loss:.4f} | "
                f"Training accuracy: {accuracy:.1f}%"
            )

    # Save only the learned model parameters
    np.savez(
        MODEL_FILE,
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2
    )

    # Save vocabulary, labels and response data separately
    metadata = {
        "vocabulary": vocabulary,
        "labels": labels,
        "intents": intents,
        "hidden_size": hidden_size
    }

    with open(META_FILE, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print("\nTraining complete!")
    print(f"Saved model: {MODEL_FILE.name}")
    print(f"Saved metadata: {META_FILE.name}")


if __name__ == "__main__":
    main()


import json
import random
import re
from pathlib import Path
from calculator import calculate
from memory import remember, recall, forget, list_memories
from document_search import search_documents, load_documents
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "model.npz"
META_FILE = BASE_DIR / "model_metadata.json"


import unicodedata
import re


def tokenize(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    return re.findall(r"[a-z0-9']+", text)


def make_bow(tokens, vocabulary):
    vector = np.zeros(len(vocabulary), dtype=np.float64)
    for token in set(tokens):
        if token in vocabulary:
            vector[vocabulary[token]] = 1.0
    return vector


def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def load_model():
    if not MODEL_FILE.exists() or not META_FILE.exists():
        raise FileNotFoundError(
            "Model files not found. Run train.py first."
        )

    parameters = np.load(MODEL_FILE)

    with open(META_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    model = {
        "W1": parameters["W1"],
        "b1": parameters["b1"],
        "W2": parameters["W2"],
        "b2": parameters["b2"]
    }

    return model, metadata


def predict(text, model, metadata):
    tokens = tokenize(text)
    vocabulary = metadata["vocabulary"]

    # Avoid making predictions on completely unknown input
    if not tokens or not any(word in vocabulary for word in tokens):
        return None, 0.0

    x = make_bow(tokens, vocabulary).reshape(1, -1)

    # Inference: forward propagation only
    a1 = np.tanh(x @ model["W1"] + model["b1"])
    probabilities = softmax(a1 @ model["W2"] + model["b2"])

    intent_id = int(np.argmax(probabilities[0]))
    confidence = float(probabilities[0, intent_id])

    return metadata["labels"][intent_id], confidence


def main():
    try:
        model, metadata = load_model()
    except FileNotFoundError as error:
        print(error)
        return

    print("=" * 44)
    print("       MiniMind AI - Offline Chat")
    print("=" * 44)
    print("Type 'exit' to quit.\n")

    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nMiniMind: Goodbye!")
            break

        if message.lower() in {"exit", "quit", "bye!"}:
            print("MiniMind: Goodbye!")
            break

        if not message:
            continue

        
        if message.lower().startswith("/calc "):
            expression = message[6:].strip()

            try:
                print("MiniMind:", calculate(expression))
            except (
                ValueError,
                SyntaxError,
                ZeroDivisionError,
                OverflowError
            ):
                print("MiniMind: I couldn't calculate that expression.")

            continue
        
        # OFFLINE DOCUMENT SEARCH

        if message.lower() == "/docs":
            documents = load_documents()

            if not documents:
                print(
                    "MiniMind: No readable documents found. "
                    "Add TXT or Markdown files to the documents folder."
                )
            else:
                sources = sorted({
                    item["source"] for item in documents
                })

                print(
                    f"MiniMind: Found {len(sources)} document(s) "
                    f"containing {len(documents)} passages."
                )

                for source in sources:
                    print(f"  - {source}")

            continue

        if message.lower().startswith("/ask "):
            question = message[5:].strip()

            if not question:
                print("MiniMind: Please enter a question after /ask.")
                continue

            results = search_documents(question, top_k=3)

            if not results:
                print(
                    "MiniMind: I couldn't find matching information "
                    "in your local documents."
                )
            else:
                print("MiniMind: Here are the most relevant passages:")

                for index, result in enumerate(results, start=1):
                    print(
                        f"\n[{index}] Source: {result['source']} "
                        f"(passage {result['chunk']})"
                    )
                    print(result["text"])

            continue

            
        # MEMORY COMMANDS

        # Remember a fact using natural language
        match = re.fullmatch(
            r"remember that (.+?) is (.+)",
            message,
            re.IGNORECASE
        )

        if match:
            key = match.group(1).strip()
            value = match.group(2).strip()

            if remember(key, value):
                print(f"MiniMind: I'll remember that {key} is {value}.")
            else:
                print("MiniMind: Please provide both a fact and its value.")
            continue

        # Recall a fact using natural language
        match = re.fullmatch(
            r"(?:what is|what's) (.+?)[?!.]*",
            message,
            re.IGNORECASE
        )

        if match:
            key = match.group(1).strip()
            value = recall(key)

            if value is not None:
                print(f"MiniMind: {key.capitalize()} is {value}.")
            else:
                print(f"MiniMind: I don't have a memory for '{key}'.")
            continue

        # Forget a saved fact
        match = re.fullmatch(
            r"forget (?:that )?(.+)",
            message,
            re.IGNORECASE
        )

        if match:
            key = match.group(1).strip()

            if forget(key):
                print(f"MiniMind: I've forgotten '{key}'.")
            else:
                print(f"MiniMind: I couldn't find '{key}' in memory.")
            continue

        # Display all saved facts
        if message.lower() in {"/memory", "show my memories"}:
            memories = list_memories()

            if memories:
                print("MiniMind: Here is what I remember:")
                for key, value in memories.items():
                    print(f"  - {key}: {value}")
            else:
                print("MiniMind: My memory is currently empty.")
            continue

        
        tag, confidence = predict(message, model, metadata)

        if tag is None or confidence < 0.55:
            print(
                "MiniMind: I'm not sure how to answer that yet. "
                "You can teach me by adding more training examples."
            )
            continue

        intent = next(
            item for item in metadata["intents"]
            if item["tag"] == tag
        )

        response = random.choice(intent["responses"])

        print(f"MiniMind [{tag}, {confidence:.0%}]: {response}")


if __name__ == "__main__":
    main()

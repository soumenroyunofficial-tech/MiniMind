
import json
import re
from pathlib import Path

MEMORY_FILE = Path(__file__).resolve().parent / "memory.json"


def normalize_key(key):
    """Normalize a memory key for consistent lookup."""
    key = re.sub(r"[^\w\s]", "", key.lower())
    return " ".join(key.split())


def load_memory():
    """Load saved memories, or return an empty dictionary."""
    if not MEMORY_FILE.exists():
        return {}

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, dict) and all(
            isinstance(k, str) and isinstance(v, str)
            for k, v in data.items()
        ):
            return data

        print("MiniMind: Memory file has an invalid format.")
        return {}

    except (json.JSONDecodeError, OSError):
        print("MiniMind: Could not read memory.json.")
        return {}


def save_memory(memory):
    """Save memories to the local disk."""
    with open(MEMORY_FILE, "w", encoding="utf-8") as file:
        json.dump(memory, file, ensure_ascii=False, indent=2)


def remember(key, value):
    """Store or update a memory."""
    key = normalize_key(key)
    value = value.strip()

    if not key or not value:
        return False

    memory = load_memory()
    memory[key] = value
    save_memory(memory)
    return True


def recall(key):
    """Retrieve a memory by its key."""
    memory = load_memory()
    key = normalize_key(key)
    return memory.get(key)


def forget(key):
    """Delete one memory."""
    memory = load_memory()
    key = normalize_key(key)

    if key not in memory:
        return False

    del memory[key]
    save_memory(memory)
    return True


def list_memories():
    """Return all saved memories."""
    return load_memory()

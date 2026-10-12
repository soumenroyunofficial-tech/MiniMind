
import json
from pathlib import Path

from chat import load_model, predict

BASE_DIR = Path(__file__).resolve().parent
TEST_FILE = BASE_DIR / "test_cases.json"


def main():
    model, metadata = load_model()

    with open(TEST_FILE, "r", encoding="utf-8") as file:
        tests = json.load(file)

    correct = 0

    for item in tests:
        predicted, confidence = predict(
            item["text"], model, metadata
        )

        expected = item["expected_intent"]
        passed = predicted == expected

        correct += int(passed)

        print(f"\nInput: {item['text']}")
        print(f"Expected: {expected}")
        print(f"Predicted: {predicted}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Result: {'PASS' if passed else 'FAIL'}")

    accuracy = correct / len(tests) if tests else 0.0

    print(f"\nTest accuracy: {accuracy:.2%}")
    print(f"Correct: {correct}/{len(tests)}")


if __name__ == "__main__":
    main()

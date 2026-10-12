
import re
from pathlib import Path

DOCUMENTS_DIR = Path(__file__).resolve().parent / "documents"

# Common words that usually contribute little to document search.
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "what",
    "which", "who", "how", "why", "when", "where", "can",
    "could", "would", "should", "do", "does", "did", "i",
    "me", "my", "you", "your", "it", "to", "of", "in",
    "on", "for", "and", "or", "with", "about", "tell",
    "please", "explain", "show", "find", "from", "that"
}


def tokenize(text):
    """Extract searchable words from text."""
    return re.findall(r"[a-z0-9]+", text.lower())


def split_into_chunks(text, max_words=120):
    """Split a document into manageable passages."""
    paragraphs = [
        p.strip() for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    chunks = []

    for paragraph in paragraphs:
        words = paragraph.split()

        # Split long paragraphs into smaller passages.
        for start in range(0, len(words), max_words):
            chunk = " ".join(words[start:start + max_words])
            if chunk:
                chunks.append(chunk)

    return chunks


def load_documents():
    """Read local TXT and Markdown files."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    documents = []

    supported_extensions = {".txt", ".md"}

    for path in sorted(DOCUMENTS_DIR.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in supported_extensions:
            continue

        try:
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as error:
            print(f"Skipping {path.name}: {error}")
            continue

        for number, chunk in enumerate(split_into_chunks(text), start=1):
            documents.append({
                "source": str(path.relative_to(DOCUMENTS_DIR)),
                "chunk": number,
                "text": chunk
            })

    return documents


def search_documents(query, top_k=3):
    """Rank local document passages by word overlap."""
    query_tokens = [
        word for word in tokenize(query)
        if word not in STOP_WORDS
    ]

    query_terms = set(query_tokens)

    if not query_terms:
        return []

    documents = load_documents()
    results = []

    for document in documents:
        doc_tokens = tokenize(document["text"])
        doc_terms = set(doc_tokens)

        matched_terms = query_terms & doc_terms

        if not matched_terms:
            continue

        # Reward matching distinct query terms and modestly
        # reward repeated occurrences.
        coverage = len(matched_terms) / len(query_terms)
        frequency = sum(
            min(doc_tokens.count(term), 3)
            for term in matched_terms
        )

        score = coverage + 0.05 * frequency

        results.append({
            **document,
            "score": score,
            "matched_terms": sorted(matched_terms)
        })

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:top_k]


def main():
    documents = load_documents()

    print(f"Documents indexed: {len(documents)} passages")
    print(f"Folder: {DOCUMENTS_DIR}")

    if not documents:
        print("Add TXT or Markdown files to the documents folder.")
        return

    while True:
        query = input("\nSearch (or 'exit'): ").strip()

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            continue

        results = search_documents(query)

        if not results:
            print("No matching passages found.")
            continue

        for result in results:
            print(
                f"\nSource: {result['source']} "
                f"(passage {result['chunk']})"
            )
            print(f"Matched words: {', '.join(result['matched_terms'])}")
            print(result["text"])


if __name__ == "__main__":
    main()

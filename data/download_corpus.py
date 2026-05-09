"""
Downloads a small English sample from HuggingFaceFW/finepdfs using streaming.
Saves each document as a .txt file in data/corpus/.
Stops when target chunk count is reached so cost stays predictable.
"""
import os
import sys
import re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from chunking.semantic_chunker import SemanticChunker

TARGET_CHUNKS   = 1200   # stop collecting documents once we estimate this many chunks
MIN_DOC_CHARS   = 2000   # skip documents that are too short to be meaningful
MAX_DOC_CHARS   = 40000  # skip extremely long documents (books, reports) to keep variety
CHUNK_SIZE      = 400    # same settings as rest of project

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")
os.makedirs(CORPUS_DIR, exist_ok=True)


def clean_text(text: str) -> str:
    """Basic cleaning: collapse whitespace, remove non-printable characters."""
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)  # keep only printable ASCII
    text = re.sub(r'\n{3,}', '\n\n', text)         # collapse excessive blank lines
    text = re.sub(r' {2,}', ' ', text)             # collapse multiple spaces
    return text.strip()


def estimate_chunks(text: str) -> int:
    return max(1, len(text) // CHUNK_SIZE)


def main():
    chunker   = SemanticChunker(max_chunk_size=CHUNK_SIZE, overlap=50)
    total_chunks = 0
    doc_count    = 0

    print(f"Streaming English finepdfs... target: {TARGET_CHUNKS} chunks\n")

    dataset = load_dataset(
        "HuggingFaceFW/finepdfs",
        "eng_Latn",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    for record in dataset:
        raw_text = record.get("text", "")
        if not raw_text:
            continue

        text = clean_text(raw_text)

        if len(text) < MIN_DOC_CHARS or len(text) > MAX_DOC_CHARS:
            continue

        # estimate how many chunks this doc adds before committing
        estimated = estimate_chunks(text)
        if total_chunks + estimated > TARGET_CHUNKS + 200:
            break

        doc_count += 1
        filename = f"doc_{doc_count:03d}.txt"
        filepath = os.path.join(CORPUS_DIR, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)

        chunks = chunker.chunk_document(text, metadata={"source": filename})
        total_chunks += len(chunks)

        # show a one-line preview of what was saved
        preview = text[:80].replace("\n", " ")
        print(f"  [{doc_count:03d}] {len(chunks):3d} chunks  {filename}  \"{preview}...\"")

        if total_chunks >= TARGET_CHUNKS:
            break

    print(f"\nDone. {doc_count} documents saved to data/corpus/")
    print(f"Total estimated chunks: {total_chunks}")
    print(f"Estimated enrichment cost: ${total_chunks * 0.000086:.4f}")
    print(f"Estimated full evaluation cost: ${(total_chunks * 0.000086) + (20 * 3 * 0.00018):.4f}")


if __name__ == "__main__":
    main()

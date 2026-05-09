import re
from typing import List, Dict

class SemanticChunker:
    def __init__(self, max_chunk_size: int = 500, overlap: int = 50):
        """
        Initialize the chunker.
        max_chunk_size: Target maximum length of a chunk in characters.
        overlap: How many characters to overlap between chunks to preserve context.
        """
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Splits text into sentences. This is the core of 'semantic' chunking — 
        we respect natural language boundaries rather than cutting in the middle of a word.
        """
        # A simple regex that splits on periods, question marks, or exclamation marks 
        # followed by a space.
        sentences = re.split(r'(?<=[.!?]) +', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def chunk_document(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Groups sentences into chunks that are close to max_chunk_size, 
        with some overlap between consecutive chunks.
        """
        if metadata is None:
            metadata = {}

        sentences = self.split_into_sentences(text)
        chunks = []
        
        current_chunk = ""
        
        for i, sentence in enumerate(sentences):
            # If adding the next sentence exceeds our limit, save the current chunk
            if len(current_chunk) + len(sentence) > self.max_chunk_size and current_chunk:
                chunks.append({
                    "text": current_chunk.strip(),
                    "metadata": metadata.copy()
                })
                
                # Start the new chunk with the overlap
                # We do this by keeping the last sentence(s) from the previous chunk
                # that fit within our overlap budget
                words = current_chunk.split()
                overlap_text = ""
                # Simple approximation: grab words from the end until we hit overlap limit
                for word in reversed(words):
                    if len(overlap_text) + len(word) < self.overlap:
                        overlap_text = word + " " + overlap_text
                    else:
                        break
                        
                current_chunk = overlap_text + sentence + " "
            else:
                current_chunk += sentence + " "
                
        # Don't forget the last chunk!
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "metadata": metadata.copy()
            })
            
        # Add chunk ID to metadata for tracking
        for i, chunk in enumerate(chunks):
            chunk["metadata"]["chunk_id"] = i
            
        return chunks

# --- Testing the chunker ---
if __name__ == "__main__":
    # Let's read our sample aviation document
    with open("data/sample_docs/aviation_manual.txt", "r") as f:
        text = f.read()
        
    chunker = SemanticChunker(max_chunk_size=400, overlap=50)
    
    # We pass metadata like the source file, so we know where this chunk came from
    chunks = chunker.chunk_document(text, metadata={"source": "aviation_manual.txt"})
    
    print(f"Created {len(chunks)} chunks.\\n")
    
    for i, chunk in enumerate(chunks[:3]): # Just print first 3
        print(f"--- Chunk {i} ---")
        print(f"Metadata: {chunk['metadata']}")
        print(f"Length: {len(chunk['text'])} characters")
        print(f"Text: {chunk['text']}\\n")

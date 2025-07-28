import fitz  # PyMuPDF
from typing import List, Tuple

def extract_chunks(pdf_path: str, chunk_size: int = 300) -> List[Tuple[str, int, str]]:
    doc = fitz.open(pdf_path)
    chunks = []

    for page_num, page in enumerate(doc):
        text = page.get_text()
        tokens = text.split()
        for i in range(0, len(tokens), chunk_size):
            chunk = " ".join(tokens[i:i + chunk_size])
            chunks.append((pdf_path, page_num + 1, chunk))
    return chunks


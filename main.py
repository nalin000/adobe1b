from config import *
from utils.pdf_parser import extract_chunks
from models.retriever import Retriever
from models.generator import Generator
from utils.utils import save_output, get_timestamp
import os

# === Inputs ===
persona = {
    "role": "Travel Planner",
    "focus": "Group trip planning, destination curation, budgeting"
}
job = "Plan a trip of 4 days for a group of 10 college friends"


# === Load Documents ===
all_chunks = []
doc_paths = [str(DATA_DIR / f) for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
for path in doc_paths:
    all_chunks.extend(extract_chunks(path))

# === Retrieval ===
retriever = Retriever(EMBEDDING_MODEL)
retriever.build_index(all_chunks)
top_chunks = retriever.query(job, top_k=10)

# === Generation ===
generator = Generator(GENERATION_MODEL)

extracted_sections = []
refined_texts = []

for rank, (doc, page, text) in enumerate(top_chunks, 1):
    summary = generator.generate_summary(text, job)
    extracted_sections.append({
        "document": os.path.basename(doc),
        "page_number": page,
        "section_title": summary[:40] + "...",
        "importance_rank": rank
    })
    refined_texts.append({
        "document": os.path.basename(doc),
        "refined_text": summary,
        "page_number": page
    })

# === Save JSON ===
metadata = {
    "input_documents": [os.path.basename(d) for d in doc_paths],
    "persona": persona,
    "job_to_be_done": job,
    "processing_timestamp": get_timestamp()
}
save_output(metadata, extracted_sections, refined_texts, OUTPUT_FILE)
print(f"Output saved to {OUTPUT_FILE}")

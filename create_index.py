import faiss
import fitz  # PyMuPDF
import os
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load Sentence Transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Function to split PDF into overlapping text chunks
def split_pdf_to_chunks(pdf_path, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    doc = fitz.open(pdf_path)
    full_text = "\n\n".join([page.get_text("text") for page in doc])
    return text_splitter.split_text(full_text)

# Folder containing all PDFs
pdf_folder = "Data"
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

all_chunks = []

# Process each PDF and collect all text chunks
for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_folder, pdf_file)
    print(f"Processing: {pdf_file}")
    chunks = split_pdf_to_chunks(pdf_path)
    all_chunks.extend(chunks)

# Save all text chunks for later use
with open("text_chunks.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(all_chunks))

# Encode all text chunks and create FAISS index
chunk_embeddings = model.encode(all_chunks, convert_to_numpy=True)
index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
index.add(chunk_embeddings)

# Save FAISS index
faiss.write_index(index, "faiss_index")

print("FAISS index created and saved successfully!")
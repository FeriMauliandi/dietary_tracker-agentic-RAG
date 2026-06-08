import os
import sys
import torch
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.loader import load_data
from src.core.config import settings

load_dotenv()

CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")

def ingest_all_sources():
    print("Memulai proses Ingestion Data ke ChromaDB...")
    
    # Masukkan URL web dan path file lokal (relatif atau absolut) ke dalam list ini
    sources = [
        "data/nutrition.csv"
    ]
    
    if not sources:
        print("❌ Tidak ada sumber data di dalam list 'sources'.")
        return

    all_documents = []
    
    for source in sources:
        try:
            docs = load_data(source)
            all_documents.extend(docs)
            print(f"✅ Berhasil memuat: {source}")
        except Exception as e:
            print(f"⚠️ Gagal memuat {source}: {e}")

    if not all_documents:
        print("❌ Gagal mengekstrak dokumen dari semua sumber. Proses dihentikan.")
        return

    # Proses Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(all_documents)
    
    # Embedding & Penyimpanan ke ChromaDB
    print("Memulai proses embedding dan penyimpanan...")
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
    
    vector_store = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=CHROMA_DB_DIR
    )
    
    print(f"🎉 Selesai! {len(chunks)} chunks tersimpan di ChromaDB.")

if __name__ == "__main__":
    ingest_all_sources()
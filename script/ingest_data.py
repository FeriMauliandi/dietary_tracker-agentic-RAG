import os
import sys
import torch
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Mengimpor fungsi load_data dari folder src.utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.loader import load_data

load_dotenv()

CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")

def ingest_all_sources():
    print("Memulai proses Ingestion Data ke ChromaDB...")
    
    # Masukkan URL web dan path file lokal (relatif atau absolut) ke dalam list ini
    sources = [
        # "https://kalbenutritionals.com/en/health-corner/ternyata-sarapan-favorit-orang-indonesia-masih-rendah-serat",
        # "https://www.alodokter.com/pilihan-menu-sarapan-sehat-agar-tubuh-berenergi",
        # "https://hellosehat.com/nutrisi/fakta-gizi/porsi-sarapan-yang-ideal/",
        # "https://www.emc.id/id/care-plus/inilah-komposisi-zat-gizi-yang-penting-dikonsumsi-saat-sarapan",
        "https://www.andrafarm.com/_andra.php?_i=daftar-tkpi",
        "https://www.panganku.org/id-ID/semua_nutrisi",
        "data/nutrition.csv"
    ]
    
    if not sources:
        print("❌ Tidak ada sumber data di dalam list 'sources'.")
        return

    all_documents = []
    
    # Looping semua sumber menggunakan loader.py milikmu
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
    print("🤖 Memulai proses embedding dan penyimpanan...")
    embeddings = HuggingFaceEmbeddings(
        model_name="LazarusNLP/all-indo-e5-small-v4", 
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
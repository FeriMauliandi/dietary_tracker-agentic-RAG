import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever, MultiQueryRetriever
from langchain_core.documents import Document
from src.core.config import settings
from src.utils.embeddings import get_cached_embeddings

def get_retriever(search_type="similarity", final_k=3):
    print("Menyiapkan Pure Hybrid Retriever (BM25 + Vector) yang dioptimalkan untuk entitas...")
    
    embeddings = get_cached_embeddings()
    
    vector_store = Chroma(
        persist_directory=settings.CHROMA_DB_DIR, 
        embedding_function=embeddings
    )
    
    # 1. Hitung dokumen untuk Defensive Programming
    db_data = vector_store.get()
    num_docs = len(db_data.get('documents', [])) if db_data else 0
    
    if num_docs == 0:
        print("⚠️ Peringatan: Database kosong. Menggunakan k=1 sebagai fallback.")
        safe_k = 1
    else:
        safe_k = min(final_k, num_docs)

    # 2. Vector Retriever murni
    vector_retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": safe_k}
    )
    
    # 3. BM25 Retriever murni
    if num_docs == 0:
        return vector_retriever
        
    docs = [
        Document(page_content=txt, metadata=meta or {}) 
        for txt, meta in zip(db_data['documents'], db_data.get('metadatas', []))
    ]
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = safe_k
    
    # 4. Ensemble (BM25 diberi bobot lebih besar karena mencari entitas persis)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.7, 0.3] # 70% mengandalkan kecocokan teks persis, 30% semantik
    )
    
    return ensemble_retriever
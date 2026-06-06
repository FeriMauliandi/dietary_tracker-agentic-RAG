import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever, MultiQueryRetriever
from langchain_core.documents import Document
from src.core.config import settings

def get_retriever(llm, search_type="similarity", final_k=3, fetch_k=3):
    print(f"Menyiapkan Advanced Retriever (Multi-Query + Hybrid + Reranker)...")
    
    # 1. Setup Konfigurasi Vektor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL, 
        model_kwargs={"device": device}
    )
    
    vector_store = Chroma(
        persist_directory=settings.CHROMA_DB_DIR, 
        embedding_function=embeddings
    )
    
    # 2. Vector Retriever
    vector_retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": fetch_k}
    )
    
    # 3. Setup BM25 Retriever
    db_data = vector_store.get()
    if not db_data or not db_data.get('documents'):
        print("⚠️ Peringatan: Tidak ada dokumen untuk BM25. Menggunakan Vector Retriever.")
        base_retriever = vector_retriever
    else:
        docs = [
            Document(page_content=txt, metadata=meta or {}) 
            for txt, meta in zip(db_data['documents'], db_data.get('metadatas', []))
        ]
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = fetch_k
        
        # Base Retriever (Hybrid)
        base_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]
        )
    
    # 4. Multi-Query Retriever
    # Membungkus base_retriever agar satu pertanyaan dipecah jadi banyak variasi oleh LLM
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, 
        llm=llm
    )
    
    # 5. Reranker Setup (Menggunakan Cross-Encoder)
    cross_encoder = HuggingFaceCrossEncoder(
        model_name="BAAI/bge-reranker-base", 
        model_kwargs={"device": device}
    )
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=final_k)
    
    # 6. Final Pipeline (Reranker menilai hasil dari Multi-Query)
    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=multi_query_retriever
    )
    
    return final_retriever
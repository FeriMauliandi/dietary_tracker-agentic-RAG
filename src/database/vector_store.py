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

def get_hybrid_retriever(search_type="similarity", final_k=3):
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
        weights=[0.6, 0.4] # 60% mengandalkan kecocokan teks persis, 40% semantik
    )
    
    return ensemble_retriever

def get_advanced_retriever(llm, search_type="similarity", final_k=3, fetch_k=5):
    print("Menyiapkan Advanced Retriever (MultiQuery + Hybrid + Reranker) untuk General Chat...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = get_cached_embeddings()
    vector_store = Chroma(persist_directory=settings.CHROMA_DB_DIR, embedding_function=embeddings)
    
    db_data = vector_store.get()
    num_docs = len(db_data.get('documents', [])) if db_data else 0
    
    if num_docs == 0:
        safe_fetch_k = 1
        safe_final_k = 1
    else:
        # fetch_k lebih besar agar reranker punya banyak pilihan dokumen untuk disortir
        safe_fetch_k = min(fetch_k, num_docs) 
        safe_final_k = min(final_k, num_docs)

    vector_retriever = vector_store.as_retriever(search_kwargs={"k": safe_fetch_k})
    
    if num_docs == 0:
        base_retriever = vector_retriever
    else:
        docs = [Document(page_content=txt, metadata=meta or {}) for txt, meta in zip(db_data['documents'], db_data.get('metadatas', []))]
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = safe_fetch_k
        
        # Bobot seimbang untuk literatur umum
        base_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5])
    
    multi_query_retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
    
    cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base", model_kwargs={"device": device})
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=safe_final_k)
    
    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=multi_query_retriever
    )
    return final_retriever
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from src.agents.state import DietaryTrackerState
from src.core.config import settings
from src.database.vector_store import get_hybrid_retriever, get_advanced_retriever
from src.tools.nutrition_api import fetch_combined_nutrition_data

llm = ChatGroq(
    model=settings.LLM_MODEL,
    temperature=0, 
)

diet_retriever = get_hybrid_retriever(final_k=3)
chat_retriever = get_advanced_retriever(llm, final_k=3,  fetch_k=5)

class IntentClassification(BaseModel):
    intent: str = Field(
        description="Pilih 'track_diet' JIKA pengguna menyebutkan detail makanan yang mereka konsumsi. Pilih 'general_chat' JIKA pengguna hanya menyapa, basa-basi, bertanya seputar nutrisi secara umum")

class FoodItem(BaseModel):
    asli: str = Field(description="Nama makanan/minuman dalam bahasa Indonesia")
    english: str = Field(description="Terjemahan bahasa Inggris (misal: 'ayam bakar' -> 'grilled chicken')")

class ExtractionResult(BaseModel):
    items: List[FoodItem] = Field(description="Daftar item yang diekstrak")

# --- NODES ---
def intent_node(state: DietaryTrackerState) -> Dict[str, Any]:
    print("🚦 [Router Node] Menganalisis niat (intent) pengguna...")
    structured_llm = llm.with_structured_output(IntentClassification)
    
    prompt = PromptTemplate.from_template(
        "Analisis input pengguna berikut dan tentukan niat utamanya.\n\n"
        "Input: {user_input}\n"
    )
    chain = prompt | structured_llm
    response = chain.invoke({"user_input": state["user_input"]})
    
    print(f"[Router Node] Keputusan: {response.intent}")
    return {"intent": response.intent}

def general_chat_node(state: DietaryTrackerState) -> Dict[str, Any]:
    print("[General Chat Node] Menjawab pertanyaan umum dengan bantuan RAG...")
    user_input = state.get("user_input", "")
    
    try:
        docs = chat_retriever.invoke(user_input)
        context = "\n\n".join(doc.page_content for doc in docs)
        
        if not context:
            context = "Tidak ada literatur spesifik yang ditemukan di database."
            
        print(f"[General Chat Node] Literatur pendukung ditemukan:\n{context}\n================================")
    except Exception as e:
        print(f"[General Chat Node] Error RAG: {e}")
        context = "Gagal mengambil data dari database lokal."

    prompt = PromptTemplate.from_template(
        "Kamu adalah asisten ahli gizi yang cerdas dan ramah.\n"
        "Jawab pertanyaan atau sapaan pengguna berikut dengan bahasa Indonesia yang sopan dan ringkas.\n"
        "JIKA pengguna bertanya seputar teori nutrisi, kesehatan, atau diet, GUNAKAN referensi literatur berikut sebagai dasar jawabanmu.\n"
        "JIKA literatur tidak relevan dengan pertanyaan, abaikan literatur tersebut.\n\n"
        "Referensi Literatur: {context}\n\n"
        "Input Pengguna: {user_input}"
    )
    
    chain = prompt | llm
    response = chain.invoke({
        "user_input": user_input,
        "context": context
    })
    
    return {
        "final_analysis": response.content,
        "literature_context": context
    }

def extraction_node(state: DietaryTrackerState) -> Dict[str, Any]:
    print("[Extraction Node] Sedang mengekstrak dan menerjemahkan entitas...")
    
    structured_llm = llm.with_structured_output(ExtractionResult)
    
    prompt = PromptTemplate.from_template(
        "Kamu adalah asisten ahli gizi bilingual. Ekstrak nama makanan dan minuman dari teks berikut.\n\n"
        "Teks: {user_input}\n\n"
        "Tugas:\n"
        "1. Ambil nama makanan dalam bahasa aslinya (Indonesia).\n"
        "2. Berikan terjemahan bahasa Inggris yang akurat untuk pencarian database USDA."
    )
    
    chain = prompt | structured_llm
    response = chain.invoke({"user_input": state["user_input"]})
    
    extracted_data = [{"asli": item.asli, "english": item.english} for item in response.items]
    
    print(f"[Extraction Node] Hasil: {extracted_data}")
    return {"extracted_items": extracted_data}

def api_tool_node(state: DietaryTrackerState) -> Dict[str, Any]:
    print("[API Tool Node] Mengambil data makronutrien (FatSecret & USDA)...")
    items_data = state.get("extracted_items", [])
    
    if not items_data:
        return {"nutrition_data": {"summary": "Tidak ada data makanan untuk dianalisis."}}
    
    # Lempar dictionary yang berisi versi Indo & English ke fungsi API
    nutrition_result = fetch_combined_nutrition_data(items_data)
    
    print(f"[API Tool Node] Hasil nutrisi akhir: {nutrition_result}")
    return {"nutrition_data": nutrition_result}

def rag_node(state: DietaryTrackerState) -> Dict[str, Any]:
    print("[RAG Node] Melakukan Hybrid Search (BM25 + Vector) di database...")
    items_data = state.get("extracted_items", [])
    
    # RAG hanya butuh bahasa aslinya (Indonesia) untuk dicari di database lokalmu
    if items_data:
        nama_asli_list = [item["asli"] for item in items_data]
        search_query = " ".join(nama_asli_list)
    else:
        search_query = state.get("user_input", "") 
        
    print(f"[RAG Node] Kueri pencarian diubah menjadi: '{search_query}'")
    
    try:
        docs = diet_retriever.invoke(search_query)
        context = "\n\n".join(doc.page_content for doc in docs)
        
        if not context:
            context = "Tidak ada literatur spesifik yang ditemukan di database."
            
        print(f"[RAG Node] Literatur yang ditemukan:\n{context}\n ================================")
        return {"literature_context": context}
    except Exception as e:
        print(f"[RAG Node] Error: {e}")
        return {"literature_context": "", "error_logs": [f"RAG Error: {str(e)}"]}

def synthesizer_node(state: DietaryTrackerState) -> Dict[str, Any]:
    print("[Synthesizer Node] Merumuskan analisis akhir...")
    
    items_data = state.get("extracted_items", [])
    # Ekstrak nama asli agar LLM menjawab pengguna dengan bahasa Indonesia
    nama_asli_list = [item["asli"] for item in items_data]
    
    prompt = PromptTemplate.from_template(
        "Kamu adalah konsultan kebugaran dan nutrisi berbasis sains.\n\n"
        "Input Asli Pengguna: '{user_input}'\n"
        "Makanan yang diekstrak: {items}\n"
        "Data Nutrisi (Estimasi): {nutrition}\n"
        "Literatur nutrisi Pendukung: {context}\n\n"
        "Tugas Utama:\n"
        "1. Berikan analisis nutrisi per item secara singkat.\n"
        "2. Jelaskan apakah asupannya sudah ideal DENGAN MEMPERHATIKAN KONTEKS WAKTU MAKAN.\n"
        "Gunakan bahasa Indonesia yang profesional dan ringkas. Beri saran pelengkap jika perlu."
    )
    chain = prompt | llm
    
    response = chain.invoke({
        "user_input": state.get("user_input", ""),
        "items": ", ".join(nama_asli_list),
        "nutrition": state.get("nutrition_data", {}).get("summary", "Data tidak tersedia"),
        "context": state.get("literature_context", "Tidak ada referensi.")
    })
    return {"final_analysis": response.content}
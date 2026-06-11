import json
import re
from typing import Any, Dict, List

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate

from src.agents.state import DietaryTrackerState
from src.core.config import settings
from src.database.vector_store import get_advanced_retriever, get_hybrid_retriever
from src.tools.nutrition_api import fetch_combined_nutrition_data


# ---------------------------------------------------------------------------
# LLM & Retrievers
# ---------------------------------------------------------------------------

llm = ChatGroq(model=settings.LLM_MODEL, temperature=0)

diet_retriever = get_hybrid_retriever(final_k=3)
chat_retriever = get_advanced_retriever(llm, final_k=3, fetch_k=5)


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class IntentClassification(BaseModel):
    intent: str = Field(
        description=(
            "Pilih 'track_diet' JIKA pengguna menyebutkan detail makanan yang mereka konsumsi. "
            "Pilih 'general_chat' JIKA pengguna hanya menyapa, basa-basi, atau bertanya seputar nutrisi secara umum."
        )
    )


class FoodItem(BaseModel):
    asli: str = Field(description="Nama makanan/minuman dalam bahasa Indonesia")
    english: str = Field(description="Terjemahan bahasa Inggris (misal: 'ayam bakar' -> 'grilled chicken')")


class ExtractionResult(BaseModel):
    items: List[FoodItem] = Field(description="Daftar item yang diekstrak")


class ClarificationDecision(BaseModel):
    needs_clarification: bool = Field(
        description="True jika input belum cukup untuk estimasi nutrisi (porsi/jumlah dan jam makan tidak jelas)."
    )
    question: str = Field(
        description="Pertanyaan klarifikasi singkat dalam bahasa Indonesia. Kosongkan jika tidak perlu."
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMMON_TRANSLATIONS: Dict[str, str] = {
    "siomay": "steamed fish dumpling",
    "tahu": "tofu",
    "kentang": "potato",
    "kol rebus": "boiled cabbage",
    "kol": "cabbage",
}

PORTION_PATTERN = re.compile(
    r"\b\d+\s*(porsi|piring|mangkuk|potong|gram|g|kg|gelas|buah|bungkus|sendok|sdm|sdt)\b"
    r"|\b(se)?(porsi|piring|mangkuk|gelas|bungkus|buah)\b",
    re.IGNORECASE,
)

CLARIFICATION_NOISE_PATTERN = re.compile(
    r"\b(porsi|utama|waktu|jam|pukul|pagi|siang|sore|malam|sarapan|breakfast|lunch|dinner|brunch)\b",
    re.IGNORECASE,
)

TIME_PATTERN = re.compile(
    r"\b(pagi|siang|sore|malam|sarapan|breakfast|lunch|dinner|brunch)\b"
    r"|\b(makan\s+)?(pagi|siang|sore|malam)\b"
    r"|\b(jam|pukul)\s*\d{1,2}([:.]\d{2})?\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_extracted_items(items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Deduplicate and normalize a list of extracted food items."""
    normalized: List[Dict[str, str]] = []
    seen: set = set()

    for item in items:
        asli = str(item.get("asli", "")).strip().lower()
        english = str(item.get("english", "")).strip().lower()

        if not asli or asli in seen:
            continue

        seen.add(asli)
        normalized.append({
            "asli": asli,
            "english": english or COMMON_TRANSLATIONS.get(asli, asli),
        })

    return normalized


def likely_clarification_only(user_input: str) -> bool:
    """Return True if the input contains only clarification tokens (no food names)."""
    text = PORTION_PATTERN.sub(" ", user_input.lower())
    text = TIME_PATTERN.sub(" ", text)
    text = CLARIFICATION_NOISE_PATTERN.sub(" ", text)
    text = re.sub(r"[^a-zA-Z\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return not text


def fallback_extract_items(user_input: str) -> List[Dict[str, str]]:
    """Rule-based fallback extraction when the LLM structured output fails."""
    text = user_input.lower()
    text = re.sub(r"\b\d+\s*(porsi|piring|mangkuk|potong|gram|gelas|buah|bungkus)\b", "", text)
    text = re.sub(r"\b(saya|aku|makan|minum|isinya|isi|dengan|detail|tambahan|porsi)\b", "", text)

    parts = re.split(r",|\bdan\b|\+|/", text)
    items: List[Dict[str, str]] = []

    for part in parts:
        name = re.sub(r"[^a-zA-Z\s-]", " ", part).strip()
        name = re.sub(r"\s+", " ", name)
        if len(name) < 3:
            continue
        items.append({"asli": name, "english": COMMON_TRANSLATIONS.get(name, name)})

    return normalize_extracted_items(items)


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------

def intent_node(state: DietaryTrackerState) -> Dict[str, Any]:
    """Classify user intent: 'track_diet' or 'general_chat'."""
    print("[Router Node] Menganalisis niat (intent) pengguna...")

    # If we're mid-clarification, continue tracking diet without re-classifying.
    if state.get("needs_clarification") and state.get("extracted_items"):
        print("[Router Node] Melanjutkan klarifikasi diet yang tertunda.")
        return {"intent": "track_diet", "needs_clarification": False}

    structured_llm = llm.with_structured_output(IntentClassification)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tentukan intent dari input pengguna. track_diet: jika ada detail makanan. general_chat: sapaan atau tanya nutrisi umum."),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{user_input}"),
    ])

    response = (prompt | structured_llm).invoke({
        "user_input": state["user_input"],
        "messages": state.get("messages", []),
    })

    print(f"[Router Node] Keputusan: {response.intent}")
    return {"intent": response.intent}


def general_chat_node(state: DietaryTrackerState) -> Dict[str, Any]:
    """Answer general nutrition questions using RAG-backed context."""
    print("[General Chat Node] Menjawab pertanyaan umum...")

    user_input = state.get("user_input", "")
    messages = state.get("messages", [])

    try:
        docs = chat_retriever.invoke(user_input)
        context = "\n\n".join(doc.page_content for doc in docs) or "Tidak ada literatur."
    except Exception as e:
        print(f"Error RAG: {e}")
        context = "Gagal ambil database."

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Asisten gizi. Jawab ringkas & sopan. Gunakan literatur jika relevan.\nLiteratur: {context}"),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{user_input}"),
    ])

    response = (prompt | llm).invoke({
        "user_input": user_input,
        "context": context,
        "messages": messages,
    })

    new_messages = messages + [HumanMessage(content=user_input), AIMessage(content=response.content)]
    return {
        "final_analysis": response.content,
        "literature_context": context,
        "messages": new_messages,
    }


def extraction_node(state: DietaryTrackerState) -> Dict[str, Any]:
    """Extract food entities from user input."""
    print("[Extraction Node] Ekstraksi entitas...")

    user_input = state.get("user_input", "")
    messages = state.get("messages", [])
    existing_items = state.get("extracted_items", [])

    # If the user is only supplying clarification details, keep existing items.
    if existing_items and likely_clarification_only(user_input):
        print("[Extraction Node] Input hanya berisi detail klarifikasi; item lama dipertahankan.")
        return {"extracted_items": normalize_extracted_items(existing_items)}

    try:
        structured_llm = llm.with_structured_output(ExtractionResult)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Ekstrak makanan (Indo & English) dari input pengguna."),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{user_input}"),
        ])
        response = (prompt | structured_llm).invoke({
            "user_input": user_input,
            "messages": messages,
        })
        extracted_data = [{"asli": item.asli, "english": item.english} for item in response.items]
    except Exception as e:
        print(f"Fallback extraction: {e}")
        extracted_data = fallback_extract_items(user_input)

    # Merge with any previously extracted items.
    if existing_items:
        existing_names = {ex["asli"] for ex in existing_items}
        for item in extracted_data:
            if item["asli"] not in existing_names:
                existing_items.append(item)
        extracted_data = existing_items

    extracted_data = normalize_extracted_items(extracted_data)
    print(f"[Extraction Node] Hasil: {extracted_data}")
    return {"extracted_items": extracted_data}


def clarification_node(state: DietaryTrackerState) -> Dict[str, Any]:
    """Ask the user for missing portion or meal-time information."""
    print("[Clarification Node] Cek kelengkapan data...")

    items_data = state.get("extracted_items", [])
    user_input = state.get("user_input", "")
    messages = state.get("messages", [])

    # Build a combined text from all previous user messages for pattern matching.
    context_text = " ".join(
        str(getattr(msg, "content", ""))
        for msg in messages
        if isinstance(msg, HumanMessage)
    )
    completeness_text = f"{context_text} {user_input}".strip()

    if not items_data:
        return {
            "needs_clarification": True,
            "clarification_question": "Apa yang Anda konsumsi?",
            "final_analysis": "Apa yang Anda konsumsi?",
        }

    has_portion = bool(PORTION_PATTERN.search(completeness_text))
    has_time = bool(TIME_PATTERN.search(completeness_text))
    needs_clarification = not (has_portion and has_time)

    if not needs_clarification:
        return {"needs_clarification": False}

    main_item = items_data[0].get("asli", "makanan")
    if not has_portion and not has_time:
        question = f"Berapa porsi {main_item} tersebut dan kapan waktu makannya?"
    elif not has_portion:
        question = f"Berapa porsi {main_item} tersebut?"
    else:
        question = "Kapan waktu makannya?"

    new_messages = messages + [HumanMessage(content=user_input), AIMessage(content=question)]
    return {
        "needs_clarification": True,
        "clarification_question": question,
        "final_analysis": question,
        "messages": new_messages,
    }


def api_tool_node(state: DietaryTrackerState) -> Dict[str, Any]:
    """Fetch nutrition data for all extracted food items."""
    print("[API Tool Node] Mengambil data nutrisi...")

    items_data = state.get("extracted_items", [])
    if not items_data:
        return {"nutrition_data": {"summary": "Tidak ada data."}}

    nutrition_result = fetch_combined_nutrition_data(items_data)
    print(f"[API Tool Node] Hasil: {nutrition_result}")
    return {"nutrition_data": nutrition_result}


def rag_node(state: DietaryTrackerState) -> Dict[str, Any]:
    """Retrieve supporting nutrition literature via hybrid search."""
    print("[RAG Node] Hybrid Search...")

    items_data = state.get("extracted_items", [])
    queries: List[str] = []

    for item in items_data:
        queries.append(item["asli"])
        if item.get("english") and item["english"] != item["asli"]:
            queries.append(item["english"])

    search_query = " ".join(queries) if queries else state.get("user_input", "")

    try:
        docs = diet_retriever.invoke(search_query)
        context = "\n\n".join(doc.page_content for doc in docs) or "Tidak ada literatur."
        return {"literature_context": context}
    except Exception as e:
        print(f"RAG Error: {e}")
        return {"literature_context": "", "error_logs": [str(e)]}


def synthesizer_node(state: DietaryTrackerState) -> Dict[str, Any]:
    """Synthesize nutrition data and literature into a final analysis."""
    print("[Synthesizer Node] Analisis akhir...")

    user_input = state.get("user_input", "")
    messages = state.get("messages", [])

    prompt = PromptTemplate.from_template(
        "Kamu adalah konsultan kebugaran dan nutrisi berbasis sains.\n\n"
        "Input Asli Pengguna: '{user_input}'\n"
        "Makanan yang diekstrak: {items}\n"
        "Data Nutrisi (Estimasi): {nutrition}\n"
        "Literatur Nutrisi Pendukung: {context}\n\n"
        "Tugas Utama:\n"
        "1. Berikan analisis nutrisi per item secara singkat.\n"
        "2. Jelaskan apakah asupannya sudah ideal DENGAN MEMPERHATIKAN KONTEKS WAKTU MAKAN.\n"
        "Gunakan bahasa Indonesia yang profesional dan ringkas. Beri saran pelengkap jika perlu."
    )

    response = (prompt | llm).invoke({
        "user_input": user_input,
        "items": ", ".join(i["asli"] for i in state.get("extracted_items", [])),
        "nutrition": state.get("nutrition_data", {}).get("summary", "N/A"),
        "context": state.get("literature_context", "N/A"),
        "messages": messages,
    })

    new_messages = messages + [HumanMessage(content=user_input), AIMessage(content=response.content)]
    return {"final_analysis": response.content, "messages": new_messages}
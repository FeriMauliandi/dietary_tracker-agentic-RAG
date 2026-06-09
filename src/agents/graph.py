from langgraph.graph import StateGraph, END
import os, sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_dir)

# Import State dan Nodes yang sudah dibuat
from src.agents.state import DietaryTrackerState
from src.agents.nodes import (
    intent_node,
    general_chat_node,
    extraction_node,
    api_tool_node,
    rag_node,
    synthesizer_node
)

def route_intent(state: DietaryTrackerState):
    """Membaca state intent dan menentukan node selanjutnya"""
    if state.get("intent") == "track_diet":
        return "extraction"
    return "general_chat"

# Inisialisasi StateGraph
workflow = StateGraph(DietaryTrackerState)

# Daftarkan semua nodes
workflow.add_node("intent_router", intent_node)
workflow.add_node("general_chat", general_chat_node)
workflow.add_node("extraction", extraction_node)
workflow.add_node("api_tool", api_tool_node)
workflow.add_node("rag", rag_node)
workflow.add_node("synthesizer", synthesizer_node)

workflow.set_entry_point("intent_router")

# 4. Tambahkan Logika Cabang (Conditional Edges)
workflow.add_conditional_edges(
    "intent_router",
    route_intent,
    {
        "extraction": "extraction",   # Jika return "extraction", pergi ke extraction_node
        "general_chat": "general_chat" # Jika return "general_chat", pergi ke general_chat_node
    }
)

# 5. Susun Alur Rute Percakapan Umum
workflow.add_edge("general_chat", END)

workflow.add_edge("extraction", "api_tool")
workflow.add_edge("extraction", "rag")

workflow.add_edge("api_tool", "synthesizer")
workflow.add_edge("rag", "synthesizer")

workflow.add_edge("synthesizer", END)

app = workflow.compile()

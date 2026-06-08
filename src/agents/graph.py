from langgraph.graph import StateGraph, END
import os, sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_dir)

# Import State dan Nodes yang sudah dibuat
from src.agents.state import DietaryTrackerState
from src.agents.nodes import (
    extraction_node,
    api_tool_node,
    rag_node,
    synthesizer_node
)

# Inisialisasi StateGraph
workflow = StateGraph(DietaryTrackerState)

# Daftarkan semua nodes
workflow.add_node("extraction", extraction_node)
workflow.add_node("api_tool", api_tool_node)
workflow.add_node("rag", rag_node)
workflow.add_node("synthesizer", synthesizer_node)

workflow.set_entry_point("extraction")

workflow.add_edge("extraction", "api_tool")
workflow.add_edge("extraction", "rag")

workflow.add_edge("api_tool", "synthesizer")
workflow.add_edge("rag", "synthesizer")

workflow.add_edge("synthesizer", END)

app = workflow.compile()

if __name__ == "__main__":
    print("🚀 === MEMULAI SIMULASI AGENTIC RAG ===")
    
    # Simulasi input dari pengguna
    user_input = (
        "Habis puasa seharian, tadi buka minum segelas air kelapa campur sedikit garam, "
        "terus makan malamnya pakai dada ayam bakar 1 porsi. Gimana secara nutrisinya?"
    )
    
    # Inisialisasi State Awal
    initial_state = {
        "user_input": user_input,
        "extracted_items": [],
        "nutrition_data": {},
        "literature_context": "",
        "final_analysis": "",
        "error_logs": []
    }
    
    # Menjalankan workflow
    print(f"\n📥 Input Pengguna: '{user_input}'\n")
    
    final_state = app.invoke(initial_state)
    
    print("\n✅ === HASIL ANALISIS AKHIR ===")
    print(final_state["final_analysis"])
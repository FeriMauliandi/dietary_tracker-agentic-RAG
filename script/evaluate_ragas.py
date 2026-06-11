import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from unittest.mock import MagicMock

# Mock missing vertexai module before importing ragas
try:
    import langchain_community.chat_models.vertexai
except ImportError:
    mock_module = MagicMock()
    sys.modules["langchain_community.chat_models.vertexai"] = mock_module

import asyncio
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

# Menambahkan root direktori ke sys.path agar bisa mengimpor module src
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(root_dir)

from src.agents.graph import app
from src.utils.embeddings import get_cached_embeddings
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
# Note: Newer ragas versions might want imports from ragas.metrics.collections
# but we will keep these for now as they still work with warnings, 
# or we can try to use the new ones if we want to be clean.
from langchain_groq import ChatGroq
from src.core.config import settings

# Load environment variables
load_dotenv()

# Konfigurasi LLM & Embeddings untuk Evaluasi
eval_llm = ChatGroq(
    model=settings.LLM_MODEL,
    temperature=0,
)
eval_embeddings = get_cached_embeddings()

# Dataset evaluasi (Sesuaikan dengan data literatur yang kamu miliki)
# Ground truth adalah jawaban ideal yang seharusnya diberikan oleh sistem
eval_questions = [
    {
        "question": "apa itu protein?",
        "ground_truth": "Protein adalah makronutrien (nutrisi makro) esensial yang berfungsi sebagai zat pembangun utama dalam tubuh. Terdiri dari rantai asam amino, protein sangat penting untuk pertumbuhan, pembentukan dan perbaikan sel atau jaringan tubuh, serta pembentukan enzim dan hormon."
    },
    {
        "question": "apa itu nutrisi?",
        "ground_truth": "Nutrisi adalah proses tersedianya energi dan bahan kimia dari makanan yang penting untuk pembentukan, pemeliharaan dan penggantian sel tubuh."
    },
    {
        "question": "apa itu karbohidrat?",
        "ground_truth": "Karbohidrat adalah gula sederhana (monosakarida dan disakarida) dan gula kompleks (polisakarida). Karbohidrat terdiri dari karbon, hidrogen, dan oksigen. Gula, sirup, madu, buah, dan susu adalah sumber karbohidrat sederhana. Roti, sereal, kentang, beras, pasta, dan gandum berisi karbohidrat kompleks"
    }
]

async def run_evaluation():
    results = []
    
    print(f"Memulai evaluasi {len(eval_questions)} pertanyaan...")
    
    for item in eval_questions:
        print(f"Mengevaluasi: {item['question']}")
        
        # Jalankan workflow LangGraph
        inputs = {"user_input": item["question"]}
        final_state = await app.ainvoke(inputs)
        
        # Ambil data yang dibutuhkan RAGAS
        answer = final_state.get("final_analysis", "")
        context = final_state.get("literature_context", "")
        
        # RAGAS mengharapkan contexts dalam bentuk list of strings
        # Kita split berdasarkan separator yang digunakan di nodes.py (\n\n)
        contexts = [c.strip() for c in context.split("\n\n") if c.strip()]
        if not contexts:
            contexts = ["Tidak ada konteks yang ditemukan."]

        results.append({
            "question": item["question"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"]
        })

    # Konversi ke format Dataset HuggingFace yang diminta RAGAS
    dataset = Dataset.from_list(results)
    
    # Jalankan evaluasi
    print("Menghitung skor RAGAS...")
    
    try:
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=eval_llm,
            embeddings=eval_embeddings
        )
        
        # Tampilkan hasil
        df = result.to_pandas()
        print("\n=== HASIL EVALUASI ===")
        print(df)
        
        # Simpan ke CSV
        output_dir = os.path.join(root_dir, "data", "eval")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "ragas_results.csv")
        df.to_csv(output_path, index=False)
        print(f"\nHasil evaluation disimpan di: {output_path}")
        
    except Exception as e:
        print(f"Terjadi kesalahan saat evaluasi: {e}")
        print("\nTips: Pastikan kamu sudah menginstal library yang diperlukan:")
        print("pip install ragas datasets pandas")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_evaluation())

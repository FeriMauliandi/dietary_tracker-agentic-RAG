<div align="center">

# 🥗 Dietary Tracker AI

### Intelligent Nutrition Tracking with Agentic RAG & LangGraph

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_Workflow-orange)](https://langchain.com/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-teal?logo=fastapi)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-blueviolet)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/Groq-LLM_Inference-black)](https://groq.com)

</div>

---

## 📖 Tentang Proyek

**Dietary Tracker AI** adalah aplikasi pelacak nutrisi cerdas berbasis *Agentic Workflow*. Sistem ini tidak hanya mencatat kalori, tetapi bertindak sebagai konsultan gizi otonom yang mampu menimbang kapan harus menarik data dari API nutrisi global, dan kapan harus menggali literatur medis dari database lokal.

Proyek ini dibangun untuk mendemonstrasikan implementasi **Advanced RAG** (Retrieval-Augmented Generation) dan orkestrasi LLM menggunakan ekosistem Python modern, mengatasi tantangan umum seperti kesenjangan bahasa (*language barrier*) pada API dan halusinasi data.

---

## 🏗️ Arsitektur Sistem (Agentic Workflow)

Sistem menggunakan **LangGraph** untuk mengatur alur kerja cerdas (*conditional routing*). 

### Alur Orkestrasi (LangGraph Router)

```text
Input Pengguna → 🚦 Router Node (Intent Classification)
                      │
          ┌───────────┴───────────┐
          │                       │
     [track_diet]           [general_chat]
          │                       │
 🤖 Extraction Node      💬 General Chat Node
 (Pydantic Output)       (Advanced RAG Pipeline)
          │                       │
   ┌──────┴──────┐                │
   │             │                │
🌐 API Tool   📚 RAG Node         │
(FatSecret/  (Hybrid Search)      │
 USDA)           │                │
   └──────┬──────┘                │
          │                       │
   🧠 Synthesizer Node            │
          │                       │
          └───────────┬───────────┘
                      ↓
               Jawaban Akhir
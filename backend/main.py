from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.agents.graph import app as ai_agent_app
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path="data/cache/langchain_cache.db"))


app = FastAPI(
    title="Dietary Tracker Multi-Agent API",
    description="API untuk analisis nutrisi otonom menggunakan LangGraph & Groq",
    version="1.0.0"
)

class DietRequest(BaseModel):
    user_input: str

class DietResponse(BaseModel):
    extracted_items: List[str]
    final_analysis: str

@app.get("/")
def read_root():
    return {"message": "Dietary Tracker Agent API berjalan dengan lancar! 🚀"}

@app.post("/api/v1/analyze", response_model=DietResponse)
async def analyze_diet(request: DietRequest):
    print(f"📥 Menerima request analisis untuk: {request.user_input}")
    
    initial_state = {
        "user_input": request.user_input,
        "extracted_items": [],
        "nutrition_data": {},
        "literature_context": "",
        "final_analysis": "",
        "error_logs": []
    }
    
    try:
        result_state = ai_agent_app.invoke(initial_state)
    
        items_list = []
        for item in result_state.get("extracted_items", []):
            if isinstance(item, dict):
                items_list.append(item.get("asli", ""))
            else:
                items_list.append(item)

        # Kembalikan response
        return DietResponse(
            final_analysis=result_state["final_analysis"],
            extracted_items=items_list, 
        )
        
    except Exception as e:
        print(f"error saat memproses graph: {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan internal pada agen AI.")
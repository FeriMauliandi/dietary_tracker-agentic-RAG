import os
from dotenv import load_dotenv

# Memuat isi dari file .env
load_dotenv()

class Settings:
    """Kelas terpusat untuk menyimpan semua konfigurasi proyek."""
    
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    USDA_API_KEY = os.getenv("USDA_API_KEY")
    FATSECRET_CLIENT_ID = os.getenv("FATSECRET_CLIENT_ID")
    FATSECRET_CLIENT_SECRET = os.getenv("FATSECRET_CLIENT_SECRET")
    
    # Path Direktori
    # Mengarah ke folder dietary-tracker-agent/data/chroma_db
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    CHROMA_DB_DIR = os.path.join(BASE_DIR, "data", "chroma_db")
    
    # Konfigurasi Model
    LLM_MODEL = "openai/gpt-oss-20b"
    EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# Inisialisasi objek settings agar bisa diimpor ke file lain
settings = Settings()
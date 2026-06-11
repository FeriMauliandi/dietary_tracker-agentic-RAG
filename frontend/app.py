import streamlit as st
import requests

# Konfigurasi endpoint FastAPI yang sudah kita buat
API_URL = "http://127.0.0.1:8000/api/v1/analyze"

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Dietary Tracker AI",
    page_icon="🥗",
    layout="centered"
)

st.title("🥗 Agentic Dietary Tracker")
st.markdown("""
Asisten AI ini menggunakan Langchain advanced RAG (Vector + BM25) dan LangGraph multi-agent workflow untuk menganalisis asupan nutrisi Anda 
berdasarkan input teks natural dan literatur jurnal medis.
""")

# Inisialisasi riwayat obrolan di session state Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

# Menampilkan riwayat obrolan di layar
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kolom input untuk pengguna
if prompt := st.chat_input("Ceritakan apa yang baru saja Anda makan/minum..."):
    # 1. Tampilkan pesan pengguna di layar
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. Simpan pesan pengguna ke riwayat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 3. Kirim request ke FastAPI Backend
    with st.chat_message("assistant"):
        # Menggunakan status spinner agar terlihat proses berpikirnya
        with st.spinner("Agen sedang menganalisis (Ekstraksi, API(USDA food data), & RAG)..."):
            try:
                response = requests.post(
                    API_URL, 
                    json={
                        "user_input": prompt,
                        "session_id": st.session_state.session_id
                    },
                    timeout=60 # Timeout 60 detik untuk menunggu LLM
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Mengambil data dari response FastAPI
                    extracted_items = data.get("extracted_items", [])
                    final_analysis = data.get("final_analysis", "")
                    needs_clarification = data.get("needs_clarification", False)
                    
                    # Memformat jawaban akhir
                    if needs_clarification:
                         formatted_response = f"🔍 **Klarifikasi:**\n{final_analysis}"
                    else:
                         formatted_response = f"**Item terdeteksi:** {', '.join(extracted_items)}\n\n---\n\n**Analisis Gizi & Literatur:**\n{final_analysis}"
                    
                    # Menampilkan jawaban di layar
                    st.markdown(formatted_response)
                    
                    # Menyimpan jawaban asisten ke riwayat
                    st.session_state.messages.append({"role": "assistant", "content": formatted_response})
                    
                else:
                    error_msg = f"❌ Error dari server: {response.status_code}"
                    st.error(error_msg)
                    
            except requests.exceptions.ConnectionError:
                st.error("🔌 Gagal terhubung ke backend. Pastikan server FastAPI sudah berjalan di port 8000.")
            except Exception as e:
                st.error(f"⚠️ Terjadi kesalahan: {str(e)}")
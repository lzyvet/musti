import os
import zipfile
import glob
import subprocess
import streamlit as st
from difflib import get_close_matches
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. OTOMATİK ZIP BİRLEŞTİRME VE ÇIKARMA ---
if not os.path.exists("./chroma_db"):
    zip_parts = sorted(glob.glob("./chroma_db.z*"))
    if zip_parts:
        try:
            # Linux üzerinde 7z kuruluysa doğrudan ilk parçadan çıkartır (BadZipFile hatasını önler)
            subprocess.run(["7z", "x", zip_parts[0], "-o.", "-y"], check=True)
        except Exception:
            # 7z yoksa varsayılan binary birleştirme dener
            with open("./chroma_db.zip", "wb") as output_file:
                for part in zip_parts:
                    with open(part, "rb") as f:
                        output_file.write(f.read())
            
            if os.path.exists("./chroma_db.zip"):
                with zipfile.ZipFile("./chroma_db.zip", "r") as zip_ref:
                    zip_ref.extractall(".")

# --- 2. STREAMLIT ARAYÜZ VE AYARLAR ---
st.set_page_config(page_title="Veteriner Tıbbı Yapay Zeka Asistanı", page_icon="🩺", layout="wide")

DB_DIR = "./chroma_db"
WORDS_FILE = "./pdf_words.txt"

@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    if os.path.exists(DB_DIR):
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return None

def get_closest_suggestions(query):
    if not os.path.exists(WORDS_FILE):
        return []
    try:
        with open(WORDS_FILE, "r", encoding="utf-8") as f:
            words_list = f.read().split()
        query_clean = query.lower().strip()
        prefix_matches = [w for w in words_list if w.startswith(query_clean) and w != query_clean]
        fuzzy_matches = get_close_matches(query_clean, words_list, n=5, cutoff=0.5)
        combined = list(dict.fromkeys(prefix_matches + fuzzy_matches))
        return combined[:5]
    except Exception:
        return []

with st.sidebar:
    st.title("🩺 Veteriner Tıbbı Asistanı")
    st.success("✅ **18 Akademik Kitap Yüklü ve Aktif**")
    if st.button("Sohbeti Temizle"):
        st.session_state.messages = []
        st.rerun()

st.title("💬 İleri Düzey Veteriner Tıbbı & PDF Analiz Asistanı")
st.write("Sisteminizde 18 adet klinik kaynak ve kılavuz taranmaktadır.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Soru veya tıbbi terim yazın (Örn: slm, enro, kedi astım tedavisi...)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    query_lower = prompt.lower().strip()
    greetings = ["merhaba", "selam", "slm", "günaydın", "iyi günler", "kimsin", "ne işe yararsın", "nasılsın"]

    if any(g == query_lower or g in query_lower.split() for g in greetings) and len(query_lower.split()) <= 3:
        bot_msg = (
            "🧬 **Merhaba Sayın Hekimim! Ben Veteriner Tıbbı Yapay Zeka Asistanınızım.**\n\n"
            "Sistemimde hazır indekslenmiş **18 Temel Klinik Kitap ve Uzmanlık Kılavuzu** aktif olarak taranmaktadır.\n"
            "💡 *Doğrudan klinik vakalarınızı, dozaj sorularınızı veya hastalık semptomlarını sorabilirsiniz!*"
        )
        st.session_state.messages.append({"role": "assistant", "content": bot_msg})
        with st.chat_message("assistant"):
            st.markdown(bot_msg)
    else:
        api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            st.error("⚠️ Lütfen Streamlit ayarlarından 'GROQ_API_KEY' tanımını kontrol edin.")
        else:
            vector_store = load_vector_store()
            if vector_store is None:
                st.error("⚠️ 'chroma_db' dizini bulunamadı! Lütfen zip dosyalarının yüklendiğinden emin olun.")
            else:
                with st.spinner("18 kitaptan araştırılıyor..."):
                    try:
                        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
                        
                        llm = ChatGroq(
                            groq_api_key=api_key, 
                            model_name="llama-3.3-70b-versatile", 
                            temperature=0.1
                        )
                        
                        system_prompt = (
                            "Sen veteriner hekimliği alanında uzmanlaşmış klinik kararları destekleyen tıbbi bir yapay zeka asistansın.\n\n"
                            "Önemli Kurallar:\n"
                            "1. Sadece verilen bağlamdaki (context) klinik, farmakolojik ve akademik bilgileri kullanarak net ve bilimsel yanıt ver.\n"
                            "2. Yüklenen belge İngilizce olsa dahi yanıtı HER ZAMAN akıcı, profesyonel veteriner terminolojisine uygun bir Türkçe ile ver.\n"
                            "3. Yanıtında bilginin alındığı klinik kitaba/bölüme mutlaka atıf yap.\n"
                            "4. Eğer aranan yanıt verilen bağlamda yoksa 'Bu bilgi sistemdeki veteriner literatüründe yer almamaktadır.' de.\n\n"
                            "Bağlam:\n{context}"
                        )
                        
                        chat_prompt = ChatPromptTemplate.from_messages([
                            ("system", system_prompt),
                            ("human", "{input}"),
                        ])
                        
                        question_answer_chain = create_stuff_documents_chain(llm, chat_prompt)
                        retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)
                        
                        response = retrieval_chain.invoke({"input": prompt})
                        bot_msg = response["answer"]
                        
                        suggestions = get_closest_suggestions(prompt)
                        if suggestions:
                            bot_msg += f"\n\n💡 *Aramanıza en yakın belgedeki terimler:* **{', '.join(suggestions)}**"
                            
                        st.session_state.messages.append({"role": "assistant", "content": bot_msg})
                        with st.chat_message("assistant"):
                            st.markdown(bot_msg)
                            
                    except Exception as e:
                        st.error(f"❌ Bir hata oluştu: {str(e)}")

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

app = FastAPI(title="VetHelper AI API")

# Android uygulamasının erişebilmesi için CORS izinleri
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Anahtarı Kontrolü
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Vektör Veritabanı Yükleme
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
persist_dir = os.path.dirname(os.path.abspath(__file__))

try:
    vector_db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
except Exception:
    vector_db = None

# Veri Yapıları
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if not client:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY sunucuda tanımlı değil.")
    
    user_query = request.query
    kitap_baglami = ""
    
    # RAG - Veritabanı Araması
    if vector_db:
        try:
            docs = vector_db.similarity_search(user_query, k=8)
            if docs:
                kitap_baglami = "\n\n".join([doc.page_content for doc in docs])
        except Exception:
            pass

    # Sistem Talimatı (Etken Madde Öneri Mantığı Dahil)
    sistem_talimati = """
    [CRITICAL: ANSWER IN TURKISH ONLY.]
    Sen uzman bir veteriner hekim yardımcı asistanısın.

    GÖREVİN VE KURALLARIN:
    1. Kullanıcının sorusunu ve verilen KAYNAK METİNLERİ dikkatlice incele.
    2. Eğer kullanıcının sorduğu etken madde, ilaç veya terim tam olarak metinlerde geçmiyorsa veya yazım hatası içeriyorsa KESİNLİKLE "bilmiyorum" deme.
    3. Metinlerdeki anlam bakımından en yakın 3 etken maddeyi veya ilacı belirle ve kullanıcıya kibarca öneri olarak sun:
       "Aramak istediğiniz etken madde bulunamadı. Şunlardan birini mi kastettiniz?"
       - 1. [En Yakın Etken Madde 1]
       - 2. [En Yakın Etken Madde 2]
       - 3. [En Yakın Etken Madde 3]
    4. Sorulan soru net ise klinik ve farmakolojik bilgiyi eksiksiz açıkla.
    """

    try:
        prompt_content = f"KAYNAK METİNLER:\n{kitap_baglami if kitap_baglami else 'Doğrudan metin bulunamadı.'}\n\nSORU:\n{user_query}"
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": sistem_talimati},
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.2,
            max_tokens=2048
        )
        cevap = completion.choices[0].message.content
    except Exception as e:
        cevap = f"Hata oluştu: {str(e)}"

    return ChatResponse(response=cevap)

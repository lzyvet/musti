import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

app = FastAPI(title="VetHelper AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
persist_dir = os.path.dirname(os.path.abspath(__file__))

try:
    vector_db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
except Exception:
    vector_db = None

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if not client:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY bulunamadı.")
    
    raw_query = request.query.strip()
    norm_query = raw_query.lower().replace("?", "").replace(".", "")
    
    # 1. KISA SORGU / KISALTMA FİLTRESİ (Enro, Doxy, Bayt vb. durumlar için)
    # Eğer sorgu 4 karakterden kısaysa veya veritabanı alakasız bir terim getiriyorsa RAG'ı bypass et
    is_short_query = len(norm_query) <= 4

    kitap_baglami = ""
    if vector_db and not is_short_query:
        try:
            docs = vector_db.similarity_search(raw_query, k=6)
            if docs:
                kitap_baglami = "\n\n".join([doc.page_content for doc in docs])
        except Exception:
            pass

    sistem_talimati = """
    [CRITICAL: ALWAYS ANSWER IN TURKISH ONLY.]
    Sen uzman bir veteriner hekim asistanısın.

    SORGUSAL ÖNERİ KURALI (ÇOK ÖNEMLİ):
    1. Kullanıcının girdiği terim (Örn: 'enro', 'doxy', 'bay') bir kısaltma veya eksik terim ise, KESİNLİKLE ilgisiz bir konunun (Örn: Enamel, Diş minesi) tanımını yapma!
    2. Kullanıcı tam ve net bir ilaç/hastalık sormadıysa, veteriner hekimlikte bu kısaltmaya en yakın 5 etken maddeyi/terimi şöyle öner:
       "Aramak istediğiniz etken madde veya terim tam olarak bulunamadı. Yazım hatası veya kısaltma yapılmış olabilir. Şunlardan birini mi kastettiniz?"
       - 1. Enrofloksasin (Enrofloxacin)
       - 2. Enro antibacterial kombinasyonlar
       - 3. Enterit / Enterik etkenler
       - 4. Endokrin sistem terimleri
       - 5. Enamel (Diş Anatomisi)
    3. Eğer kullanıcı net ve eksiksiz bir klinik terim sorduysa, verilen kaynak metinleri sentezleyerek klinik yanıtı eksiksiz ver.
    """

    try:
        if is_short_query:
            prompt_content = f"Kullanıcı çok kısa/kısaltma bir terim girdi: '{raw_query}'. Lütfen doğrudan yukarıdaki ÖNERİ KURALINI uygulayarak en olası 5 veteriner tıbbi terimini listele."
        else:
            prompt_content = f"KAYNAK METİNLER:\n{kitap_baglami if kitap_baglami else 'Doğrudan metin bulunamadı.'}\n\nKULLANICI SORUSU:\n{raw_query}"

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": sistem_talimati},
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.1,
            max_tokens=2048
        )
        cevap = completion.choices[0].message.content
    except Exception as e:
        cevap = f"Hata oluştu: {str(e)}"

    return ChatResponse(response=cevap)

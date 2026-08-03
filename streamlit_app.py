import streamlit as st
import os
from groq import Groq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

st.set_page_config(page_title="🎯 VetHelper Ai", layout="centered")
st.title("💬 VetHelper Ai")

GROQ_API_KEY = "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"

if not GROQ_API_KEY:
    st.warning("⚠️ Lütfen geçerli bir Groq API anahtarı sağlayın.")
else:
    client = Groq(api_key=GROQ_API_KEY)

    @st.cache_resource
    def load_vector_db():
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        persist_dir = "vethelperAi/yerel_kitap_hafizasi"
        
        # Klasör yoksa oluştur
        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir)
        
        # Veritabanını yükle veya oluştur
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    # Vektör DB'yi yükle
    if "vector_db" not in st.session_state:
        try:
            st.session_state.vector_db = load_vector_db()
            doc_count = st.session_state.vector_db._collection.count()
            
            # Eğer hiç doküman yoksa, PDF'leri otomatik yükle
            if doc_count == 0:
                st.sidebar.warning("⚠️ Veritabanı boş! PDF'ler yükleniyor...")
                
                # 📂 PDF'lerin bulunduğu klasör
                pdf_klasoru = "pdf_kitap"
                
                if os.path.exists(pdf_klasoru):
                    all_docs = []
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    
                    for pdf_file in os.listdir(pdf_klasoru):
                        if pdf_file.endswith(".pdf"):
                            file_path = os.path.join(pdf_klasoru, pdf_file)
                            try:
                                with st.spinner(f"📖 {pdf_file} işleniyor..."):
                                    pdf_reader = PdfReader(file_path)
                                    text = ""
                                    for page in pdf_reader.pages:
                                        text += page.extract_text() + "\n"
                                    
                                    if text.strip():
                                        docs = text_splitter.create_documents([text])
                                        all_docs.extend(docs)
                                        st.sidebar.success(f"✅ {pdf_file} eklendi")
                            except Exception as e:
                                st.sidebar.error(f"❌ {pdf_file} hatası: {e}")
                    
                    if all_docs:
                        st.session_state.vector_db.add_documents(all_docs)
                        st.session_state.vector_db.persist()
                        doc_count = st.session_state.vector_db._collection.count()
                        st.sidebar.success(f"✅ Toplam {doc_count} doküman yüklendi!")
                else:
                    st.sidebar.error(f"❌ '{pdf_klasoru}' klasörü bulunamadı!")
            
            st.sidebar.success(f"📚 {doc_count} doküman yüklendi!")
            
        except Exception as e:
            st.error(f"❌ Kütüphane yüklenemedi: {e}")
            st.session_state.vector_db = None

    # 📂 PDF Yükleme Arayüzü (manuel ekleme için)
    with st.sidebar:
        st.header("📚 Kütüphane Yönetimi")
        st.info("PDF'leri 'pdf_kitap' klasörüne koyun veya buradan yükleyin.")
        
        uploaded_files = st.file_uploader("Yeni PDF Ekle", type=["pdf"], accept_multiple_files=True)
        
        if uploaded_files:
            all_docs = []
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            
            for pdf_file in uploaded_files:
                with st.spinner(f"📖 {pdf_file.name} işleniyor..."):
                    try:
                        pdf_reader = PdfReader(pdf_file)
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text() + "\n"
                        
                        if text.strip():
                            docs = text_splitter.create_documents([text])
                            all_docs.extend(docs)
                            st.success(f"✅ {pdf_file.name} eklendi!")
                    except Exception as e:
                        st.error(f"❌ {pdf_file.name} hatası: {e}")
            
            if all_docs:
                if st.session_state.vector_db is None:
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
                    st.session_state.vector_db = Chroma.from_documents(
                        documents=all_docs,
                        embedding=embeddings,
                        persist_directory="vethelperAi/yerel_kitap_hafizasi"
                    )
                else:
                    st.session_state.vector_db.add_documents(all_docs)
                st.session_state.vector_db.persist()
                doc_count = st.session_state.vector_db._collection.count()
                st.success(f"✅ Toplam {doc_count} doküman kaydedildi!")
                st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- SOHBET EKRANI ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])

    # --- SOHBET GİRİŞİ ---
    if user_query := st.chat_input("📚 Kitaplarla ilgili sorunuzu buraya yazın..."):
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        norm_query = user_query.strip().lower().replace("?", "").replace(".", "")
        
        # --- STATİK KİMLİK KONTROLLERİ ---
        if norm_query in ["sen nesin", "sen kimsin", "nedir bu"]:
            cevap = "Ben bir veteriner hekim/tekniker yapay zeka destek modeliyim. Sanveta tarafından 2026'da geliştirildim."
        elif any(x in norm_query for x in ["neler biliyorsun", "ne içeriyorsun", "ne konuda bilgi verebilirsin", "içerik nedir"]):
            cevap = (
                "Bu dijital kütüphane, modern veteriner hekimliğin en kritik iki damarı olan "
                "egzotik/yaban hayvanları ile kedi ve köpek (küçük hayvan) sağlığı üzerine "
                "muazzam bir uzmanlık birikimini bir araya getiriyor.\n\n"
                "Sürüngenler, kemirgenler, gelincikler ve kuşlar gibi hassas türlerin anatomik ve fizyolojik sırları, "
                "klinik yönetimleri, ilaç dozajları ve cerrahi operasyonları ile küçük hayvan hekimliğinde "
                "kardiyoloji, iç hastalıkları ve acil müdahaleler bu literatürün kapsamındadır."
            )
        else:
            # --- RAG ARAMA ---
            kitap_baglami = ""
            if st.session_state.vector_db:
                try:
                    docs = st.session_state.vector_db.similarity_search(user_query, k=8)
                    if docs:
                        kitap_baglami = "\n\n".join([doc.page_content for doc in docs])
                        st.sidebar.info(f"🔍 {len(docs)} doküman bulundu")
                    else:
                        st.sidebar.warning("⚠️ Doğrudan eşleşen doküman bulunamadı!")
                except Exception as e:
                    st.error(f"Arama hatası: {e}")

            # --- DEĞİŞTİRİLEN VE DÜZELTİLEN PROMPT BÖLÜMÜ ---
            sistem_talimati = """
            [CRITICAL: ANSWER IN TURKISH ONLY.]
            Sen uzman bir veteriner hekim yardımcı asistanısın.

            GÖREVİN VE ÖNERİ KURALLARIN:
            1. Kullanıcının sorduğu soru veya terim (Örn: 'enro') tam olarak net değilse, kısaltmaysa, harf hatası barındırıyorsa veya aranan metinlerle %100 örtüşmüyorsa KESİNLİKLE ilgisiz bir konuya (Örn: Enamel/Diş) atlayıp onun tanımını yapma ve doğrudan "bilmiyorum/kütüphanede yok" deme.
            2. Bunun yerine, kullanıcının yazmak istediği en olası 5 veteriner etken maddesini, ilacını veya tıbbi terimini analiz et ve şu şekilde öneri sun:
               "Aramak istediğiniz terim veya etken madde tam olarak bulunamadı. Yazım hatası veya kısaltma yapılmış olabilir. Şunlardan birini mi kastettiniz?"
               - 1. [En Olası Etken Madde 1 (Örn: Enrofloksasin)]
               - 2. [En Olası Etken Madde 2]
               - 3. [En Olası Etken Madde 3]
               - 4. [En Olası Etken Madde 4]
               - 5. [En Olası Etken Madde 5]
            3. Eğer kullanıcı net ve eksiksiz bir klinik soru veya etken madde sorduysa, verilen KAYNAK METİNLERİ sentezleyerek doğru ve eksiksiz Türkçe bilgi ver.
            """

            try:
                prompt_input = f"KAYNAK METİNLER:\n{kitap_baglami if kitap_baglami else 'Doğrudan metin eşleşmesi sağlanamadı.'}\n\nKULLANICI SORUSU/TERİMİ:\n{user_query}"
                
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": sistem_talimati},
                        {"role": "user", "content": prompt_input}
                    ],
                    temperature=0.2
                )
                cevap = completion.choices[0].message.content
            except Exception as e:
                cevap = f"❌ Bir hata oluştu: {str(e)}"

        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(cevap)
        st.session_state.messages.append({"role": "assistant", "content": cevap})
        st.rerun()

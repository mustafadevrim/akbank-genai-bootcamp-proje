# -*- coding: utf-8 -*-
import streamlit as st
import google.generativeai as genai
import os
import time

# Gerekli LangChain kütüphaneleri
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
# Chroma ve SelfQueryRetriever importları
from langchain_chroma import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
import sys # Hata durumunda çıkış yapmak için

# --- 1. Konfigurasyon ve API Anahtarı ---

st.set_page_config(
    page_title="Türk Mutfağı Tarifi Chatbotu",
    page_icon="🍲"
)
st.title("🍲 Türk Mutfağı Tarifi Chatbotu")
st.caption("Akbank GenAI Bootcamp Projesi - RAG Chatbot")

try:
    # API anahtarını Streamlit secrets'tan al
    if 'GOOGLE_API_KEY' not in st.secrets:
        st.error("GOOGLE_API_KEY bulunamadı. Lütfen Streamlit Secrets'a ekleyin.")
        st.stop()
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
except Exception as e:
    st.error(f"API Anahtarı yüklenirken bir hata oluştu: {e}")
    st.stop()

# --- 2. Veri Yükleme ve RAG Mimarisi Kurulumu (SelfQueryRetriever ile - TEMİZ VERİ İÇİN) ---

@st.cache_resource(show_spinner="🔄 Chatbot motoru hazırlanıyor...") # Spinner mesajını buraya taşıdık
def load_rag_pipeline_final():
    """
    TEMİZLENMİŞ veri setini yükler, RAG mimarisini (SelfQueryRetriever ve Konteks Enjeksiyonu ile) kurar.
    """
    try:
        # Adım 1: Güncel Veriyi Çekme
        if not os.path.exists("akbank-genai-bootcamp-proje"):
            st.info("Veri seti GitHub'dan ilk kez çekiliyor...")
            os.system("rm -rf akbank-genai-bootcamp-proje")
            os.system("git clone https://github.com/mustafadevrim/akbank-genai-bootcamp-proje")
        else:
            st.info("Mevcut veri seti kullanılıyor.") # Cache sayesinde tekrar çekmeye gerek yok

        # Adım 2: Konteks Enjeksiyonu ile Parçalama (Temiz Veri Formatına Güveniyoruz)
        with open("akbank-genai-bootcamp-proje/tarifler.txt", "r", encoding="utf-8") as f:
            tum_tarifler_metni = f.read()

        tarif_listesi = tum_tarifler_metni.split("\n---\n")
        documents = []
        basariyla_parcalanan_tarif_sayisi = 0

        for tarif_metni in tarif_listesi:
            if not tarif_metni.strip(): continue

            # Başlık, Malzemeler ve Yapılışı ayırma (TEMİZ formata göre)
            parts = tarif_metni.split("\nMalzemeler:\n", 1)
            if len(parts) < 2: continue # Format hatası, atla
            baslik_content = parts[0].strip()

            parts2 = parts[1].split("\nYapılışı:\n", 1)
            if len(parts2) < 2: continue # Format hatası, atla
            # MALZEMELER + ARA BÖLÜMLER (varsa) buradadır
            malzemeler_ve_arasi_content = parts2[0].strip()
            yapilis_content = parts2[1].strip()

            # 3 chunk oluşturma (Konteks Enjeksiyonu)
            doc_metadata = {"source": baslik_content}
            documents.append(Document(page_content=baslik_content, metadata=doc_metadata))
            documents.append(Document(page_content=f"{baslik_content}\nMalzemeler:\n{malzemeler_ve_arasi_content}", metadata=doc_metadata))
            documents.append(Document(page_content=f"{baslik_content}\nYapılışı:\n{yapilis_content}", metadata=doc_metadata))
            basariyla_parcalanan_tarif_sayisi += 1

        st.info(f"{basariyla_parcalanan_tarif_sayisi} tarif {len(documents)} parçaya bölündü.")
        if not documents:
            st.error("Hiçbir tarif parçalanamadı veya tarifler.txt boş.")
            return None

        # Adım 3: Embedding ve Vektör DB (Chroma)
        st.info("Embedding modeli (MiniLM) yükleniyor...")
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        # Cache folder belirtmek hataları azaltabilir
        cache_dir = os.path.join(os.getcwd(), ".cache_embedding")
        os.makedirs(cache_dir, exist_ok=True)
        embeddings = HuggingFaceEmbeddings(model_name=model_name, cache_folder=cache_dir)

        st.info("Vektör veritabanı (Chroma) oluşturuluyor...")
        vector_store = Chroma.from_documents(documents, embeddings)

        # Adım 4: Self-Querying Retriever Kurulumu
        st.info("Generation modeli (Gemini) ve Self-Querying Retriever kuruluyor...")
        # Retriever için daha az yaratıcı model
        llm_for_retriever = ChatGoogleGenerativeAI(model="models/gemini-flash-latest", temperature=0)

        metadata_field_info = [
            AttributeInfo(
                name="source",
                description="Tarifin başlığı, örneğin 'Başlık: Karnıyarık' veya 'Başlık: Menemen'",
                type="string",
            ),
        ]
        document_content_description = "Türk mutfağı yemek tarifleri"

        retriever = SelfQueryRetriever.from_llm(
            llm_for_retriever,
            vector_store,
            document_content_description,
            metadata_field_info,
            verbose=False # Logları kapalı tutalım
        )

        # Adım 5: RAG Pipeline
        # Cevap üretimi için kullanılacak LLM
        llm_for_qa = ChatGoogleGenerativeAI(model="models/gemini-flash-latest", temperature=0.7, top_p=0.85)

        rag_pipeline = RetrievalQA.from_chain_type(
            llm=llm_for_qa,
            chain_type="stuff",
            retriever=retriever # SelfQueryRetriever'ı kullan
        )

        return rag_pipeline

    except Exception as e:
        st.error(f"RAG Pipeline yüklenirken hata oluştu: {e}")
        st.exception(e)
        return None

# RAG Pipeline'ı yükle
rag_chain = load_rag_pipeline_final() # Fonksiyon adını güncelledik


# --- 3. Chat Arayüzü --- (Değişiklik yok)

if rag_chain is not None:
    st.success("✅ Chatbot hazır! Tarif sormaya başlayabilirsiniz.")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Merhaba! Hangi Türk yemeği hakkında tarif almak istersiniz?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Örn: Karnıyarık nasıl yapılır? Malzemeleri nelerdir?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Tarif aranıyor (Self-Querying ile) ve cevap oluşturuluyor..."):
                try:
                    response = rag_chain.invoke(prompt)
                    cevap = response['result']
                    st.markdown(cevap)
                    st.session_state.messages.append({"role": "assistant", "content": cevap})
                except Exception as e:
                    st.error(f"Cevap oluşturulurken bir hata oluştu: {e}")
else:
    st.error("Chatbot yüklenemedi. Lütfen API anahtarınızı kontrol edin veya sayfayı yenileyin.")

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

# --- 1. Konfigurasyon ve API Anahtarı ---

st.set_page_config(
    page_title="Türk Mutfağı Tarifi Chatbotu",
    page_icon="🍲"
)
st.title("🍲 Türk Mutfağı Tarifi Chatbotu")
st.caption("Akbank GenAI Bootcamp Projesi - RAG Chatbot")

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY bulunamadı. Lütfen Streamlit Secrets'a ekleyin.")
        st.stop()
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
except KeyError:
    st.error("GOOGLE_API_KEY bulunamadı. Lütfen Streamlit Secrets'a ekleyin.")
    st.stop()
except Exception as e:
    st.error(f"API Anahtarı yüklenirken bir hata oluştu: {e}")
    st.stop()

# --- 2. Veri Yükleme ve RAG Mimarisi Kurulumu (SelfQueryRetriever ile) ---

@st.cache_resource
def load_rag_pipeline_self_query():
    """
    Veri setini yükler, RAG mimarisini (SelfQueryRetriever ile) kurar.
    """
    try:
        with st.spinner("🔄 Veri seti GitHub'dan çekiliyor..."):
            os.system("rm -rf akbank-genai-bootcamp-proje")
            os.system("git clone https://github.com/mustafadevrim/akbank-genai-bootcamp-proje")

        with st.spinner("🥣 Tarifler yükleniyor ve KONTEKST ENJEKSİYONU ile parçalanıyor..."):
            with open("akbank-genai-bootcamp-proje/tarifler.txt", "r", encoding="utf-8") as f:
                tum_tarifler_metni = f.read()

            tarif_listesi = tum_tarifler_metni.split("\n---\n")
            documents = []
            for tarif_metni in tarif_listesi:
                if not tarif_metni.strip(): continue
                parts = tarif_metni.split("\nMalzemeler:\n", 1)
                if len(parts) < 2: continue
                baslik_content = parts[0].strip()
                parts2 = parts[1].split("\nYapılışı:\n", 1)
                if len(parts2) < 2: continue
                malzemeler_content = parts2[0].strip()
                yapilis_content = parts2[1].strip()

                # Self-Querying için metadata'yı DÜZGÜN tanımlıyoruz
                doc_metadata = {"source": baslik_content}

                documents.append(Document(page_content=baslik_content, metadata=doc_metadata))
                documents.append(Document(page_content=f"{baslik_content}\nMalzemeler:\n{malzemeler_content}", metadata=doc_metadata))
                documents.append(Document(page_content=f"{baslik_content}\nYapılışı:\n{yapilis_content}", metadata=doc_metadata))

            if not documents:
                st.error("Veri seti parçalanamadı veya tarifler.txt boş.")
                return None

        with st.spinner("🧠 Embedding modeli (MiniLM) yükleniyor..."):
            model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            embeddings = HuggingFaceEmbeddings(model_name=model_name, cache_folder='./embedding_cache')

        with st.spinner("📚 Vektör veritabanı (Chroma) oluşturuluyor..."):
            vector_store = Chroma.from_documents(documents, embeddings)

        with st.spinner("🤖 Generation modeli (Gemini) ve Self-Querying Retriever kuruluyor..."):
            # Self-Querying için daha düşük sıcaklıkta (daha az yaratıcı) bir LLM kullanmak daha iyi olabilir
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
                llm_for_retriever, # Filtre oluşturmak için kullanılacak LLM
                vector_store,
                document_content_description,
                metadata_field_info,
                verbose=False # Deploy'da logları kapatıyoruz
            )

        # Cevap üretimi için kullanılacak LLM (biraz daha yaratıcı olabilir)
        llm_for_qa = ChatGoogleGenerativeAI(model="models/gemini-flash-latest", temperature=0.7, top_p=0.85)

        # RetrievalQA zincirini SelfQueryRetriever ile kuruyoruz
        rag_pipeline = RetrievalQA.from_chain_type(
            llm=llm_for_qa, # Cevap üretecek LLM
            chain_type="stuff",
            retriever=retriever # Akıllı SelfQueryRetriever'ı kullan
        )

        return rag_pipeline

    except Exception as e:
        st.error(f"RAG Pipeline yüklenirken hata oluştu: {e}")
        st.exception(e)
        return None

# RAG Pipeline'ı yükle
with st.spinner("⏳ Chatbot hazırlanıyor (Self-Querying ile)... Lütfen bekleyin..."):
    rag_chain = load_rag_pipeline_self_query()


# --- 3. Chat Arayüzü ---

if rag_chain is not None:
    st.success("✅ Chatbot hazır! Tarif sormaya başlayabilirsiniz.")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Merhaba! Hangi Türk yemeği hakkında tarif almak istersiniz?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Örn: Künefe nasıl yapılır?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Tarif aranıyor (Self-Querying ile) ve cevap oluşturuluyor..."):
                try:
                    # RAG zincirini çalıştır (SelfQueryRetriever arka planda çalışacak)
                    response = rag_chain.invoke(prompt)
                    cevap = response['result']
                    st.markdown(cevap)
                    st.session_state.messages.append({"role": "assistant", "content": cevap})
                except Exception as e:
                    st.error(f"Cevap oluşturulurken bir hata oluştu: {e}")
else:
    st.error("Chatbot yüklenemedi. Lütfen API anahtarınızı kontrol edin veya sayfayı yenileyin.")

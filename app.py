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
    Akıllı parçalama mantığı içerir.
    """
    try:
        with st.spinner("🔄 Veri seti GitHub'dan çekiliyor..."):
            os.system("rm -rf akbank-genai-bootcamp-proje")
            os.system("git clone https://github.com/mustafadevrim/akbank-genai-bootcamp-proje")

        with st.spinner("🥣 Tarifler yükleniyor ve AKILLI PARÇALAMA ile işleniyor..."):
            with open("akbank-genai-bootcamp-proje/tarifler.txt", "r", encoding="utf-8") as f:
                tum_tarifler_metni = f.read()

            tarif_listesi = tum_tarifler_metni.split("\n---\n")
            documents = []
            basariyla_parcalanan_tarif_sayisi = 0

            for tarif_metni in tarif_listesi:
                if not tarif_metni.strip(): continue

                # Önce Başlığı Ayır
                baslik_parts = tarif_metni.split("\nMalzemeler:\n", 1)
                if len(baslik_parts) < 2:
                    st.warning(f"Format hatası (Malzemeler bulunamadı): {tarif_metni[:50]}...")
                    continue
                baslik_content = baslik_parts[0].strip()

                # Sonra Yapılışı Ayır (Metnin sonundan başlayarak)
                yapilis_parts = tarif_metni.split("\nYapılışı:\n", 1)
                if len(yapilis_parts) < 2:
                    st.warning(f"Format hatası (Yapılışı bulunamadı): {baslik_content}")
                    continue
                yapilis_content = yapilis_parts[1].strip()

                # Başlık ile Yapılışı arasında kalan her şeyi Malzemeler (+ ara bölümler) olarak al
                malzemeler_ve_arasi_content = baslik_parts[1].split("\nYapılışı:\n", 1)[0].strip()

                # Şimdi 3 chunk'ı doğru içerikle oluşturalım
                doc_metadata = {"source": baslik_content}

                # Chunk 1: Sadece Başlık
                documents.append(Document(page_content=baslik_content, metadata=doc_metadata))

                # Chunk 2: Başlık + Malzemeler (ve ara bölümler)
                documents.append(Document(
                    page_content=f"{baslik_content}\nMalzemeler:\n{malzemeler_ve_arasi_content}",
                    metadata=doc_metadata
                ))

                # Chunk 3: Başlık + Yapılışı
                documents.append(Document(
                    page_content=f"{baslik_content}\nYapılışı:\n{yapilis_content}",
                    metadata=doc_metadata
                ))
                basariyla_parcalanan_tarif_sayisi += 1

            st.info(f"{basariyla_parcalanan_tarif_sayisi} tarif başarıyla {len(documents)} parçaya bölündü.")
            if not documents:
                st.error("Hiçbir tarif parçalanamadı veya tarifler.txt boş.")
                return None

        with st.spinner("🧠 Embedding modeli (MiniLM) yükleniyor..."):
            model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            embeddings = HuggingFaceEmbeddings(model_name=model_name, cache_folder='./embedding_cache')

        with st.spinner("📚 Vektör veritabanı (Chroma) oluşturuluyor..."):
            vector_store = Chroma.from_documents(documents, embeddings)

        with st.spinner("🤖 Generation modeli (Gemini) ve Self-Querying Retriever kuruluyor..."):
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
                verbose=False
            )

        llm_for_qa = ChatGoogleGenerativeAI(model="models/gemini-flash-latest", temperature=0.7, top_p=0.85)

        rag_pipeline = RetrievalQA.from_chain_type(
            llm=llm_for_qa,
            chain_type="stuff",
            retriever=retriever
        )

        return rag_pipeline

    except Exception as e:
        st.error(f"RAG Pipeline yüklenirken hata oluştu: {e}")
        st.exception(e)
        return None

# RAG Pipeline'ı yükle
with st.spinner("⏳ Chatbot hazırlanıyor (Self-Querying & Akıllı Parçalama ile)... Lütfen bekleyin..."):
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
                    response = rag_chain.invoke(prompt)
                    cevap = response['result']
                    st.markdown(cevap)
                    st.session_state.messages.append({"role": "assistant", "content": cevap})
                except Exception as e:
                    st.error(f"Cevap oluşturulurken bir hata oluştu: {e}")
else:
    st.error("Chatbot yüklenemedi. Lütfen API anahtarınızı kontrol edin veya sayfayı yenileyin.")

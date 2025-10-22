# Akbank GenAI Bootcamp Projesi: Türk Mutfağı Tarifi Chatbotu

Bu proje, Akbank GenAI Bootcamp kapsamında geliştirilmiş RAG (Retrieval Augmented Generation) tabanlı bir chatbot uygulamasıdır. Chatbot, Türk mutfağına ait popüler yemek tarifleri hakkında sorulan soruları yanıtlamak üzere tasarlanmıştır.

## Projenin Amacı

Projenin temel amacı, belirli bir bilgi kümesi (Türk yemek tarifleri) üzerinden soruları yanıtlayabilen, RAG mimarisine dayalı bir chatbot geliştirmektir. Kullanıcılar, web arayüzü üzerinden doğal dilde sorular sorarak (örneğin, "Karnıyarık nasıl yapılır?", "Menemen malzemeleri nelerdir?") istedikleri tarif bilgilerine ulaşabilirler.

## Veri Seti Hakkında Bilgi

Projede kullanılan veri seti, `tarifler.txt` adlı dosyada yer alan 10 adet popüler Türk yemeği tarifinden oluşmaktadır. Her tarif, aşağıdaki temel formatı takip etmektedir:

* **Başlık:** Yemeğin adı (Örn: "Başlık: Menemen")
* **Malzemeler:** Gerekli malzemelerin listesi (Ara başlıklar buraya dahil edilmiştir)
* **Yapılışı:** Yemeğin hazırlanış adımları

Veri seti, RAG mimarisinin etkin çalışabilmesi için tutarlı bir formatta düzenlenmiştir.

## Kullanılan Yöntemler (Çözüm Mimarisi)

Proje, RAG mimarisi üzerine kurulmuştur ve aşağıdaki teknolojiler ve yöntemler kullanılmıştır:

* **Mimari:** Retrieval Augmented Generation (RAG)
* **Frameworkler:** LangChain (RAG pipeline'ı ve bileşen entegrasyonu için), Streamlit (Web arayüzü için).
* **Veri Yükleme ve Parçalama (Chunking):**
    * Tarifler `tarifler.txt` dosyasından yüklendi.
    * **Konteks Enjeksiyonu Stratejisi:** Her tarif, anlamsal bütünlüğü korumak ve retriever'ın doğru bilgiyi bulmasını kolaylaştırmak için 3 parçaya (chunk) ayrıldı:
        1.  Sadece Başlık
        2.  Başlık + Malzemeler (ve ara bölümler)
        3.  Başlık + Yapılışı
    * Bu parçalama stratejisi, retriever'ın yanlış tarifleri getirmesi sorununu çözmek için seçilmiştir. Her parçaya başlık bilgisinin eklenmesi, parçaların hangi tarife ait olduğunun netleşmesini sağlamıştır.
* **Embedding Modeli:** Açık kaynaklı `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` modeli kullanılarak metin parçaları vektörlere dönüştürüldü.
* **Vektör Veritabanı:** Metin vektörlerini depolamak ve anlamsal arama yapmak için açık kaynaklı `ChromaDB` kullanıldı.
* **Retrieval (Geri Getirme) Stratejisi:**
    * **Self-Querying Retriever (LangChain):** Basit vektör aramasının benzer içerikli farklı tarifleri karıştırması sorununu aşmak için LangChain'in `SelfQueryRetriever`'ı kullanıldı.
    * Bu retriever, kullanıcının sorusunu önce bir LLM'e (Gemini) göndererek sorudan bir metadata filtresi (örn: `source = "Başlık: Karnıyarık"`) oluşturur. Ardından vektör aramasını *sadece* bu filtreye uyan belgeler üzerinde yapar. Bu sayede, yalnızca ilgili tarife ait parçaların getirilmesi garanti altına alınır.
* **Generation (Üretim) Modeli:** Kullanıcının sorusuna ve retriever tarafından bulunan tarif parçalarına (bağlam) dayanarak nihai cevabı üretmek için Google'ın `Gemini-Flash` modeli (`models/gemini-flash-latest`) kullanıldı.

## Elde Edilen Sonuçlar

Geliştirilen RAG chatbot, `tarifler.txt` dosyasında bulunan 10 yemek tarifi özelinde, malzemeler ve yapılış adımları hakkındaki soruları başarıyla yanıtlayabilmektedir. Self-Querying Retriever kullanımı sayesinde, benzer içeriklere sahip olsalar bile doğru tarife ait bilgilerin getirildiği ve LLM'in bu bilgilere dayanarak tutarlı cevaplar ürettiği gözlemlenmiştir.

## Web Uygulaması Linki

Chatbot'u canlı olarak test etmek için aşağıdaki linki ziyaret edebilirsiniz:

[https://akbank-proje-mustafa.streamlit.app/](https://akbank-proje-mustafa.streamlit.app/)

## Product Kılavuzu (Nasıl Kullanılır?)

1.  Yukarıdaki linke tıklayarak web uygulamasını açın.
2.  Uygulama yüklendiğinde ("Chatbot hazır!" mesajını gördüğünüzde) alt kısımdaki sohbet kutucuğuna bilgi almak istediğiniz yemeğin tarifini sorun.
3.  Örnek Sorular:
    * "Mercimek çorbası nasıl yapılır?"
    * "Baklava malzemeleri nelerdir?"
    * "Karnıyarık nasıl yapılır?"
    * "Mercimek çorbası malzemeleri nelerdir?"
4.  Enter'a basın ve chatbot'un size tarifi bulup getirmesini bekleyin.

---

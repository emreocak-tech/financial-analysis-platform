📈 FinanScribe: Scalable Financial Analysis & ML Forecasting
FinanScribe, finansal verileri (Hisse Senetleri, Emtialar, Döviz) analiz etmek ve geleceğe yönelik fiyat tahminleri üretmek amacıyla geliştirilmiş, modüler ve genişletilebilir bir analiz platformudur. Proje, temiz kod prensipleri ve sağlam bir yazılım mimarisi üzerine inşa edilmiştir.

Uygulamanın Linki : https://ai-financial-analyzer-dao9ilbkgbbd26bmxp4zob.streamlit.app/

🎯 Projenin Amacı ve Teknik Vizyonu
Bu proje, sadece bir tahmin aracı değil; karmaşık finansal verilerin nasıl yapılandırılabileceğini, farklı tahmin modellerinin (Deep Learning & İstatistiksel) aynı çatı altında nasıl yönetilebileceğini gösteren bir mühendislik çalışmasıdır.

Temel Yetkinlikler:
Mimari Tasarım: Proje, tamamen Abstract Base Classes (ABC) kullanılarak tasarlanmıştır. Bu sayede yeni bir varlık sınıfı (örneğin Kripto Paralar) sisteme sadece birkaç satır kodla entegre edilebilir.

Hibrit ML Yaklaşımı: * PyTorch (ANN): Geçmiş verilerdeki non-lineer ilişkileri öğrenen yapay sinir ağları.

FB Prophet: Mevsimsellik (seasonality) ve tatil etkilerini hesaplayan istatistiksel modelleme.

LLM Entegrasyonu: Gemini 3-Flash API kullanılarak, teknik analiz verilerinin "Economy Professor" personasıyla anlamlı raporlara dönüştürülmesi.


🛠️ Teknoloji Yığını (Tech Stack)AlanKullanılan TeknolojilerDilPython 3.11ArayüzStreamlitMakine ÖğrenmesiPyTorch, FB ProphetVeri İşlemePandas, NumPy, YFinanceGörselleştirmeMatplotlib (Seaborn-style)KonteynırlaştırmaDockerYapay ZekaGoogle Gemini Generative AI


🏗️ Proje Yapısı ve Tasarım Desenleri
Proje, SOLID prensiplerine uygun olarak modüler bir yapıda kurgulanmıştır:

AbstractClass & AbstractOil: Polimorfizm kullanılarak tüm finansal araçlar için ortak bir arayüz tanımlandı.

Encapsulation: Veri çekme, işleme ve modelleme mantıkları birbirinden tamamen izole edildi.

Error Handling: API limitleri, dosya bulunamadı hataları ve veri uyumsuzlukları için kapsamlı try-except blokları uygulandı.


🐳 Dağıtım ve Çalıştırma
Docker ile Çalıştırma
Projeyi herhangi bir bağımlılık kurmadan doğrudan ayağa kaldırabilirsiniz:
docker build -t finanscribe .
docker run -p 8501:8501 finanscribe

Yerel Kurulum
.env dosyanızı oluşturun (API anahtarları ve dosya yolları).

pip install -r requirements.txt komutunu çalıştırın.

streamlit run ui.py ile uygulamayı başlatın.

📊 Örnek Çıktılar
Forex Analizi: PyTorch modeli ile dolar kuru olasılık hesaplamaları.

Stock Market: Prophet ile 360 günlük trend projeksiyonları.

Commodity: Altın, gümüş ve petrol için 30 günlük hareketli ortalamalar ve volatilite analizleri.

Not: Bu proje bir yatırım aracı değil, bir veri bilimi ve yazılım mimarisi örneğidir.

📈 FinanScribe: Advanced Financial Forecasting & Scalable ML Architecture
FinanScribe, modern yazılım mimarisi prensipleriyle inşa edilmiş; hisse senetleri, emtialar (Altın, Gümüş, Petrol) ve döviz kurları üzerinde derinlemesine analizler yapan bir FinTech projesidir. Proje, sadece veri görselleştirme değil, PyTorch tabanlı yapay sinir ağları ve Facebook Prophet ile zaman serisi tahmini yeteneklerini tek bir çatıda birleştirir.

Uygulamanın Linki : https://ai-financial-analyzer-dao9ilbkgbbd26bmxp4zob.streamlit.app/

🛠️ Teknik Yetkinlikler ve Mimari (Key Technical Skills)
Bu proje, bir yazılım mühendisinin sahip olması gereken temel modern yetkinlikleri sergilemek amacıyla geliştirilmiştir:

OOP & Yazılım Tasarımı: Proje tamamen Soyut Sınıflar (Abstract Base Classes) üzerine kurulmuştur. Modüler yapısı sayesinde yeni finansal araçlar (Kripto paralar vb.) sisteme minimum eforla entegre edilebilir.

Hibrit Makine Öğrenmesi (ML): * PyTorch (ANN): Geçmiş fiyat hareketlerindeki karmaşık, non-lineer ilişkileri öğrenmek için tasarlanmış özel katmanlı yapay sinir ağları.

FB Prophet: Pazar trendlerini, yıllık/aylık mevsimselliği ve tatil etkilerini analiz eden istatistiksel modeller.

LLM Entegrasyonu: Google Gemini 3-Flash API kullanılarak, elde edilen teknik verilerin doğal dil işleme (NLP) ile profesyonel bir finansal rapora dönüştürülmesi sağlanmıştır.

Versiyon Kontrolü (Git): Projenin geliştirme süreci boyunca Git aktif olarak kullanılmış; kodun gelişimi, değişiklik geçmişi ve versiyon takibi disiplinli bir şekilde yönetilmiştir.

Konteynırlaştırma: Dockerfile eklenerek projenin her ortamda (Local, Cloud, VPS) izole ve hatasız çalışması garanti altına alınmıştır.

🚀 Teknoloji Yığını (Tech Stack)
Dil: Python 3.11

UI Framework: Streamlit

Analiz & Tahmin: PyTorch, Prophet, Pandas, NumPy, YFinance

Yapay Zeka: Gemini AI API

DevOps: Docker, Git (Version Control), Streamlit Cloud

📦 Kurulum ve Çalıştırma
Docker ile Hızlı Başlatma
Projenin tüm bağımlılıklarıyla birlikte tek komutla çalışması için:

Bash
docker build -t finanscribe .
docker run -p 8501:8501 finanscribe
Yerel Geliştirme Ortamı
Depoyu klonlayın: git clone <repo-url>

Bağımlılıkları yükleyin: pip install -r requirements.txt

.env dosyasını oluşturun ve gerekli API anahtarlarını tanımlayın.

Uygulamayı başlatın: streamlit run ui.py

📂 Proje Yapısı
AbstractClass.py: Finansal araçlar için standart arayüzü (blueprint) tanımlar.

stock_market.py & crude_oil.py ...: Veri çekme ve modelleme mantığının izole edildiği sınıflar.

ui.py: Kullanıcı etkileşimi ve görselleştirme katmanı.

Dockerfile: Uygulamanın konteynır mimarisi.

⚖️ Yasal Uyarı
Bu proje teknik bir portfolyo çalışmasıdır. İçerdiği tahminler ve analizler kesinlikle yatırım tavsiyesi değildir. Finansal kayıplardan uygulama geliştiricisi sorumlu tutulamaz.

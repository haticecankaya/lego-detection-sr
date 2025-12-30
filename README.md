# 🧱 Lego Nesne Tespiti ve Görüntü Netleştirme

Bu proje, düşük çözünürlüklü veya geniş açılı görüntülerdeki küçük Lego parçalarını tespit etmek ve bu parçaların görüntü kalitesini yapay zeka destekli yöntemlerle artırmak amacıyla geliştirilmiştir.

---

## 📋 1. Proje Hakkında ve Amaç

Siber güvenlik ve görüntü işleme alanlarında, net olmayan görüntülerden bilgi çıkarımı kritik bir öneme sahiptir. Bu proje, **nesne tespiti (Object Detection)** ve **süper çözünürlük (Super Resolution)** teknolojilerini birleştirerek aşağıdaki problemleri çözmeyi hedefler:

1.  Karmaşık arka planlarda küçük nesnelerin (Lego parçaları) tespiti.
2.  Tespit edilen düşük çözünürlüklü parçaların detaylarının geri kazanılması.
3.  Kullanıcı dostu bir arayüz ile analiz süreçlerinin kolaylaştırılması.

**Kullanılan Temel Teknolojiler:**
* **YOLOv8:** Hızlı ve hassas nesne tespiti için.
* **Real-ESRGAN:** Görüntü netleştirme ve çözünürlük artırma (4x upscaling) için.
* **Streamlit:** İnteraktif web arayüzü için.

---

## 🏗️ 2. Proje Mimarisi ve Dosya Yapısı

Proje, yönetilebilirliği ve ölçeklenebilirliği artırmak amacıyla **modüler mimari** prensiplerine göre tasarlanmıştır.


### Veri Akış Diyagramı
1.  **Girdi:** Kullanıcı Streamlit üzerinden resim yükler.
2.  **Tespit:** Görüntü `src/model_utils.py` üzerinden YOLOv8 modeline gönderilir.
3.  **Kırpma:** Tespit edilen koordinatlara (Bounding Box) göre nesneler kırpılır.
4.  **İyileştirme:** Kırpılan her parça RealESRGAN modeline sokularak 4 kat büyütülür.
5.  **Çıktı:** Orijinal ve netleştirilmiş görüntüler yan yana kullanıcıya sunulur.

---

##⚙️ 3. Kurulum Talimatları

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin:

### Gereksinimler
* Python 3.8 veya üzeri
* Git

### Adım 1: Depoyu Klonlayın
```bash
git clone [https://github.com/kullanici_adi/proje_adi.git](https://github.com/kullanici_adi/proje_adi.git)
cd proje_adi
```
### Adım 2: Sanal Ortam Oluşturun
```bash
python -m venv venv
venv\Scripts\activate
```
### Adım 3: Kütüphaneleri Yükleyin
```bash
pip install -r requirements.txt
```
## Adım 4: Model Dosyalarını Yerleştirin
Projenin çalışması için gerekli olan önceden eğitilmiş model ağırlıklarının `models/` klasörü altında bulunduğundan emin olun:
* **best.pt:** Lego parçalarını tanımak için eğitilmiş YOLOv8 modeli.
* **RealEsrgan_x4plus.pth:** Tespit edilen parçaları netleştirmek için kullanılan Süper Çözünürlük (SR) modeli.

---

## 🚀 3. Kullanım Talimatları

Uygulamayı başlatmak için terminalde proje ana dizinindeyken şu komutu çalıştırın:

```bash
streamlit run app.py
```
---

## 🧪 4. Test Senaryoları

Projenin kararlılığını ve modüler yapısını doğrulamak için `pytest` kütüphanesi kullanılmıştır. Yazılan birim testler (unit tests); model yükleme süreçlerini, hata yönetimini ve görüntü işleme fonksiyonlarını kontrol ederek yazılımın güvenilirliğini sağlar.

### Testleri Çalıştırma
Proje ana dizinindeyken aşağıdaki komutu terminale yazarak tüm testleri başlatabilirsiniz:

```bash
pytest tests/
```
---

## 📝 Lisans ve Katkı

* **Geliştirici:** Hatice Çankaya
* **Tarih:** Aralık 2025


---
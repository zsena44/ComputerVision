# MNIST El Yazısı Rakam Tanıma – CNN (Computer Vision Ödevi)

PyTorch ile eğitilmiş bir evrişimli sinir ağı (CNN), MNIST veri setinde 0–9 rakamlarını sınıflandırır.
Model Gradio arayüzüyle uygulama hâline getirilmiş ve telefonla çekilmiş gerçek fotoğraflarla test edilmiştir.

## Dosyalar
| Dosya | Açıklama |
|---|---|
| `model.py` | CNN mimarisi (`MnistCNN`) |
| `preprocess.py` | Telefon fotoğrafını 28x28 MNIST formatına çeviren ön işleme |
| `train.py` | Eğitim (54k/6k/10k ayrımı), test metrikleri, grafikler |
| `test_real_photos.py` | `fotolar/` klasöründeki gerçek fotoğrafları test eder |
| `app.py` | Gradio web uygulaması (fotoğraf yükleme + çizim tuvali) |
| `mnist_cnn.pt` | Eğitilmiş model ağırlıkları |
| `outputs/` | Metrikler, grafikler, karışıklık matrisi, gerçek foto sonuçları |
| `Rapor_MNIST_CNN.docx` | Proje raporu |

## Çalıştırma
```bash
pip install -r requirements.txt
python train.py                 # eğitim + değerlendirme (MNIST otomatik indirilir)
python test_real_photos.py      # gerçek fotoğraf testi
python app.py                   # http://127.0.0.1:7860 adresinde arayüz
```
Fotoğraf dosya adının ilk karakteri gerçek etikettir (ör. `4_numara6.jpg` → 4).

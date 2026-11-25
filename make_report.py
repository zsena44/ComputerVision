"""Ödev şablonunu temel alarak dolu proje raporunu (Rapor_MNIST_CNN.docx) üretir."""
import json, os, sys
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH

TEMPLATE = sys.argv[1] if len(sys.argv) > 1 else "CNN_ Computer Vision Odevi.docx"
m = json.load(open("outputs/metrics.json"))
real = json.load(open("outputs/real_photo_results.json", encoding="utf-8"))
comments = json.load(open("outputs/real_photo_comments.json", encoding="utf-8"))

doc = Document(TEMPLATE)
body = doc.element.body
for el in list(body):              # şablonun stillerini koru, içeriği temizle
    if not el.tag.endswith("sectPr"):
        body.remove(el)
STYLES = [s.name for s in doc.styles]


def para(text, bold=False, size=None, align=None, italic=False):
    p = doc.add_paragraph()
    r = p.add_run(text); r.bold = bold; r.italic = italic
    if size: r.font.size = Pt(size)
    if align is not None: p.alignment = align
    return p


def bullet(text, label=None):
    p = doc.add_paragraph("•  ")
    p.paragraph_format.left_indent = Cm(0.6)
    p.paragraph_format.first_line_indent = Cm(-0.4)
    if label:
        p.add_run(label).bold = True
    p.add_run(text)


def figure(path, caption, width=16):
    doc.add_picture(path, width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    para(caption, italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)


def table(rows, header=True, widths=None):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    if "Table Grid" in STYLES:
        t.style = "Table Grid"
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j); c.text = ""
            run = c.paragraphs[0].add_run(str(val)); run.font.size = Pt(9.5)
            if header and i == 0: run.bold = True
            if widths: c.width = Cm(widths[j])
    doc.add_paragraph()
    return t


def pct(v):
    return f"%{v * 100:.2f}".replace(".", ",")


params = f"{m['params']:,}".replace(",", ".")
para("COMPUTER VISION ÖDEVİ", bold=True, size=18, align=WD_ALIGN_PARAGRAPH.CENTER)
para("Derin Öğrenme Uygulaması – El Yazısı Rakam Tanıma (MNIST + CNN)", bold=True, size=13, align=WD_ALIGN_PARAGRAPH.CENTER)
para("Öğrenci: ______________________     Numara: ______________", align=WD_ALIGN_PARAGRAPH.CENTER)

doc.add_heading("1. Ödevin Amacı", 1)
para("Bu çalışmada açık kaynaklı MNIST veri seti kullanılarak el yazısı rakamları (0–9) sınıflandıran bir evrişimli "
     "sinir ağı (CNN) geliştirilmiş, modelin başarımı test kümesinde ölçülmüş, model Gradio ile çalışan bir web "
     "uygulamasına dönüştürülmüş ve telefonla çekilmiş gerçek rakam fotoğraflarıyla denenmiştir.")

doc.add_heading("2. Görev (Task)", 1)
para("Seçilen problem: Görüntü sınıflandırma (image classification). Girdi olarak tek bir el yazısı rakam "
     "görüntüsü alınır; çıktı olarak 10 sınıftan (0, 1, …, 9) biri ve her sınıfın olasılığı üretilir.")

doc.add_heading("3. Model (Architecture)", 1)
para("Problem küçük boyutlu (28×28) gri seviye görüntülerden oluştuğu için, LeNet fikrini geliştiren, VGG tarzı "
     f"(art arda 3×3 evrişim) iki bloklu bir CNN tasarlanmıştır. Toplam parametre sayısı {params}'dir.")
table([
    ["Katman", "Ayrıntı", "Çıktı boyutu"],
    ["Girdi", "1 kanal, gri seviye, normalize edilmiş", "1×28×28"],
    ["Blok 1", "Conv3×3(32) → BN → ReLU → Conv3×3(32) → BN → ReLU → MaxPool2×2 → Dropout(0.25)", "32×14×14"],
    ["Blok 2", "Conv3×3(64) → BN → ReLU → Conv3×3(64) → BN → ReLU → MaxPool2×2 → Dropout(0.25)", "64×7×7"],
    ["Sınıflandırıcı", "Flatten → Linear(3136→256) → BN → ReLU → Dropout(0.5)", "256"],
    ["Çıkış", "Linear(256→10) → Softmax", "10"],
], widths=[3, 10, 3])
para("Batch Normalization eğitimi hızlandırır ve kararlı hâle getirir; Dropout aşırı öğrenmeyi (overfitting) "
     "azaltır; MaxPool uzamsal boyutu küçülterek küçük kaymalara karşı dayanıklılık sağlar.")

doc.add_heading("4. Veri Seti (Dataset)", 1)
para("MNIST: 28×28 piksel, gri seviye, 70.000 el yazısı rakam görüntüsü (10 sınıf, yaklaşık dengeli). "
     "Veri seti torchvision.datasets.MNIST ile otomatik indirilmiştir. Kümeler şu şekilde ayrılmıştır:")
table([
    ["Küme", "Görüntü sayısı", "Kaynak"],
    ["Eğitim (train)", "54.000", "Resmî eğitim kümesinin %90'ı (seed=42 ile rastgele)"],
    ["Doğrulama (validation)", "6.000", "Resmî eğitim kümesinin %10'u – en iyi model seçimi için"],
    ["Test", "10.000", "Resmî test kümesi – yalnızca son değerlendirmede kullanıldı"],
], widths=[4, 3, 9])
para("Ön işleme: Pikseller [0,1] aralığına ölçeklenip MNIST ortalaması (0.1307) ve standart sapması (0.3081) ile "
     "normalize edilmiştir. Eğitimde gerçek fotoğraflara genellemeyi artırmak için veri artırma (RandomAffine: "
     "±12° döndürme, %10 kaydırma, 0.85–1.15 ölçek, 8° shear) uygulanmıştır.")

doc.add_heading("5. Model Ayarları ve Eğitim", 1)
table([
    ["Ayar", "Değer", "Gerekçe"],
    ["Learning rate", f"Başlangıç {m['lr']}, OneCycleLR ile en fazla {m['max_lr']}",
     "OneCycle önce LR'yi artırıp sonra düşürür; hızlı ve kararlı yakınsama sağlar"],
    ["Batch size", str(m["batch_size"]), "CPU'da hız ile gradyan kararlılığı arasında iyi bir denge"],
    ["Epoch", str(m["epochs"]), "Doğrulama doğruluğu bu noktada doymaktadır (bkz. Şekil 1)"],
    ["Optimizer", f"AdamW (weight decay = {m['weight_decay']})",
     "Adam'ın uyarlanabilir öğrenme hızı + ayrık ağırlık çürümesi ile düzenlileştirme"],
    ["Aktivasyon", "ReLU (gizli katmanlar)", "Basit, hızlı ve kaybolan gradyan sorununu azaltır"],
    ["Çıkış fonksiyonu", "Softmax (10 sınıf)", "Sınıf olasılıklarını üretir, toplamları 1'dir"],
    ["Loss", "Cross-Entropy Loss", "Çok sınıflı sınıflandırma için standart kayıp; softmax ile birlikte çalışır"],
    ["Düzenlileştirme", "Dropout 0.25/0.5, BatchNorm, veri artırma", "Aşırı öğrenmeyi engellemek için"],
    ["Model seçimi", "En yüksek doğrulama doğruluğu veren epoch", "Test kümesi seçimde kullanılmadı"],
], widths=[3.2, 5, 7.8])
figure("outputs/training_curves.png", "Şekil 1. Eğitim ve doğrulama loss / accuracy eğrileri")
para("Eğitim doğruluğunun doğrulama doğruluğundan biraz düşük görünmesinin nedeni, eğitimde veri artırma ve "
     "Dropout'un etkin olmasıdır; doğrulama değerlendirmesi ise artırmasız ve Dropout kapalı yapılır. İki eğri "
     "arasında ayrışma olmaması aşırı öğrenme olmadığını gösterir.")

doc.add_heading("6. Performans Değerlendirmesi", 1)
para("Sınıflandırma problemi olduğu için Accuracy, Precision, Recall ve F1-score (10 sınıf üzerinden makro "
     "ortalama) kullanılmıştır. Sonuçlar 10.000 görüntülük, eğitimde hiç görülmemiş test kümesine aittir.")
table([
    ["Metrik", "Değer"],
    ["Test Accuracy", pct(m["test_accuracy"])],
    ["Precision (macro)", pct(m["precision_macro"])],
    ["Recall (macro)", pct(m["recall_macro"])],
    ["F1-score (macro)", pct(m["f1_macro"])],
    ["En iyi doğrulama accuracy", pct(m["best_val_accuracy"])],
    ["Test loss", f"{m['test_loss']:.4f}"],
    ["Toplam eğitim süresi", f"{m['train_time_sec']:.0f} sn (≈ {m['train_time_sec'] / 60:.1f} dk, {m['epochs']} epoch, CPU)"],
    ["Tahmin süresi – 10.000 test görüntüsü", f"{m['test_10k_inference_sec']:.2f} sn"],
    ["Tahmin süresi – tek görüntü", f"{m['single_image_inference_ms']:.2f} ms"],
], widths=[7, 7])
cls_rows = [["Sınıf", "Precision", "Recall", "F1", "Örnek"]]
for line in open("outputs/classification_report.txt").read().splitlines():
    parts = line.split()
    if len(parts) == 5 and parts[0].isdigit():
        cls_rows.append(parts)
para("Sınıf bazında sonuçlar:", bold=True)
table(cls_rows, widths=[2.5, 3, 3, 3, 2.5])
figure("outputs/confusion_matrix.png", "Şekil 2. Test kümesi karışıklık matrisi (confusion matrix)", width=12)
figure("outputs/misclassified.png", "Şekil 3. Yanlış sınıflandırılan bazı test örnekleri")
para(comments["test_errors"])

doc.add_heading("7. Uygulama (Deployment)", 1)
para("Eğitilen model Gradio ile bir web uygulamasına dönüştürülmüştür (app.py). Uygulama çalıştırıldığında "
     "http://127.0.0.1:7860 adresinde açılır ve iki giriş yöntemi sunar:")
bullet(" Telefon/bilgisayardan rakam fotoğrafı yükleme.", "Fotoğraf yükle:")
bullet(" Tarayıcıdaki tuvale fare ile rakam çizme.", "Çiz:")
para("Çıktı olarak en olası 3 sınıf ve olasılıkları ile modele verilen 28×28'lik ön işlenmiş görüntü gösterilir. "
     "Fotoğraflar MNIST formatına şu adımlarla dönüştürülür (preprocess.py): gri seviyeye çevirme → Gauss "
     "bulanıklaştırma → Otsu eşikleme ve renk tersleme (MNIST'te rakam beyaz, zemin siyahtır) → morfolojik "
     "kapama → bağlı bileşen analiziyle kenardaki leke/gölgelerin atılması → rakamın kırpılıp en-boy oranı "
     "korunarak 20×20'ye sığdırılması → 28×28 tuvale yerleştirilip kütle merkezine göre ortalanması.")
if os.path.exists("outputs/app_screenshot.png"):
    figure("outputs/app_screenshot.png", "Şekil 4. Gradio uygulaması: solda fotoğraf girişi, sağda tahmin (6, %100) ve modele verilen 28×28 görüntü", width=15)

doc.add_heading("8. Gerçek Görüntülerle Test ve Değerlendirme", 1)
n = len(real["results"]); ok = sum(r["dogru"] for r in real["results"])
para(f"Telefon kamerasıyla beyaz kâğıda kalemle yazılmış {n} farklı rakam fotoğrafı çekilmiş ve uygulamadaki "
     "ön işleme + model hattından geçirilmiştir. Fotoğraflar farklı kalem renkleri (siyah ve mavi), çizgi "
     f"kalınlıkları ve çözünürlükler içermektedir. Sonuç: {ok}/{n} doğru (%{real['accuracy'] * 100:.0f}).")
figure("outputs/real_photo_results.png",
       "Şekil 5. Gerçek fotoğraflar: orijinal görüntü, modele verilen 28×28 girdi ve sınıf olasılıkları", width=10.5)
rows = [["Fotoğraf", "Gerçek", "Tahmin", "Güven", "Sonuç", "İlk 3 tahmin"]]
for r in real["results"]:
    rows.append([r["dosya"], r["gercek"], r["tahmin"], f"%{r['guven'] * 100:.1f}".replace(".", ","), "Doğru" if r["dogru"] else "Yanlış",
                 ", ".join(f"{c} (%{p * 100:.1f})".replace(".", ",") for c, p in r["top3"])])
table(rows, widths=[3.2, 1.5, 1.5, 1.8, 1.8, 6.2])
para("Başarılı durumların yorumu", bold=True)
for c in comments["points"]:
    bullet(" " + c["text"], c["label"])

para("Zorlayıcı testler – başarısız durumlar", bold=True)
para("Tüm fotoğraflar doğru sınıflandırıldığı için modelin sınırlarını görmek amacıyla aynı fotoğraflar dört farklı "
     "koşulda tekrar modele verilmiştir: (1) ön işleme yapılmadan, yalnızca gri seviyeye çevirip 28×28'e "
     "küçülterek; (2) 45°, (3) 90° ve (4) 180° döndürülerek.")
figure("outputs/stress_tests.png", "Şekil 6. Zorlayıcı testler (yeşil = doğru, kırmızı = yanlış)", width=16)
names = {"naive": "Ön işlemesiz (ham)", "rot45": "45° döndürülmüş", "rot90": "90° döndürülmüş", "rot180": "180° döndürülmüş"}
srows = [["Koşul", "Doğruluk", "Tahminler (gerçek → tahmin)"]]
for k, v in real["stress"].items():
    srows.append([names[k], f"{sum(r['dogru'] for r in v)}/{len(v)}",
                  ", ".join(f"{r['gercek']}→{r['tahmin']}" for r in v)])
table(srows, widths=[4, 2.2, 9.8])
for c in comments["failures"]:
    bullet(" " + c["text"], c["label"])
para(comments["summary"])

doc.add_heading("9. Proje Raporu", 1)
table([
    ["Alan", "Açıklama / Öğrenci yanıtı"],
    ["Project Title", "CNN ile El Yazısı Rakam Tanıma (MNIST)"],
    ["Problem / Task", "Görüntü sınıflandırma – 10 sınıflı (0–9) el yazısı rakam tanıma"],
    ["Dataset", "MNIST – 70.000 adet 28×28 gri seviye görüntü; 54.000 eğitim / 6.000 doğrulama / 10.000 test"],
    ["Dataset URL", "https://yann.lecun.com/exdb/mnist/  (torchvision.datasets.MNIST ile indirildi)"],
    ["Model Architecture", "Özel CNN (VGG tarzı, 2 blok): [Conv-BN-ReLU]×2-MaxPool-Dropout (32 ve 64 filtre) "
                           f"+ FC(256) + FC(10); {params} parametre"],
    ["Input Size", "1×28×28 (gri seviye)"],
    ["Hyperparameters", f"LR = {m['lr']} (OneCycleLR, max {m['max_lr']}), batch size = {m['batch_size']}, "
                        f"epoch = {m['epochs']}, weight decay = {m['weight_decay']}, dropout = 0.25/0.5, veri artırma (RandomAffine)"],
    ["Optimizer", "AdamW"],
    ["Activation / Output Function", "ReLU (gizli katmanlar) / Softmax (çıkış)"],
    ["Loss Function", "Cross-Entropy Loss"],
    ["Evaluation Metrics", "Accuracy, Precision, Recall, F1-score (macro), Confusion Matrix, eğitim ve tahmin süresi"],
    ["Training Environment", f"Python {m['python']}, PyTorch {m['torch']}, {m['device'].upper()}, Windows 11"],
    ["Best Performance", f"Test Accuracy {pct(m['test_accuracy'])}, F1 {pct(m['f1_macro'])}, Precision "
                         f"{pct(m['precision_macro'])}, Recall {pct(m['recall_macro'])}; gerçek fotoğraflarda {ok}/{n}"],
    ["Inference Time", f"Tek görüntü ≈ {m['single_image_inference_ms']:.2f} ms (CPU); 10.000 test görüntüsü "
                       f"{m['test_10k_inference_sec']:.2f} sn; eğitim süresi ≈ {m['train_time_sec'] / 60:.1f} dk"],
    ["Deployment Technology", "Gradio (fotoğraf yükleme + çizim tuvali), python app.py"],
    ["Repository", "MNIST_CNN_Odev klasörü (model.py, preprocess.py, train.py, test_real_photos.py, app.py, "
                   "mnist_cnn.pt, outputs/) – GitHub linki: ______________"],
], widths=[4.5, 11.5])

doc.add_heading("Sonuç", 1)
para(f"Geliştirilen CNN, MNIST test kümesinde {pct(m['test_accuracy'])} doğruluğa ulaşmış ve tek bir görüntüyü CPU "
     f"üzerinde yaklaşık {m['single_image_inference_ms']:.1f} ms'de sınıflandırmaktadır. Uygun bir ön işleme hattıyla "
     f"birlikte telefonla çekilen gerçek fotoğraflarda {ok}/{n} başarı elde edilmiştir. Gerçek dünyada başarıyı "
     "etkileyen başlıca etkenler arka plan/aydınlatma, kalem kalınlığı ve rakamın yazım stilinin MNIST'ten farkıdır.")
doc.save("Rapor_MNIST_CNN.docx")
print("Rapor_MNIST_CNN.docx oluşturuldu")

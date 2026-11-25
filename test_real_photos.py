"""Telefonla çekilen rakam fotoğraflarını modele verir; sonuçları görsel ve tablo olarak kaydeder.
Dosya adının ilk karakteri gerçek etikettir (ör. 4_numara6.jpg -> 4)."""
import json, os, sys, time
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from model import MnistCNN
from preprocess import photo_to_mnist, to_tensor

PHOTO_DIR = sys.argv[1] if len(sys.argv) > 1 else "fotolar"
OUT = "outputs"
model = MnistCNN(); model.load_state_dict(torch.load("mnist_cnn.pt", map_location="cpu")); model.eval()

files = sorted(f for f in os.listdir(PHOTO_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
rows = []
fig, axes = plt.subplots(len(files), 3, figsize=(10, 2.8 * len(files)), gridspec_kw={"width_ratios": [1, 1, 1.6]})
for ax, f in zip(axes, files):
    img = ImageOps.exif_transpose(Image.open(os.path.join(PHOTO_DIR, f))).convert("RGB")
    true = int(f[0])
    t0 = time.perf_counter()
    arr = photo_to_mnist(img)
    with torch.no_grad():
        prob = torch.softmax(model(to_tensor(arr)), 1)[0]
    ms = (time.perf_counter() - t0) * 1000
    pred = int(prob.argmax()); top3 = prob.topk(3)
    rows.append({"dosya": f, "gercek": true, "tahmin": pred, "guven": round(float(prob[pred]), 4),
                 "dogru": pred == true, "sure_ms": round(ms, 2),
                 "top3": [(int(i), round(float(v), 4)) for v, i in zip(top3.values, top3.indices)]})
    ok = pred == true
    ax[0].imshow(img); ax[0].set_title(f"Orijinal: {f}", fontsize=9); ax[0].axis("off")
    ax[1].imshow(arr, cmap="gray"); ax[1].set_title("Model girdisi (28x28)", fontsize=9); ax[1].axis("off")
    colors = ["#2a9d8f" if i == pred and ok else "#e63946" if i == pred else "#adb5bd" for i in range(10)]
    ax[2].bar(range(10), prob.numpy(), color=colors); ax[2].set_xticks(range(10)); ax[2].set_ylim(0, 1)
    ax[2].set_title(f"Gerçek: {true} | Tahmin: {pred} ({prob[pred]:.1%}) {'✓' if ok else '✗'}",
                    fontsize=10, color="#2a9d8f" if ok else "#e63946")
fig.tight_layout(); fig.savefig(f"{OUT}/real_photo_results.png", dpi=130)


def paper_color(img):
    """Döndürmede açılan köşeleri kağıt rengiyle doldurmak için kenar piksellerinin medyanı."""
    a = np.array(img)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    return tuple(int(v) for v in np.median(border, axis=0))


def predict_arr(arr):
    with torch.no_grad():
        prob = torch.softmax(model(to_tensor(arr)), 1)[0]
    return int(prob.argmax()), float(prob.max())


# Zorlayıcı deneyler: (a) ön işleme olmadan ham fotoğraf, (b) fotoğrafın döndürülmesi
stress = {"naive": [], "rot45": [], "rot90": [], "rot180": []}
fig, axes = plt.subplots(4, len(files), figsize=(2 * len(files), 8.8))
for j, f in enumerate(files):
    img = ImageOps.exif_transpose(Image.open(os.path.join(PHOTO_DIR, f))).convert("RGB")
    true = int(f[0])
    variants = {
        "naive": np.array(img.convert("L").resize((28, 28)), dtype=np.float32) / 255.0,  # tersleme/kırpma yok
        "rot45": photo_to_mnist(img.rotate(45, expand=True, fillcolor=paper_color(img))),
        "rot90": photo_to_mnist(img.transpose(Image.Transpose.ROTATE_90)),
        "rot180": photo_to_mnist(img.transpose(Image.Transpose.ROTATE_180)),
    }
    for i, (k, arr) in enumerate(variants.items()):
        pred, conf = predict_arr(arr)
        stress[k].append({"dosya": f, "gercek": true, "tahmin": pred, "guven": round(conf, 4), "dogru": pred == true})
        a = axes[i, j]; a.imshow(arr, cmap="gray"); a.axis("off")
        a.set_title(f"G:{true} T:{pred} ({conf:.0%})", fontsize=9, color="#2a9d8f" if pred == true else "#e63946")
for i, name in enumerate(["Ön işlemesiz (ham)", "45° döndürülmüş", "90° döndürülmüş", "180° döndürülmüş"]):
    axes[i, 0].text(-0.15, 0.5, name, transform=axes[i, 0].transAxes, rotation=90, va="center", ha="right", fontsize=10)
fig.suptitle("Zorlayıcı testler: model girdileri ve tahminler (G=Gerçek, T=Tahmin)")
fig.tight_layout(); fig.savefig(f"{OUT}/stress_tests.png", dpi=130)

acc = sum(r["dogru"] for r in rows) / len(rows)
stress_acc = {k: sum(r["dogru"] for r in v) / len(v) for k, v in stress.items()}
json.dump({"accuracy": acc, "results": rows, "stress_accuracy": stress_acc, "stress": stress},
          open(f"{OUT}/real_photo_results.json", "w"), indent=2, ensure_ascii=False)
for k, v in stress.items():
    print(f"{k:7s} acc={stress_acc[k]:.2f} ", [(r["gercek"], r["tahmin"]) for r in v])
for r in rows:
    print(f"{r['dosya']:16s} gerçek={r['gercek']} tahmin={r['tahmin']} güven={r['guven']:.3f} {'DOĞRU' if r['dogru'] else 'YANLIŞ'}  top3={r['top3']}")
print(f"Gerçek fotoğraf doğruluğu: {acc:.2%}")

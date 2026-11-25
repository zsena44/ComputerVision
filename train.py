"""MNIST üzerinde CNN eğitimi, değerlendirme ve grafiklerin üretilmesi."""
import json, os, platform, time
import numpy as np
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)
from model import MnistCNN
from preprocess import MNIST_MEAN, MNIST_STD

SEED, BATCH, EPOCHS, LR, WD = 42, 128, 12, 1e-3, 1e-4
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(SEED); np.random.seed(SEED)
torch.set_num_threads(os.cpu_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

norm = transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
train_tf = transforms.Compose([   # gerçek fotoğraflara genelleme için hafif veri artırma
    transforms.RandomAffine(degrees=12, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=8),
    transforms.ToTensor(), norm])
eval_tf = transforms.Compose([transforms.ToTensor(), norm])

full_train = datasets.MNIST("data", train=True, download=True, transform=train_tf)
full_train_eval = datasets.MNIST("data", train=True, download=True, transform=eval_tf)
test_set = datasets.MNIST("data", train=False, download=True, transform=eval_tf)

# 60.000 eğitim görüntüsü -> 54.000 eğitim + 6.000 doğrulama; 10.000 test ayrı
g = torch.Generator().manual_seed(SEED)
train_idx, val_idx = random_split(range(len(full_train)), [54000, 6000], generator=g)
train_set = torch.utils.data.Subset(full_train, list(train_idx))
val_set = torch.utils.data.Subset(full_train_eval, list(val_idx))

train_dl = DataLoader(train_set, BATCH, shuffle=True)
val_dl = DataLoader(val_set, 512)
test_dl = DataLoader(test_set, 512)

model = MnistCNN().to(device)
n_params = sum(p.numel() for p in model.parameters())
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-3, epochs=EPOCHS, steps_per_epoch=len(train_dl))


def evaluate(dl):
    model.eval(); loss, preds, labels = 0.0, [], []
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss += criterion(out, y).item() * len(y)
            preds.append(out.argmax(1).cpu()); labels.append(y.cpu())
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return loss / len(labels), accuracy_score(labels, preds), preds, labels


history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}
best_val, t0 = 0.0, time.time()
for epoch in range(1, EPOCHS + 1):
    model.train(); tl, tc, n = 0.0, 0, 0
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward(); optimizer.step(); scheduler.step()
        tl += loss.item() * len(y); tc += (out.argmax(1) == y).sum().item(); n += len(y)
    vl, va, _, _ = evaluate(val_dl)
    for k, v in zip(history, [tl / n, tc / n, vl, va]):
        history[k].append(v)
    if va > best_val:
        best_val = va
        torch.save(model.state_dict(), "mnist_cnn.pt")
    print(f"Epoch {epoch:2d}/{EPOCHS} | train loss {tl/n:.4f} acc {tc/n:.4f} | val loss {vl:.4f} acc {va:.4f} | {time.time()-t0:.0f}s", flush=True)
train_time = time.time() - t0

# En iyi doğrulama modeliyle test
model.load_state_dict(torch.load("mnist_cnn.pt"))
t1 = time.time()
test_loss, test_acc, preds, labels = evaluate(test_dl)
batch_infer = time.time() - t1
p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")

# Tek görüntü tahmin süresi (gecikme)
model.eval(); x1 = test_set[0][0].unsqueeze(0).to(device)
with torch.no_grad():
    for _ in range(20): model(x1)
    t2 = time.perf_counter()
    for _ in range(500): model(x1)
single_ms = (time.perf_counter() - t2) / 500 * 1000

report = classification_report(labels, preds, digits=4)
print(report)
metrics = {
    "test_accuracy": test_acc, "precision_macro": p, "recall_macro": r, "f1_macro": f1,
    "test_loss": test_loss, "best_val_accuracy": best_val,
    "train_time_sec": train_time, "test_10k_inference_sec": batch_infer,
    "single_image_inference_ms": single_ms, "params": n_params,
    "epochs": EPOCHS, "batch_size": BATCH, "lr": LR, "max_lr": 3e-3, "weight_decay": WD,
    "device": str(device), "cpu": platform.processor(), "torch": torch.__version__,
    "python": platform.python_version(), "history": history,
}
json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=2)
open(f"{OUT}/classification_report.txt", "w").write(report)

# Grafikler
ep = range(1, EPOCHS + 1)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ep, history["train_loss"], "o-", label="Eğitim"); ax[0].plot(ep, history["val_loss"], "o-", label="Doğrulama")
ax[0].set(title="Loss", xlabel="Epoch"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(ep, history["train_acc"], "o-", label="Eğitim"); ax[1].plot(ep, history["val_acc"], "o-", label="Doğrulama")
ax[1].set(title="Accuracy", xlabel="Epoch"); ax[1].legend(); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/training_curves.png", dpi=150)

fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(confusion_matrix(labels, preds)).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Test Confusion Matrix (Accuracy = {test_acc:.4f})")
fig.tight_layout(); fig.savefig(f"{OUT}/confusion_matrix.png", dpi=150)

wrong = np.where(preds != labels)[0][:16]
fig, axes = plt.subplots(2, 8, figsize=(12, 3.9))
for a, i in zip(axes.flat, wrong):
    a.imshow(test_set.data[i], cmap="gray"); a.set_title(f"G:{labels[i]} T:{preds[i]}", fontsize=8, pad=3); a.axis("off")
fig.suptitle("Yanlış sınıflandırılan test örnekleri (G=Gerçek, T=Tahmin)")
fig.tight_layout(); fig.savefig(f"{OUT}/misclassified.png", dpi=150)
print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))

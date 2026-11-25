"""Telefon fotoğrafını MNIST formatına (28x28, siyah zemin, beyaz rakam) dönüştürür."""
import cv2
import numpy as np
from PIL import Image

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def photo_to_mnist(img):
    """img: PIL.Image veya numpy (RGB). Dönüş: 28x28 float32 dizi (0-1 aralığı)."""
    if isinstance(img, Image.Image):
        img = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img.copy()

    # Büyük fotoğrafları küçült, gürültüyü azalt
    h, w = gray.shape
    scale = 400 / max(h, w)
    if scale < 1:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Koyu mürekkep / açık kağıt -> Otsu eşikleme ile ters ikili görüntü
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = max(3, int(min(binary.shape) * 0.02) | 1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))

    # Bağlı bileşen analizi: kenara değen küçük parçaları ve lekeleri at
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n <= 1:
        return np.zeros((28, 28), np.float32)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(np.argmax(areas)) + 1
    H, W = binary.shape
    mask = np.zeros_like(binary)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches = x == 0 or y == 0 or x + bw >= W or y + bh >= H
        if i == largest or (area >= 0.15 * areas.max() and not touches):
            mask[labels == i] = 255

    # Rakamı kırp, en-boy oranını koruyarak 20x20 kutuya sığdır
    ys, xs = np.nonzero(mask)
    digit = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    dh, dw = digit.shape
    r = 20.0 / max(dh, dw)
    digit = cv2.resize(digit, (max(1, round(dw * r)), max(1, round(dh * r))), interpolation=cv2.INTER_AREA)

    # 28x28 siyah tuvalin ortasına yerleştir (MNIST: 20x20 rakam + 4 piksel kenar boşluğu)
    canvas = np.zeros((28, 28), np.uint8)
    dh, dw = digit.shape
    y0, x0 = (28 - dh) // 2, (28 - dw) // 2
    canvas[y0:y0 + dh, x0:x0 + dw] = digit

    # MNIST'teki gibi kütle merkezini görüntü merkezine kaydır
    m = cv2.moments(canvas)
    if m["m00"] > 0:
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        M = np.float32([[1, 0, 14 - cx], [0, 1, 14 - cy]])
        canvas = cv2.warpAffine(canvas, M, (28, 28))
    return canvas.astype(np.float32) / 255.0


def to_tensor(arr28):
    import torch
    t = torch.from_numpy((arr28 - MNIST_MEAN) / MNIST_STD).float()
    return t.unsqueeze(0).unsqueeze(0)

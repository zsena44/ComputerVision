"""Gradio arayüzü: fotoğraf yükle veya tuvale çiz -> rakam tahmini."""
import glob
import numpy as np
import torch
import gradio as gr
from PIL import Image
from model import MnistCNN
from preprocess import photo_to_mnist, to_tensor

model = MnistCNN(); model.load_state_dict(torch.load("mnist_cnn.pt", map_location="cpu")); model.eval()


def predict(image):
    if image is None:
        return None, None
    if isinstance(image, dict):  # Sketchpad çıktısı
        image = image.get("composite")
    img = np.array(image)
    if img.ndim == 3 and img.shape[2] == 4:  # şeffaf arka planı beyaza çevir
        alpha = img[..., 3:4] / 255.0
        img = (img[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    arr = photo_to_mnist(img)
    with torch.no_grad():
        prob = torch.softmax(model(to_tensor(arr)), 1)[0].numpy()
    preview = Image.fromarray((arr * 255).astype(np.uint8)).resize((196, 196), Image.NEAREST)
    return {str(i): float(p) for i, p in enumerate(prob)}, preview


with gr.Blocks(title="MNIST Rakam Tanıma") as demo:
    gr.Markdown("# ✍️ El Yazısı Rakam Tanıma (CNN – MNIST)\nBir rakam fotoğrafı yükleyin ya da tuvale çizin. "
                "Model 0-9 arası tahmini ve olasılıklarını gösterir.")
    with gr.Row():
        with gr.Column():
            with gr.Tab("Fotoğraf yükle"):
                inp_img = gr.Image(type="numpy", label="Rakam fotoğrafı (beyaz kağıt, koyu kalem)", height=260)
                btn_img = gr.Button("Tahmin et", variant="primary")
                gr.Examples(sorted(glob.glob("fotolar/*.jpg")), inputs=inp_img, label="Telefonla çekilen örnek fotoğraflar")
            with gr.Tab("Çiz"):
                inp_draw = gr.Sketchpad(label="Beyaz zemine siyah rakam çizin", type="numpy",
                                        brush=gr.Brush(default_size=18, colors=["#000000"], color_mode="fixed"),
                                        canvas_size=(320, 320))
                btn_draw = gr.Button("Tahmin et", variant="primary")
        with gr.Column():
            out_label = gr.Label(num_top_classes=3, label="Tahmin")
            out_prev = gr.Image(label="Modele verilen 28x28 görüntü", height=180)
    btn_img.click(predict, inp_img, [out_label, out_prev])
    btn_draw.click(predict, inp_draw, [out_label, out_prev])

if __name__ == "__main__":
    demo.launch()

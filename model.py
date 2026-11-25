import torch.nn as nn


class MnistCNN(nn.Module):
    """2 evrişim bloğu (Conv-BN-ReLU x2 + MaxPool + Dropout) + tam bağlantılı sınıflandırıcı."""

    def __init__(self, num_classes=10):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2), nn.Dropout(0.25),
            )
        self.features = nn.Sequential(block(1, 32), block(32, 64))   # 28->14->7
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(256, num_classes),  # logits; olasılık için softmax
        )

    def forward(self, x):
        return self.classifier(self.features(x))

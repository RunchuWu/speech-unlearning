import torch
import torch.nn as nn


class TinySpeakerCNN(nn.Module):
    """Tiny 3-block CNN for speaker identification from [B, 1, 40, 125] MFCC input."""

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.features(inputs)
        x = self.pool(x)
        return self.classifier(x)

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return 128-dim embedding before final classification layer."""
        x = self.features(inputs)
        x = self.pool(x)
        # classifier[0]=Flatten, [1]=Linear(1024->128), [2]=ReLU, [3]=Dropout
        for layer in list(self.classifier.children())[:-1]:
            x = layer(x)
        return x

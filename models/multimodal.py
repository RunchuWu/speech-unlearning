"""Multimodal speaker-identification models."""

from __future__ import annotations

import torch
import torch.nn as nn

from .audio_cnn import TinySpeakerCNN


class FaceEncoderCNN(nn.Module):
    """Small CNN that maps face frames to a 128-dim embedding."""

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
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
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(128, embedding_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() == 5:
            batch_size, num_frames, channels, height, width = inputs.shape
            inputs = inputs.reshape(batch_size * num_frames, channels, height, width)
            x = self.features(inputs)
            x = self.pool(x).flatten(1)
            x = self.projection(x)
            return x.reshape(batch_size, num_frames, -1).mean(dim=1)

        x = self.features(inputs)
        x = self.pool(x).flatten(1)
        return self.projection(x)


class MultimodalSpeakerCNN(nn.Module):
    """Late-fusion audio-plus-face speaker classifier."""

    def __init__(
        self,
        num_classes: int = 5,
        embedding_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.audio_encoder = TinySpeakerCNN(num_classes=num_classes)
        self.face_encoder = FaceEncoderCNN(embedding_dim=embedding_dim)
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def embed_audio(self, audio_inputs: torch.Tensor) -> torch.Tensor:
        return self.audio_encoder.embed(audio_inputs)

    def embed_face(self, face_inputs: torch.Tensor | None) -> torch.Tensor:
        if face_inputs is None:
            raise ValueError("face_inputs must be provided for multimodal embedding")
        return self.face_encoder(face_inputs)

    def embed(
        self,
        audio_inputs: torch.Tensor,
        face_inputs: torch.Tensor | None,
    ) -> torch.Tensor:
        audio_embedding = self.embed_audio(audio_inputs)
        if face_inputs is None:
            face_embedding = torch.zeros_like(audio_embedding)
        else:
            face_embedding = self.embed_face(face_inputs)

        fused = torch.cat([audio_embedding, face_embedding], dim=1)
        return self.fusion(fused)

    def forward(
        self,
        audio_inputs: torch.Tensor,
        face_inputs: torch.Tensor | None,
    ) -> torch.Tensor:
        fused_embedding = self.embed(audio_inputs, face_inputs)
        return self.classifier(fused_embedding)

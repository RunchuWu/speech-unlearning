"""Gradio demo for the celebrity speech unlearning benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from data_pipeline.celebrity_loader import (
    ALL_CELEBRITIES,
    MFCC_TIME_STEPS,
    N_MFCC,
    TARGET_SR,
    _build_mfcc_transform,
    _preprocess_waveform,
)
from models.audio_cnn import TinySpeakerCNN

try:
    import torchaudio
    import torchaudio.transforms as T
except ImportError as exc:
    raise SystemExit("torchaudio is required for the demo app") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "celebrity" / "models"
SAMPLE_ROOT = REPO_ROOT / "data" / "celebrity"

METHOD_LABELS = {
    "original": "Original",
    "oracle": "Oracle",
    "gradient_ascent": "Gradient Ascent",
    "random_label": "Random Label",
    "fine_tune": "Fine Tune",
    "scrub": "SCRUB",
    "ssd": "SSD",
    "sisa": "SISA",
}

MODEL_CACHE: Dict[str, nn.Module] = {}
MFCC_TRANSFORM = _build_mfcc_transform()
RESAMPLERS: Dict[int, T.Resample] = {}


class SISAEnsemble(nn.Module):
    def __init__(self, members: List[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = [member(inputs) for member in self.members]
        return torch.stack(logits, dim=0).mean(dim=0)


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        try:
            probe = torch.randn(1, 1, N_MFCC, MFCC_TIME_STEPS, device="mps")
            layer = nn.Sequential(
                nn.MaxPool2d(2), nn.MaxPool2d(2), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((4, 4)),
            ).to("mps")
            with torch.no_grad():
                layer(probe)
            return torch.device("mps")
        except Exception:
            pass
    return torch.device("cpu")


DEVICE = select_device()


def available_method_keys() -> List[str]:
    keys = []
    for key in METHOD_LABELS:
        if (CHECKPOINT_DIR / f"{key}.pt").exists():
            keys.append(key)
    return keys or ["original"]


def available_sample_choices() -> List[str]:
    if not SAMPLE_ROOT.exists():
        return []
    return sorted(str(path.relative_to(SAMPLE_ROOT)) for path in SAMPLE_ROOT.glob("*/*.wav"))


def checkpoint_path(method_key: str) -> Path:
    return CHECKPOINT_DIR / f"{method_key}.pt"


def load_model(method_key: str) -> nn.Module:
    if method_key in MODEL_CACHE:
        return MODEL_CACHE[method_key]

    path = checkpoint_path(method_key)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Run celebrity_benchmark.py first."
        )

    payload = torch.load(path, map_location="cpu")
    model_type = payload.get("model_type")

    if model_type == "audio_cnn":
        model = TinySpeakerCNN(num_classes=int(payload["num_classes"]))
        model.load_state_dict(payload["state_dict"])
    elif model_type == "audio_sisa_ensemble":
        members = []
        for state_dict in payload["shard_state_dicts"]:
            member = TinySpeakerCNN(num_classes=int(payload["num_classes"]))
            member.load_state_dict(state_dict)
            members.append(member)
        model = SISAEnsemble(members)
    else:
        raise ValueError(f"Unsupported checkpoint type: {model_type}")

    model.to(DEVICE)
    model.eval()
    MODEL_CACHE[method_key] = model
    return model


def _resolve_audio_path(
    uploaded_audio: Optional[str],
    selected_sample: Optional[str],
) -> Path:
    if uploaded_audio:
        return Path(uploaded_audio)
    if selected_sample:
        return SAMPLE_ROOT / selected_sample
    raise ValueError("Provide an uploaded/recorded clip or select a built-in sample.")


def preprocess_audio_file(audio_path: Path) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(str(audio_path))
    features = _preprocess_waveform(waveform, sample_rate, MFCC_TRANSFORM, RESAMPLERS)
    return features.unsqueeze(0)


def render_mfcc_image(features: torch.Tensor) -> np.ndarray:
    mfcc = features.squeeze(0).squeeze(0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 3))
    image = ax.imshow(mfcc, aspect="auto", origin="lower", cmap="magma")
    ax.set_title("MFCC")
    ax.set_xlabel("Time")
    ax.set_ylabel("Coefficient")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.canvas.draw()

    width, height = fig.canvas.get_width_height()
    buffer = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rgba = buffer.reshape(height, width, 4)
    rgb = rgba[..., :3].copy()
    plt.close(fig)
    return rgb


def predict_speaker(
    uploaded_audio: Optional[str],
    selected_sample: Optional[str],
    method_label: str,
) -> Tuple[str, Dict[str, float], np.ndarray]:
    method_key = next(
        key for key, label in METHOD_LABELS.items()
        if label == method_label
    )
    audio_path = _resolve_audio_path(uploaded_audio, selected_sample)
    features = preprocess_audio_file(audio_path)
    mfcc_image = render_mfcc_image(features)
    model = load_model(method_key)

    with torch.no_grad():
        logits = model(features.to(DEVICE))
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    top_index = int(np.argmax(probabilities))
    top_label = ALL_CELEBRITIES[top_index]
    top_confidence = float(probabilities[top_index])
    prediction_text = f"Predicted speaker: {top_label} ({top_confidence:.1%})"

    scores = {
        celebrity: float(confidence)
        for celebrity, confidence in zip(ALL_CELEBRITIES, probabilities)
    }
    return prediction_text, scores, mfcc_image


def build_interface() -> gr.Blocks:
    sample_choices = available_sample_choices()
    method_choices = [METHOD_LABELS[key] for key in available_method_keys()]

    with gr.Blocks(title="Speech Unlearning Demo") as demo:
        gr.Markdown(
            """
            # Celebrity Speech Unlearning Demo
            Compare the original and unlearned speaker-identification models on the same clip.
            Run `python celebrity_benchmark.py` first so checkpoints exist under `artifacts/celebrity/models/`.
            """
        )

        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Upload Or Record Audio",
                )
                sample_input = gr.Dropdown(
                    choices=sample_choices,
                    label="Or Pick A Built-In Sample",
                    value=sample_choices[0] if sample_choices else None,
                    allow_custom_value=False,
                )
                method_input = gr.Dropdown(
                    choices=method_choices,
                    label="Unlearning Method",
                    value=method_choices[0] if method_choices else None,
                    allow_custom_value=False,
                )
                run_button = gr.Button("Who Is This?")

            with gr.Column():
                prediction_output = gr.Textbox(label="Prediction")
                confidence_output = gr.Label(
                    label="Speaker Confidence",
                    num_top_classes=len(ALL_CELEBRITIES),
                )
                mfcc_output = gr.Image(label="MFCC Spectrogram")

        run_button.click(
            fn=predict_speaker,
            inputs=[audio_input, sample_input, method_input],
            outputs=[prediction_output, confidence_output, mfcc_output],
        )

    return demo


if __name__ == "__main__":
    build_interface().launch()

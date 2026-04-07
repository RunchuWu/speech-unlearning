# Speech Unlearning Project Plan

## Context

The existing `benchmark.py` is a clean academic demo on LibriSpeech with anonymous speakers. The goal is to build a more compelling research project that:

- Uses real celebrity voice data, with Trump as the forget target, to tell a "Right to be Forgotten" story.
- Adds 4 more unlearning methods for a rigorous comparative benchmark: FT, SCRUB, SSD, and SISA.
- Implements a full audio-plus-face multimodal model and joint unlearning.

**Data source:** `yt-dlp` from YouTube (public speeches, C-SPAN, official channels)  
**Forget target:** Trump, with other politicians as retain speakers  
**Multimodal setup:** Audio + face/video frames (late fusion)

## MacBook Pro Scale Guidelines

All experiments are designed to run comfortably on an M-series MacBook Pro using the MPS backend.

| Component | Size | Notes |
| --- | --- | --- |
| Audio clips | 5 speakers x 50 clips x 2s @ 16kHz = ~40 MB | Trivial storage |
| MFCC tensors in RAM | 250 x `[1, 40, 125]` ~= 5 MB | Entire dataset fits in memory |
| Video for multimodal | 5 speakers x 50 clips x 5 frames @ 96x96 ~= 350 MB stored | Fine |
| Training time (`TinySpeakerCNN`, 60 epochs, 250 samples) | ~30s on M2 | Very fast |
| SISA (4 shards) | ~2 min total | Acceptable |
| Full celebrity benchmark (all 6 methods) | ~5-10 min | Comfortable |

**Scale caps:** 5 speakers, 50-100 utterances each. Do not exceed 200 utterances per speaker or video frames larger than `96x96`; those can run slowly on MPS.

## Existing Code To Reuse

All from `benchmark.py`:

- `TinySpeakerCNN` (lines 143-174): reuse as audio encoder
- `AudioRecord`, `ScenarioSplit`, `RecordDataset` (lines 98-140)
- `preprocess_waveform()` (lines 233-260): identical MFCC pipeline
- `train_supervised()` (lines 541-600): shared training loop
- `unlearn_gradient_ascent()`, `unlearn_random_label()` (lines 610-773): move to module
- `evaluate_all()`, `collect_mia_scores()` (lines 851-917): reuse unchanged

## File Structure

```text
speech-unlearning/
├── benchmark.py                         (unchanged — existing demo stays intact)
├── celebrity_benchmark.py               (NEW — main celebrity entry point)
├── multimodal_benchmark.py              (NEW — audio+face entry point)
│
├── data_pipeline/
│   ├── __init__.py
│   ├── celebrity_loader.py              (NEW — yt-dlp download + MFCC preprocessing)
│   └── face_loader.py                   (NEW — face frame extraction from video clips)
│
├── models/
│   ├── __init__.py
│   ├── audio_cnn.py                     (NEW — TinySpeakerCNN extracted here + re-exported from benchmark.py)
│   └── multimodal.py                    (NEW — FaceEncoderCNN + MultimodalSpeakerCNN)
│
├── unlearning/
│   ├── __init__.py
│   └── methods.py                       (NEW — all 6 methods + registry)
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                       (NEW — EER, TAR@FAR, weight distance)
│   ├── visualization.py                 (NEW — t-SNE before/after grid, heatmaps)
│   └── mia.py                           (NEW — loss-threshold + label-only MIA)
│
├── demo/
│   └── app.py                           (NEW — Gradio interactive demo)
├── requirements.txt                     (NEW)
└── README.md                            (UPDATE)
```

## Phase 1: Data Pipeline

File: `data_pipeline/celebrity_loader.py`

### Speakers And Manifest

```python
CELEBRITY_MANIFEST = {
    "trump": [("https://...", "0:00", "0:30"), ...],  # C-SPAN/official clips
    "biden": [...],
    "obama": [...],
    "harris": [...],
    "sanders": [...],
}

FORGET_CELEBRITY = "trump"
RETAIN_CELEBRITIES = ["biden", "obama", "harris", "sanders"]
```

### Key Functions

- `download_celebrity_clips(name, manifest_entry, out_dir)`
  Calls `yt-dlp` + `ffmpeg` to extract and resample to 16 kHz mono WAV.
- `load_celebrity_records(data_dir, speakers, samples_per_speaker=50) -> Dict[str, List[AudioRecord]]`
  Walks `data/celebrity/<name>/`, calls existing `preprocess_waveform()`, and returns `AudioRecord` with integer labels.
- `build_celebrity_scenario(records) -> ScenarioSplit`
  Wraps existing `build_speaker_scenario()` logic.

### `CelebRecord` Dataclass

Extends `AudioRecord` with `celebrity_name` for plot annotations.

```python
@dataclass(frozen=True)
class CelebRecord:
    sample_id: str
    celebrity_name: str   # "trump", "biden", etc.
    label: int
    features: torch.Tensor  # [1, 40, 125]
```

## Phase 2: Unlearning Methods

File: `unlearning/methods.py`

Move GA and RL from `benchmark.py` here unchanged. Add 4 new methods.

### Method 3: Fine-Tune Only (FT)

```python
def unlearn_fine_tune(model, forget_loader, retain_loader, config) -> nn.Module:
    # Train on retain_loader only (no forget data).
    # Standard CE loss, same hyperparams as GA.
```

### Method 4: SCRUB

```python
def unlearn_scrub(model, forget_loader, retain_loader, config) -> nn.Module:
    # Requires config["teacher_model"] = frozen copy of original model
    # Forget phase: maximize KL(student || teacher) on forget set
    # Retain phase: minimize KL(student || teacher) on retain set
    # Key: F.kl_div(F.log_softmax(student_logits), F.softmax(teacher_logits))
```

### Method 5: Selective Synaptic Dampening (SSD)

```python
def compute_fisher_diagonal(model, loader, device, n=200) -> Dict[str, Tensor]:
    # Accumulate squared gradients = diagonal Fisher estimate


def unlearn_ssd(model, forget_loader, retain_loader, config) -> nn.Module:
    # 1. Compute F_forget and F_retain (Fisher diagonals)
    # 2. For each param: new = old - alpha * (F_forget / (F_retain + eps)) * old
    # No gradient loop - single analytical pass. Very fast.
```

### Method 6: SISA

Sharded, Isolated, Sliced, Aggregated.

```python
# Requires sharded training setup - train K sub-models, each on a shard that excludes the forget data
# During unlearning: retrain only the shard(s) containing forget data
# Aggregation: majority vote or averaged softmax across K sub-models
def train_sisa_shards(model_class, all_records, forget_records, num_shards=4, ...) -> List[nn.Module]:
    ...


def aggregate_sisa_predictions(models, loader, device) -> np.ndarray:
    ...


def unlearn_sisa(shard_models, forget_records, config) -> List[nn.Module]:
    ...
```

**Note:** SISA requires training-time sharding, so `celebrity_benchmark.py` trains sharded models up front and includes SISA in the comparison.

### Registry

```python
REGISTRY: Dict[str, Callable] = {
    "gradient_ascent": unlearn_gradient_ascent,
    "random_label": unlearn_random_label,
    "fine_tune": unlearn_fine_tune,
    "scrub": unlearn_scrub,
    "ssd": unlearn_ssd,
    # sisa handled separately (different interface)
}
```

## Phase 3: Model

File: `models/multimodal.py`

### `FaceEncoderCNN`

- Input: `[B, 3, 96, 96]` face crop (mean-pooled across `N` frames per clip)
- 4-block CNN: `3 -> 16 -> 32 -> 64 -> 128` channels, `BN + ReLU + MaxPool2d` after each
- `AdaptiveAvgPool2d((1, 1)) -> Flatten -> Linear(128 -> 128)`
- Output: `[B, 128]`

### `MultimodalSpeakerCNN` (Late Fusion)

- Audio branch: `TinySpeakerCNN` backbone (`features + pool`) -> `Linear(1024 -> 128)` -> 128-dim
- Face branch: `FaceEncoderCNN` -> 128-dim
- Fusion: `Concat([audio_emb, face_emb]) -> Linear(256 -> 128) -> ReLU -> Dropout(0.3) -> Linear(128 -> N)`

Late fusion allows modality-selective unlearning, such as freezing the face branch and unlearning audio only.

## Phase 4: Face Data

File: `data_pipeline/face_loader.py`

```python
def extract_face_frames(video_path, num_frames=5) -> Tensor:  # [3, 96, 96]
    # ffmpeg to extract N uniformly spaced frames -> PIL -> resize 96x96 -> mean-pool -> Tensor


@dataclass(frozen=True)
class MultimodalRecord:
    sample_id: str
    celebrity_name: str
    label: int
    audio_features: Tensor  # [1, 40, 125]
    face_features: Tensor   # [3, 96, 96]
```

Videos are sourced alongside audio using the same `yt-dlp` downloads, while keeping the video stream for face extraction.

## Phase 5: Evaluation

Files under `evaluation/`

### `visualization.py`

- `plot_tsne_embeddings(model, records, device, out_path, celebrity_names)`
  Hook on `classifier[2]` (128-dim), run t-SNE, use celebrities as legend labels, and highlight the forget speaker in red.
- `plot_before_after_tsne_grid(models_dict, ...)`
  One column per method and 2 rows (before/after); this should be the most compelling single plot.
- `plot_parameter_change_heatmap(original_state, unlearned_state, out_path)`
  Per-layer L2 delta, useful for SSD visualization.

### `metrics.py`

- `compute_eer(model, records, device, num_pairs=1000)`
  Cosine similarity of embeddings; find the threshold where FAR = FRR.
- `compute_weight_distance(model_a, model_b)`
  Per-layer Frobenius norm versus oracle retrain.

### `mia.py`

- `loss_threshold_mia(...)`
  Refactored from `benchmark.py`.
- `label_only_mia(model, forget_records, holdout_records, device, n_aug=10)`
  Add Gaussian noise to MFCC features `N` times and measure prediction stability.

## Phase 6: Celebrity Benchmark

File: `celebrity_benchmark.py`

Self-contained entry point:

- Download and load celebrity clips via `celebrity_loader.py`
- Train original, oracle, and SISA shards
- Apply all 6 methods
- Evaluate 4 core metrics plus EER and MIA variants
- Output `artifacts/celebrity/results_summary.csv`, `before_after_tsne.png`, and `results.png`

**Methods evaluated:** No Unlearning | Retrain Oracle | GA | RL | FT | SCRUB | SSD | SISA

## Phase 7: Multimodal Benchmark

File: `multimodal_benchmark.py`

### Research Questions

- Scenario A: Unlearn audio branch only. Does the face branch still identify Trump?
- Scenario B: Unlearn both branches jointly. What is the cost to retain accuracy?

### Key Function

```python
def unlearn_audio_only(multimodal_model, forget_loader, retain_loader, method, config):
    # Freeze face_encoder, apply chosen method to audio_encoder + classifier only
```

**Output:** Comparison table of Scenario A versus B for each method.

## `requirements.txt`

```text
torch>=2.1.0
torchaudio>=2.1.0
torchvision>=0.16.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
pandas>=2.0.0
numpy>=1.24.0
soundfile>=0.12.0
Pillow>=10.0.0
yt-dlp>=2024.1.0
gradio>=4.0.0
```

## Implementation Order

1. `requirements.txt`
2. `models/audio_cnn.py` - extract `TinySpeakerCNN`, add re-export in `benchmark.py`
3. `unlearning/methods.py` - GA + RL refactored, plus FT + SCRUB + SSD + SISA
4. `data_pipeline/celebrity_loader.py` - `yt-dlp` download + `preprocess_waveform()` reuse
5. `celebrity_benchmark.py` - audio-only celebrity benchmark (6 methods)
6. `evaluation/visualization.py`, `evaluation/metrics.py`, `evaluation/mia.py`
7. `data_pipeline/face_loader.py` - face frame extraction
8. `models/multimodal.py` - `FaceEncoderCNN` + `MultimodalSpeakerCNN`
9. `multimodal_benchmark.py` - Scenario A + B
10. `demo/app.py` - Gradio interactive demo
11. `README.md` update

## Interactive Demo

File: `demo/app.py`

The most compelling way to show results is a live Gradio app where users can experience unlearning directly.

### Interface

- Upload or record a short audio clip, or pick from built-in celebrity samples
- "Who is this?" speaker identification with a confidence bar chart for all 5 celebrities
- Method selector dropdown: Original | GA | RL | FT | SCRUB | SSD | SISA
- Live comparison so users can switch methods and see Trump confidence collapse while others stay high

```text
demo/
└── app.py    # Gradio app: upload audio -> speaker confidence bars, toggle between models
```

### Key Gradio Components

- `gr.Audio(source="microphone")` to record live or upload a file
- `gr.BarPlot` or `gr.Label` for confidence per celebrity
- `gr.Dropdown` to select the unlearning method
- `gr.Image` to show the MFCC spectrogram of the uploaded clip

The "Right to be Forgotten" narrative is built in: when a user plays a Trump clip through the original model, confidence is around 95%. After unlearning, confidence drops. SCRUB and SSD should show the cleanest forgetting. This makes the research result tangible.

Requires `gradio` in `requirements.txt`.

## Verification

- `python celebrity_benchmark.py --quick`
  End-to-end smoke test with 5 epochs and 10 samples per speaker.
- `python celebrity_benchmark.py`
  Full run; check `artifacts/celebrity/results_summary.csv`: Trump forget accuracy should drop from about 95% to below 30% for SCRUB and SSD.
- `python multimodal_benchmark.py --quick`
  Verify the audio-plus-face pipeline works.
- t-SNE grid
  Trump cluster should visually separate from retain speakers after unlearning.

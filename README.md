# Speech Unlearning

Repository for a small speech unlearning study, teaching demos, and an in-progress celebrity unlearning benchmark.

## Main Entry Points

- `python3 benchmark.py`
  Runs the self-contained benchmark demo and saves outputs under `artifacts/benchmark/`.
- `python3 celebrity_benchmark.py --quick`
  Runs the celebrity benchmark smoke test and saves outputs under `artifacts/celebrity/`.
- `python3 celebrity_benchmark.py`
  Runs the full celebrity benchmark once `data/celebrity/<name>/*.wav` exists or `CELEBRITY_MANIFEST` is filled in.
- `python3 multimodal_benchmark.py --quick`
  Runs the multimodal smoke test using audio clips plus companion videos and saves outputs under `artifacts/multimodal/`.
- `python3 demo/app.py`
  Launches the Gradio demo for the checkpointed celebrity models after `celebrity_benchmark.py` has produced `artifacts/celebrity/models/*.pt`.
- `python3 lessons/day1_pytorch_basics.py`
  Runs the PyTorch basics walkthrough and saves its figure under `artifacts/day1/`.
- `python3 lessons/day1_audio_basics.py`
  Runs the audio feature walkthrough and saves its outputs under `artifacts/day1/`.
- `python3 lessons/day2_speaker_unlearning.py`
  Runs the earlier teaching demo and saves outputs under `artifacts/day2/`.

## Repository Layout

```text
speech-unlearning/
├── benchmark.py
├── celebrity_benchmark.py
├── multimodal_benchmark.py
├── requirements.txt
├── data_pipeline/
│   ├── celebrity_loader.py
│   └── face_loader.py
├── demo/
│   └── app.py
├── evaluation/
│   ├── metrics.py
│   ├── mia.py
│   └── visualization.py
├── models/
│   ├── audio_cnn.py
│   └── multimodal.py
├── unlearning/
│   └── methods.py
├── lessons/
│   ├── day1_pytorch_basics.py
│   ├── day1_audio_basics.py
│   └── day2_speaker_unlearning.py
├── docs/
│   └── SETUP.md
├── artifacts/
│   ├── benchmark/
│   ├── celebrity/
│   ├── multimodal/
│   ├── day1/
│   └── day2/
└── data/
```

## Celebrity Benchmark Setup

- Install dependencies from `requirements.txt`.
- Add celebrity WAV clips under `data/celebrity/trump/`, `data/celebrity/biden/`, `data/celebrity/obama/`, `data/celebrity/harris/`, and `data/celebrity/sanders/`.
- Or fill `CELEBRITY_MANIFEST` in `data_pipeline/celebrity_loader.py` and run `python3 celebrity_benchmark.py --download`.
- Then run `python3 celebrity_benchmark.py --quick` before the full benchmark.
- The full run now saves model checkpoints under `artifacts/celebrity/models/` for downstream demo use.

## Multimodal Benchmark Setup

- Keep the audio clips under `data/celebrity/<name>/*.wav`.
- Add companion videos under `data/celebrity_video/<name>/<sample_id>.<ext>`.
- Run `python3 multimodal_benchmark.py --quick` first, then the full benchmark.

## Demo

- Run `python3 celebrity_benchmark.py` first so the audio checkpoints exist.
- Then launch `python3 demo/app.py`.
- The demo supports uploaded audio, microphone input, and any built-in samples found under `data/celebrity/`.

## Notes

- `data/` is intentionally ignored by Git. The benchmark and teaching scripts can regenerate or download the data they need.
- The audio benchmark, multimodal benchmark, and demo app are implemented, but the real celebrity data manifests still need to be curated.
- The committed files in `artifacts/` are small reference outputs from prior runs.
- Full setup and walkthrough notes live in `docs/SETUP.md`.

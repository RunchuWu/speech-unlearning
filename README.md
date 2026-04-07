# Speech Unlearning

Repository for a small speech unlearning study, teaching demos, and an in-progress celebrity unlearning benchmark.

## Main Entry Points

- `python3 benchmark.py`
  Runs the self-contained benchmark demo and saves outputs under `artifacts/benchmark/`.
- `python3 celebrity_benchmark.py --quick`
  Runs the celebrity benchmark smoke test and saves outputs under `artifacts/celebrity/`.
- `python3 celebrity_benchmark.py`
  Runs the full celebrity benchmark once `data/celebrity/<name>/*.wav` exists or `CELEBRITY_MANIFEST` is filled in.
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
├── requirements.txt
├── data_pipeline/
│   └── celebrity_loader.py
├── evaluation/
│   ├── metrics.py
│   ├── mia.py
│   └── visualization.py
├── models/
│   └── audio_cnn.py
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
│   ├── day1/
│   └── day2/
└── data/
```

## Celebrity Benchmark Setup

- Install dependencies from `requirements.txt`.
- Add celebrity WAV clips under `data/celebrity/trump/`, `data/celebrity/biden/`, `data/celebrity/obama/`, `data/celebrity/harris/`, and `data/celebrity/sanders/`.
- Or fill `CELEBRITY_MANIFEST` in `data_pipeline/celebrity_loader.py` and run `python3 celebrity_benchmark.py --download`.
- Then run `python3 celebrity_benchmark.py --quick` before the full benchmark.

## Notes

- `data/` is intentionally ignored by Git. The benchmark and teaching scripts can regenerate or download the data they need.
- The celebrity benchmark currently covers the audio-only path; multimodal and demo work are still pending.
- The committed files in `artifacts/` are small reference outputs from prior runs.
- Full setup and walkthrough notes live in `docs/SETUP.md`.

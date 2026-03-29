# Speech Unlearning

Cleaned repository layout for a small speech unlearning study and teaching demos.

## Main Entry Points

- `python3 benchmark.py`
  Runs the self-contained benchmark demo and saves outputs under `artifacts/benchmark/`.
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
├── lessons/
│   ├── day1_pytorch_basics.py
│   ├── day1_audio_basics.py
│   └── day2_speaker_unlearning.py
├── docs/
│   └── SETUP.md
├── artifacts/
│   ├── benchmark/
│   ├── day1/
│   └── day2/
└── data/
```

## Notes

- `data/` is intentionally ignored by Git. The benchmark and teaching scripts can regenerate or download the data they need.
- The committed files in `artifacts/` are small reference outputs from prior runs.
- Full setup and walkthrough notes live in `docs/SETUP.md`.

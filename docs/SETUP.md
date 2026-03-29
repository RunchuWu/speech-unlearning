# 两天操作手册：Speech Unlearning 入门

## 项目结构

- `benchmark.py`
  研究 demo 主入口，输出保存在 `artifacts/benchmark/`
- `lessons/day1_pytorch_basics.py`
  Day 1 上午脚本，输出保存在 `artifacts/day1/`
- `lessons/day1_audio_basics.py`
  Day 1 下午脚本，输出保存在 `artifacts/day1/`
- `lessons/day2_speaker_unlearning.py`
  Day 2 教学版脚本，输出保存在 `artifacts/day2/`

## 第一步：一次性环境安装（约10分钟）

打开 Mac 终端，按顺序执行：

```bash
# 1. 检查 Python 版本（需要 3.9+）
python3 --version

# 2. 创建虚拟环境（隔离依赖，避免污染系统）
python3 -m venv speech_env
source speech_env/bin/activate

# 3. 安装依赖
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install matplotlib scikit-learn pandas soundfile

# 4. 验证安装成功
python3 -c "import torch, torchaudio; print('PyTorch', torch.__version__); print('torchaudio', torchaudio.__version__)"
```

## DAY 1 上午（约3小时）：PyTorch 核心

```bash
source speech_env/bin/activate
cd /path/to/speech-unlearning
python3 lessons/day1_pytorch_basics.py
```

输出文件：

- `artifacts/day1/day1_loss_curve.png`

## DAY 1 下午（约3小时）：音频处理

```bash
python3 lessons/day1_audio_basics.py
```

输出文件：

- `artifacts/day1/day1_audio_features.png`
- `artifacts/day1/test_audio.wav`

## DAY 2（约6小时）：教学版说话人遗忘实验

```bash
python3 lessons/day2_speaker_unlearning.py
```

输出文件：

- `artifacts/day2/model_before_unlearning.pt`
- `artifacts/day2/day2_unlearning_results.png`

## 研究 Demo：最小但严格的 Benchmark

```bash
python3 benchmark.py
```

输出文件：

- `artifacts/benchmark/model_original.pt`
- `artifacts/benchmark/model_retrain.pt`
- `artifacts/benchmark/results_summary.csv`
- `artifacts/benchmark/results.png`

## 常见报错处理

**报错：`ModuleNotFoundError: No module named 'torch'`**

```bash
source speech_env/bin/activate
```

**报错：LibriSpeech FLAC 读取失败 / backend 相关错误**

```bash
pip install soundfile
```

**报错：下载数据集超时**

```bash
# 手动下载后解压到 ./data/LibriSpeech/
# 下载地址：https://www.openslr.org/12
```

## 给老师展示时的说明要点

1. 用 5 位说话人的语音训练分类器，然后比较遗忘前、重训练 oracle 和两种 unlearning 方法。
2. Unlearning 的核心目标不是单纯降低遗忘集准确率，而是在遗忘、保留效用和隐私攻击指标之间做平衡。
3. Benchmark 版比教学版更严格，因为它有单独的 retain test 和 MIA holdout。

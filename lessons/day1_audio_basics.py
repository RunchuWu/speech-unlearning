# =============================================================
# DAY 1 下午：音频处理基础（约3小时）
# 文件：lessons/day1_audio_basics.py
# 运行：python lessons/day1_audio_basics.py
# =============================================================

import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DAY1_ARTIFACT_DIR = REPO_ROOT / "artifacts" / "day1"
DAY1_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 50)
print("PART 1: 准备测试音频 + 读取")
print("=" * 50)

# 优先尝试官方示例音频；如果下载失败，就本地生成一段可视化用的示例音频
SPEECH_URLS = [
    "https://download.pytorch.org/torchaudio/tutorial-assets/steam-train-whistle-daniel_simon.wav",
    "https://download.pytorch.org/tutorial/steam-train-whistle-daniel_simon.wav",
]
SPEECH_FILE = DAY1_ARTIFACT_DIR / "test_audio.wav"
FEATURE_FIGURE_PATH = DAY1_ARTIFACT_DIR / "day1_audio_features.png"

def load_audio_file(filepath):
    """
    优先使用 torchaudio.load；若本机缺少音频 backend，则回退到标准库 wave 读取 PCM WAV。
    返回: waveform [channels, num_samples], sample_rate
    """
    try:
        return torchaudio.load(filepath)
    except RuntimeError as exc:
        print(f"torchaudio.load 不可用，回退到标准 WAV 读取: {exc}")

    with wave.open(filepath, "rb") as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()

        if wf.getcomptype() != "NONE":
            raise RuntimeError("当前回退读取器只支持未压缩 PCM WAV 文件。")

        raw_bytes = wf.readframes(num_frames)

    if sample_width == 1:
        audio = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw_bytes, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw_bytes, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"不支持的 sample width: {sample_width} bytes")

    audio = audio.reshape(-1, num_channels).T
    waveform = torch.from_numpy(audio.copy())
    return waveform, sample_rate

def save_audio_file(filepath, waveform, sample_rate):
    """
    优先使用 torchaudio.save；若本机缺少音频 backend，则回退到标准库 wave 保存 16-bit PCM WAV。
    """
    try:
        torchaudio.save(filepath, waveform, sample_rate)
        return
    except RuntimeError as exc:
        print(f"torchaudio.save 不可用，回退到标准 WAV 保存: {exc}")

    waveform = waveform.detach().cpu().clamp(-1.0, 1.0)
    audio = (waveform.numpy().T * 32767.0).astype(np.int16)

    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(waveform.shape[0])
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())

def generate_synthetic_audio(filepath, sample_rate=16000, duration=2.0):
    """
    生成一段本地示例音频，避免外部下载失败时卡住学习流程。
    这不是语音，但足够演示波形、梅尔频谱和 MFCC 的处理方式。
    """
    print("官方示例音频下载失败，改为生成本地示例音频...")
    num_samples = int(sample_rate * duration)
    t = torch.arange(num_samples, dtype=torch.float32) / sample_rate

    # 混合几个不同频率/时间模式，让时频图里能看到结构变化
    tone_low = 0.35 * torch.sin(2 * torch.pi * 220 * t)
    tone_mid = 0.25 * torch.sin(2 * torch.pi * 440 * t)
    chirp = 0.20 * torch.sin(2 * torch.pi * (300 * t + 180 * t**2))
    burst_mask = ((t > 0.70) & (t < 1.15)).float()
    burst = 0.30 * burst_mask * torch.sin(2 * torch.pi * 880 * t)

    waveform = tone_low + tone_mid + chirp + burst
    waveform = waveform / waveform.abs().max().clamp_min(1e-6)
    waveform = waveform.unsqueeze(0)  # [1, num_samples]

    save_audio_file(filepath, waveform, sample_rate)
    print(f"已生成本地示例音频: {filepath}")

def ensure_example_audio(filepath):
    if os.path.exists(filepath):
        return

    for url in SPEECH_URLS:
        try:
            print(f"下载示例音频: {url}")
            torch.hub.download_url_to_file(url, filepath)
            print(f"下载完成: {filepath}")
            return
        except Exception as exc:
            if os.path.exists(filepath):
                os.remove(filepath)
            print(f"下载失败: {type(exc).__name__}: {exc}")

    generate_synthetic_audio(filepath)

ensure_example_audio(SPEECH_FILE)

# 读取音频
waveform, sample_rate = load_audio_file(SPEECH_FILE)
print(f"波形 shape: {waveform.shape}")   # [声道数, 采样点数]
print(f"采样率: {sample_rate} Hz")
print(f"时长: {waveform.shape[1] / sample_rate:.2f} 秒")
print(f"数值范围: [{waveform.min():.3f}, {waveform.max():.3f}]")
# 音频就是一个数字序列，每个数字代表某时刻的振幅

# 如果是双声道，取第一个声道
if waveform.shape[0] > 1:
    waveform = waveform[0:1]   # shape: [1, N]
print(f"处理后 shape: {waveform.shape}")

print("\n" + "=" * 50)
print("PART 2: 特征提取——把音频变成「图像」给模型看")
print("=" * 50)

# 2-1. MFCC（最常用的语音特征）
# 原理：模拟人耳对频率的感知，提取13-40个系数
mfcc_transform = T.MFCC(
    sample_rate=sample_rate,
    n_mfcc=40,           # 提取40个MFCC系数
    melkwargs={
        "n_fft": 1024,
        "hop_length": 256,
        "n_mels": 64,
    }
)
mfcc = mfcc_transform(waveform)
print(f"MFCC shape: {mfcc.shape}")   # [1, 40, 时间帧数]
print("  ↑ 这就是送入CNN的输入张量")

# 2-2. 梅尔频谱图（更直观，像音频的「照片」）
mel_transform = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=1024,
    hop_length=256,
    n_mels=64
)
mel_spec = mel_transform(waveform)
mel_db = T.AmplitudeToDB()(mel_spec)   # 转分贝，视觉上更清晰
print(f"梅尔频谱 shape: {mel_spec.shape}")  # [1, 64, 时间帧数]

print("\n" + "=" * 50)
print("PART 3: 可视化（理解你在处理什么）")
print("=" * 50)

fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(3, 1, hspace=0.4)

# 3-1. 波形图
ax1 = fig.add_subplot(gs[0])
time_axis = np.linspace(0, waveform.shape[1] / sample_rate, waveform.shape[1])
ax1.plot(time_axis, waveform[0].numpy(), color='steelblue', linewidth=0.5)
ax1.set_title("① 原始波形（时域）— 数字序列，就是这么简单", fontsize=11)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude")
ax1.grid(True, alpha=0.3)

# 3-2. 梅尔频谱图
ax2 = fig.add_subplot(gs[1])
img = ax2.imshow(mel_db[0].numpy(), aspect='auto', origin='lower',
                  cmap='magma', extent=[0, waveform.shape[1]/sample_rate, 0, 64])
plt.colorbar(img, ax=ax2, label='dB')
ax2.set_title("② 梅尔频谱图（时频域）— 这是CNN看到的「图像」", fontsize=11)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Mel filterbank")

# 3-3. MFCC
ax3 = fig.add_subplot(gs[2])
img2 = ax3.imshow(mfcc[0].numpy(), aspect='auto', origin='lower',
                   cmap='coolwarm', extent=[0, waveform.shape[1]/sample_rate, 0, 40])
plt.colorbar(img2, ax=ax3)
ax3.set_title("③ MFCC（40维系数）— 更紧凑的特征表示", fontsize=11)
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("MFCC coefficient")

plt.savefig(FEATURE_FIGURE_PATH, dpi=120, bbox_inches='tight')
print(f"图表已保存: {FEATURE_FIGURE_PATH}")

print("\n" + "=" * 50)
print("PART 4: 数据预处理——统一长度，准备喂给模型")
print("=" * 50)

def load_and_preprocess(filepath, target_sr=16000, target_length=32000):
    """
    加载音频并统一格式：
    - 重采样到 16kHz（说话人识别标准）
    - 裁剪/补零到固定长度（2秒）
    - 提取 MFCC
    返回: tensor shape [1, 40, 125]
    """
    waveform, sr = load_audio_file(filepath)

    # 多声道 → 单声道
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # 重采样
    if sr != target_sr:
        waveform = T.Resample(sr, target_sr)(waveform)

    # 统一长度
    if waveform.shape[1] < target_length:
        # 太短：补零
        pad = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    else:
        # 太长：截断
        waveform = waveform[:, :target_length]

    # 提取 MFCC
    mfcc_fn = T.MFCC(sample_rate=target_sr, n_mfcc=40,
                      melkwargs={"n_fft": 512, "hop_length": 256, "n_mels": 64})
    features = mfcc_fn(waveform)    # [1, 40, ~125]

    return features

# 测试这个函数
features = load_and_preprocess(SPEECH_FILE)
print(f"预处理后 shape: {features.shape}")
print("shape 解读: [batch=1, n_mfcc=40, time_frames=125]")
print("→ 就是一张 40×125 的「灰度图」，可以用 CNN 处理")

print("\n✓ Day 1 下午完成！")
print("你已掌握: 读音频、提取MFCC/梅尔频谱、统一长度预处理")
print("\n明天目标: 用这些特征训练说话人分类器，然后实现遗忘")

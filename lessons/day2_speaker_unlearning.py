# =============================================================
# DAY 2：说话人分类 + 遗忘实验（约6小时，完整项目）
# 文件：lessons/day2_speaker_unlearning.py
# 运行：python lessons/day2_speaker_unlearning.py
#
# 数据来源：torchaudio 内置数据集（自动下载，约350MB）
# 5位说话人，每人50句，共250条音频
# =============================================================

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import random
from pathlib import Path

# ─────────────────────────────────────────
# 0. 全局设置
# ─────────────────────────────────────────
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# Mac M1/M2 有 MPS 加速；没有就用 CPU，50条数据 CPU 完全够
print(f"使用设备: {DEVICE}")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
DAY2_ARTIFACT_DIR = REPO_ROOT / "artifacts" / "day2"
DAY2_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = DAY2_ARTIFACT_DIR / "model_before_unlearning.pt"
RESULT_FIGURE_PATH = DAY2_ARTIFACT_DIR / "day2_unlearning_results.png"

# 说话人设置
# 从 LibriSpeech test-clean 里选5个说话人 ID
SPEAKER_IDS = [1089, 2094, 3570, 4077, 5142]
FORGET_SPEAKER = 1089          # 这个说话人要被"遗忘"
SAMPLES_PER_SPEAKER = 50       # 每人取50句（数据集够大）
TARGET_SR = 16000
CLIP_LENGTH = 32000            # 2秒 × 16000 Hz
N_MFCC = 40

print(f"说话人: {SPEAKER_IDS}")
print(f"遗忘目标: Speaker {FORGET_SPEAKER}")


# ─────────────────────────────────────────
# 1. 数据集类
# ─────────────────────────────────────────
class SpeakerDataset(Dataset):
    """
    从 LibriSpeech 加载指定说话人的音频
    标签: speaker_id → 0,1,2,3,4 的整数
    """
    def __init__(self, speaker_ids, samples_per_speaker, data_root=DATA_ROOT):
        self.samples = []      # (waveform_tensor, label)
        self.speaker2label = {spk: i for i, spk in enumerate(speaker_ids)}
        os.makedirs(data_root, exist_ok=True)

        self.mfcc_fn = T.MFCC(
            sample_rate=TARGET_SR,
            n_mfcc=N_MFCC,
            melkwargs={"n_fft": 512, "hop_length": 256, "n_mels": 64}
        )

        print("\n正在加载数据集（首次运行会下载 ~350MB LibriSpeech）...")
        print(f"数据目录: {os.path.abspath(data_root)}")
        try:
            dataset = torchaudio.datasets.LIBRISPEECH(
                root=data_root,
                url="test-clean",
                download=True
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"无法创建或写入数据目录 {os.path.abspath(data_root)}。"
                "请确认当前目录可写，或先手动创建 ./data 目录。"
            ) from exc

        # 按说话人分桶，各取 samples_per_speaker 条
        buckets = {spk: [] for spk in speaker_ids}
        try:
            for waveform, sr, transcript, spk_id, chapter_id, utt_id in dataset:
                if spk_id in buckets and len(buckets[spk_id]) < samples_per_speaker:
                    buckets[spk_id].append((waveform, sr))
                if all(len(v) >= samples_per_speaker for v in buckets.values()):
                    break  # 够了，不再遍历
        except RuntimeError as exc:
            raise RuntimeError(
                "LibriSpeech 已下载，但当前 torchaudio 无法读取音频文件。"
                "如果下一步报 backend 相关错误，请在虚拟环境里安装 soundfile: "
                "`pip install soundfile`，然后重新运行。"
            ) from exc

        # 预处理并存入 self.samples
        for spk_id, clips in buckets.items():
            label = self.speaker2label[spk_id]
            for waveform, sr in clips:
                feat = self._preprocess(waveform, sr)
                self.samples.append((feat, label))

        random.shuffle(self.samples)
        print(f"数据加载完成: {len(self.samples)} 条")
        for spk_id in speaker_ids:
            count = sum(1 for _, lbl in self.samples if lbl == self.speaker2label[spk_id])
            print(f"  Speaker {spk_id} (label {self.speaker2label[spk_id]}): {count} 条")

    def _preprocess(self, waveform, sr):
        # 多声道 → 单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # 重采样
        if sr != TARGET_SR:
            waveform = T.Resample(sr, TARGET_SR)(waveform)
        # 统一长度
        if waveform.shape[1] < CLIP_LENGTH:
            pad = CLIP_LENGTH - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :CLIP_LENGTH]
        # MFCC: [1, 40, ~125]
        return self.mfcc_fn(waveform)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ─────────────────────────────────────────
# 2. 模型定义：TinySpeakerCNN
# ─────────────────────────────────────────
class TinySpeakerCNN(nn.Module):
    """
    3层 CNN，输入: [batch, 1, 40, ~126]（MFCC图）
    输出: [batch, 5]（5类说话人）
    """
    def __init__(self, n_speakers=5):
        super().__init__()
        self.conv = nn.Sequential(
            # Block 1: [1,40,~126] → [16,20,63]
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2: [16,20,63] → [32,10,31]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3: [32,10,31] → [64,5,15]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # MPS 暂不支持把 5x15 自适应池化到 4x4（输入尺寸不能整除输出尺寸）。
        # 这里改成 5x3，既能压缩时间维，又能在 CPU/CUDA/MPS 上一致运行。
        self.pool = nn.AdaptiveAvgPool2d((5, 3))   # [64,5,15] → [64,5,3]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_speakers)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return self.classifier(x)


# ─────────────────────────────────────────
# 3. 工具函数
# ─────────────────────────────────────────
def evaluate(model, loader, device):
    """返回 (整体准确率, 每个标签的准确率列表)"""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    overall = (all_preds == all_labels).mean()

    per_class = []
    for lbl in range(len(SPEAKER_IDS)):
        mask = all_labels == lbl
        if mask.sum() > 0:
            per_class.append((all_preds[mask] == all_labels[mask]).mean())
        else:
            per_class.append(0.0)

    return overall, per_class, all_preds, all_labels


def get_forget_retain_loaders(dataset, forget_label, batch_size=16):
    """把数据集拆成 forget_set 和 retain_set"""
    forget_indices = [i for i, (_, lbl) in enumerate(dataset.samples) if lbl == forget_label]
    retain_indices = [i for i, (_, lbl) in enumerate(dataset.samples) if lbl != forget_label]

    forget_set = torch.utils.data.Subset(dataset, forget_indices)
    retain_set  = torch.utils.data.Subset(dataset, retain_indices)

    return (DataLoader(forget_set, batch_size=batch_size, shuffle=True),
            DataLoader(retain_set,  batch_size=batch_size, shuffle=True))


# ─────────────────────────────────────────
# 4. 主流程
# ─────────────────────────────────────────
def main():
    # ── 4-1. 加载数据 ──────────────────────
    full_dataset = SpeakerDataset(SPEAKER_IDS, SAMPLES_PER_SPEAKER)

    # 80/20 划分
    n = len(full_dataset)
    n_train = int(n * 0.8)
    n_test  = n - n_train
    train_set, test_set = torch.utils.data.random_split(
        full_dataset, [n_train, n_test], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=16, shuffle=False)
    forget_loader, retain_loader = get_forget_retain_loaders(
        full_dataset,
        forget_label=full_dataset.speaker2label[FORGET_SPEAKER]
    )

    print(f"\n训练集: {n_train} 条，测试集: {n_test} 条")

    # ── 4-2. 训练基础模型 ──────────────────
    print("\n" + "=" * 50)
    print("阶段 1：训练基础说话人分类器")
    print("=" * 50)

    model = TinySpeakerCNN(n_speakers=5).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    train_losses, train_accs = [], []
    EPOCHS = 60

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss, correct, total = 0, 0, 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()          # Step 1
            out = model(x)                 # Step 2
            loss = criterion(out, y)       # Step 3
            loss.backward()                # Step 4
            optimizer.step()               # Step 5

            epoch_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)

        scheduler.step()
        train_losses.append(epoch_loss / len(train_loader))
        train_accs.append(correct / total)

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            val_acc, per_cls, _, _ = evaluate(model, test_loader, DEVICE)
            print(f"Epoch {epoch:3d}/{EPOCHS}  "
                  f"loss={train_losses[-1]:.4f}  "
                  f"train_acc={train_accs[-1]*100:.1f}%  "
                  f"val_acc={val_acc*100:.1f}%")

    # 保存初始模型（用于对比）
    torch.save(model.state_dict(), MODEL_PATH)

    overall_before, per_cls_before, preds_b, labels_b = evaluate(model, test_loader, DEVICE)
    print(f"\n基础模型测试准确率: {overall_before*100:.1f}%")
    print("各说话人准确率（遗忘前）:")
    for i, spk in enumerate(SPEAKER_IDS):
        marker = "← 遗忘目标" if spk == FORGET_SPEAKER else ""
        print(f"  Speaker {spk}: {per_cls_before[i]*100:.1f}%  {marker}")

    # ── 4-3. 遗忘方法 A：梯度上升 ──────────
    print("\n" + "=" * 50)
    print("阶段 2A：遗忘方法 A — 梯度上升（Gradient Ascent）")
    print("原理：普通训练最小化 loss；梯度上升最大化 loss")
    print("      → 模型主动「忘掉」目标说话人的特征")
    print("=" * 50)

    model_ga = TinySpeakerCNN(n_speakers=5).to(DEVICE)
    model_ga.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

    optimizer_ga = torch.optim.SGD(model_ga.parameters(), lr=1e-4)
    FORGET_EPOCHS = 20

    for epoch in range(FORGET_EPOCHS):
        model_ga.train()

        # ★ 对遗忘目标数据做梯度上升（loss 取负号）
        for x, y in forget_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer_ga.zero_grad()
            out = model_ga(x)
            loss = -criterion(out, y)    # ← 注意负号，这就是梯度上升
            loss.backward()
            optimizer_ga.step()

        # 同时在 retain set 上做正常训练，避免灾难性遗忘其他人
        for x, y in retain_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer_ga.zero_grad()
            out = model_ga(x)
            loss = criterion(out, y)     # 正常训练
            loss.backward()
            optimizer_ga.step()

        if epoch % 5 == 0 or epoch == FORGET_EPOCHS - 1:
            _, per_cls, _, _ = evaluate(model_ga, test_loader, DEVICE)
            forget_lbl = full_dataset.speaker2label[FORGET_SPEAKER]
            retain_avg = np.mean([per_cls[i] for i in range(5) if i != forget_lbl])
            print(f"  Epoch {epoch:2d}: "
                  f"遗忘者准确率={per_cls[forget_lbl]*100:.1f}%  "
                  f"保留者均值={retain_avg*100:.1f}%")

    overall_ga, per_cls_ga, preds_ga, labels_ga = evaluate(model_ga, test_loader, DEVICE)
    print(f"\n方法A 测试准确率: {overall_ga*100:.1f}%")

    # ── 4-4. 遗忘方法 B：随机标签 ──────────
    print("\n" + "=" * 50)
    print("阶段 2B：遗忘方法 B — 随机标签（Random Label）")
    print("原理：用错误标签重新训练遗忘目标")
    print("      → 模型把 FORGET_SPEAKER 的特征映射到随机位置")
    print("=" * 50)

    model_rl = TinySpeakerCNN(n_speakers=5).to(DEVICE)
    model_rl.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

    optimizer_rl = torch.optim.Adam(model_rl.parameters(), lr=5e-5)
    forget_lbl = full_dataset.speaker2label[FORGET_SPEAKER]

    for epoch in range(FORGET_EPOCHS):
        model_rl.train()

        # ★ 对遗忘目标用随机错误标签训练
        for x, y in forget_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # 生成随机标签：从其他类随机选，确保不等于真实标签
            rand_labels = torch.randint(0, 4, y.shape).to(DEVICE)
            rand_labels = (rand_labels + forget_lbl + 1) % 5   # 保证不等于 forget_lbl
            optimizer_rl.zero_grad()
            out = model_rl(x)
            loss = criterion(out, rand_labels)   # 用错误标签
            loss.backward()
            optimizer_rl.step()

        # 正常训练保留集
        for x, y in retain_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer_rl.zero_grad()
            loss = criterion(model_rl(x), y)
            loss.backward()
            optimizer_rl.step()

        if epoch % 5 == 0 or epoch == FORGET_EPOCHS - 1:
            _, per_cls, _, _ = evaluate(model_rl, test_loader, DEVICE)
            retain_avg = np.mean([per_cls[i] for i in range(5) if i != forget_lbl])
            print(f"  Epoch {epoch:2d}: "
                  f"遗忘者准确率={per_cls[forget_lbl]*100:.1f}%  "
                  f"保留者均值={retain_avg*100:.1f}%")

    overall_rl, per_cls_rl, preds_rl, labels_rl = evaluate(model_rl, test_loader, DEVICE)
    print(f"\n方法B 测试准确率: {overall_rl*100:.1f}%")

    # ── 4-5. 结果可视化 ────────────────────
    print("\n" + "=" * 50)
    print("阶段 3：生成结果图表")
    print("=" * 50)

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("Speech Unlearning Experiment — Results", fontsize=14, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    spk_labels = [f"SPK-{s}" for s in SPEAKER_IDS]
    forget_lbl = full_dataset.speaker2label[FORGET_SPEAKER]
    colors = ['#ff6b6b' if i == forget_lbl else '#7c6fff' for i in range(5)]

    # 图1: 训练曲线
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(train_losses, color='steelblue', linewidth=2, label='Training Loss')
    ax1_r = ax1.twinx()
    ax1_r.plot(train_accs, color='coral', linewidth=2, linestyle='--', label='Training Acc')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color='steelblue')
    ax1_r.set_ylabel("Accuracy", color='coral')
    ax1.set_title("① 基础模型训练曲线")
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_r.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc='center right')

    # 图2: 各说话人准确率对比
    ax2 = fig.add_subplot(gs[0, 2])
    x_pos = np.arange(5)
    width = 0.28
    bars_before = ax2.bar(x_pos - width, [a*100 for a in per_cls_before], width,
                           label='Before', color='#4ade80', alpha=0.85)
    bars_ga     = ax2.bar(x_pos,          [a*100 for a in per_cls_ga],    width,
                           label='After GA', color='#7c6fff', alpha=0.85)
    bars_rl     = ax2.bar(x_pos + width,  [a*100 for a in per_cls_rl],    width,
                           label='After RL', color='#fbbf24', alpha=0.85)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"SPK\n{s}" for s in SPEAKER_IDS], fontsize=8)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 115)
    ax2.set_title("② 各说话人准确率对比")
    ax2.axvline(x=forget_lbl - 0.5, color='red', linestyle=':', alpha=0.5)
    ax2.text(forget_lbl, 108, "FORGET", ha='center', color='red', fontsize=8)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    # 图3-5: 混淆矩阵
    for col, (preds, labels, title) in enumerate([
        (preds_b,  labels_b,  "③ 遗忘前 Confusion Matrix"),
        (preds_ga, labels_ga, "④ 方法A: Gradient Ascent"),
        (preds_rl, labels_rl, "⑤ 方法B: Random Labels"),
    ]):
        ax = fig.add_subplot(gs[1, col])
        cm = confusion_matrix(labels, preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=[f"S{s}" for s in SPEAKER_IDS])
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(title, fontsize=10)
        # 高亮遗忘目标行
        for j in range(5):
            ax.add_patch(plt.Rectangle((j - 0.5, forget_lbl - 0.5), 1, 1,
                         fill=False, edgecolor='red', lw=2))

    # 图6: 汇总对比表
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')

    summary = [
        ["指标", "遗忘前", "方法A: 梯度上升", "方法B: 随机标签", "理想值"],
        ["整体准确率",
         f"{overall_before*100:.1f}%",
         f"{overall_ga*100:.1f}%",
         f"{overall_rl*100:.1f}%", "—"],
        [f"遗忘目标 (SPK-{FORGET_SPEAKER}) 准确率",
         f"{per_cls_before[forget_lbl]*100:.1f}%",
         f"{per_cls_ga[forget_lbl]*100:.1f}%",
         f"{per_cls_rl[forget_lbl]*100:.1f}%", "~20% (random)"],
        ["保留说话人均值",
         f"{np.mean([per_cls_before[i] for i in range(5) if i!=forget_lbl])*100:.1f}%",
         f"{np.mean([per_cls_ga[i] for i in range(5) if i!=forget_lbl])*100:.1f}%",
         f"{np.mean([per_cls_rl[i] for i in range(5) if i!=forget_lbl])*100:.1f}%", "保持高准确率"],
    ]

    table = ax6.table(
        cellText=summary[1:],
        colLabels=summary[0],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2d2d4e')
            cell.set_text_props(color='white', fontweight='bold')
        elif col == 0:
            cell.set_facecolor('#f0f0f8')
    ax6.set_title("⑥ 方法对比总结", fontsize=11, pad=10)

    plt.savefig(RESULT_FIGURE_PATH, dpi=130, bbox_inches='tight')
    print(f"结果图表已保存: {RESULT_FIGURE_PATH}")

    # ── 4-6. 终端打印总结 ──────────────────
    print("\n" + "=" * 50)
    print("实验总结")
    print("=" * 50)
    print(f"\n{'说话人':<14} {'遗忘前':>8} {'方法A(GA)':>10} {'方法B(RL)':>10}")
    print("-" * 46)
    for i, spk in enumerate(SPEAKER_IDS):
        marker = " ← 遗忘目标" if spk == FORGET_SPEAKER else ""
        print(f"SPK-{spk:<8}  "
              f"{per_cls_before[i]*100:>7.1f}%  "
              f"{per_cls_ga[i]*100:>9.1f}%  "
              f"{per_cls_rl[i]*100:>9.1f}%{marker}")
    print("-" * 46)
    print(f"{'整体准确率':<12}  "
          f"{overall_before*100:>7.1f}%  "
          f"{overall_ga*100:>9.1f}%  "
          f"{overall_rl*100:>9.1f}%")

    print("\n解读:")
    print(f"  方法A (梯度上升): 遗忘目标 {per_cls_ga[forget_lbl]*100:.0f}%，"
          f"保留者均值 {np.mean([per_cls_ga[i] for i in range(5) if i!=forget_lbl])*100:.0f}%")
    print(f"  方法B (随机标签): 遗忘目标 {per_cls_rl[forget_lbl]*100:.0f}%，"
          f"保留者均值 {np.mean([per_cls_rl[i] for i in range(5) if i!=forget_lbl])*100:.0f}%")
    print("\n✓ Day 2 完成！")
    print("你已实现: 数据加载 → CNN训练 → 两种遗忘方法 → 结果对比")
    print("这就是 Speech Unlearning 研究的核心实验流程！")


if __name__ == "__main__":
    main()

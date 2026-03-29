# 两天操作手册：Speech Unlearning 入门

## 第一步：一次性环境安装（约10分钟）

打开 Mac 终端，按顺序执行：

```bash
# 1. 检查 Python 版本（需要 3.9+）
python3 --version

# 2. 创建虚拟环境（隔离依赖，避免污染系统）
python3 -m venv speech_env
source speech_env/bin/activate
# 以后每次打开终端都要执行这行才能进入环境

# 3. 安装所有依赖（一次搞定）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install matplotlib scikit-learn numpy

# 4. 验证安装成功
python3 -c "import torch, torchaudio; print('PyTorch', torch.__version__, '✓'); print('torchaudio', torchaudio.__version__, '✓')"
```

成功输出示例：
```
PyTorch 2.x.x ✓
torchaudio 2.x.x ✓
```

---

## DAY 1 上午（约3小时）：PyTorch 核心

```bash
# 确保在虚拟环境里
source speech_env/bin/activate

# 进入项目目录
cd /path/to/project   # 改成你实际的路径

# 运行脚本，读注释，看输出
python3 day1_pytorch_basics.py
```

**看完输出后，在脑子里默答这几个问题：**
- `requires_grad=True` 是干什么用的？
- 训练循环的5步是什么？为什么要 `zero_grad()`？
- loss 为什么能从 1.6 降到 0.1？

---

## DAY 1 下午（约3小时）：音频处理

```bash
python3 day1_audio_basics.py
# 首次运行会优先下载一个小示例音频；若下载失败会自动生成本地示例音频
```

打开生成的 `day1_audio_features.png`，理解三张图的关系：
- 上图：原始波形（时间 vs 振幅）
- 中图：梅尔频谱（时间 vs 频率 vs 强度）
- 下图：MFCC（压缩后的频率特征）

---

## DAY 2（约6小时）：完整实验

```bash
python3 day2_speaker_unlearning.py
# 首次运行会下载 LibriSpeech test-clean（约350MB），之后不再重复下载
# CPU 上总运行时间约 20-40 分钟
```

**看每个阶段的输出：**
1. 数据加载：确认每个说话人都有 50 条
2. 训练：loss 应从 ~1.6 降到 ~0.2，准确率升到 90%+
3. 梯度上升遗忘：遗忘目标准确率应从 90%+ 跌到 30% 以下
4. 随机标签遗忘：类似效果
5. 打开 `day2_unlearning_results.png` 查看完整图表

---

## 常见报错处理

**报错：`ModuleNotFoundError: No module named 'torch'`**
```bash
source speech_env/bin/activate   # 重新激活虚拟环境
```

**报错：下载数据集超时**
```bash
# 手动下载后放到 ./data/LibriSpeech/ 目录
# 下载地址：https://www.openslr.org/12
# 选 test-clean.tar.gz（约346MB）
tar -xzf test-clean.tar.gz -C ./data/
```

**运行太慢（CPU）**
```python
# 在 day2 脚本里把这行改小：
SAMPLES_PER_SPEAKER = 20   # 从50改成20，速度快3倍，效果略降
EPOCHS = 30                # 从60改成30
```

---

## 给老师展示时的说明要点

1. **这个实验做了什么**：用5位说话人的语音训练分类器，然后用两种方法让它"忘掉"其中一人

2. **为什么这很重要**：如果用户要求删除数据（GDPR），重新训练整个模型代价太高，Unlearning 提供了高效替代方案

3. **两种方法的区别**：
   - 梯度上升：主动"反训练"，遗忘更彻底但可能影响模型稳定性
   - 随机标签：更温和，遗忘不完全但对保留集影响小

4. **下一步可以做**：在 Whisper 等大模型上验证、设计更好的评估指标、探索隐私泄露度量

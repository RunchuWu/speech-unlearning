# =============================================================
# DAY 1 上午：PyTorch 核心概念（约3小时）
# 文件：day1_pytorch_basics.py
# 运行：python day1_pytorch_basics.py
# =============================================================

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

print("=" * 50)
print("PART 1: 张量（Tensor）基础")
print("=" * 50)

# 1-1. 创建张量
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(f"1D张量: {a}")
print(f"2D张量 shape={b.shape}:\n{b}")

# 1-2. 常用操作（和 numpy 几乎一样）
print(f"\na + a = {a + a}")
print(f"a * 2 = {a * 2}")
print(f"b 转置:\n{b.T}")
print(f"b 矩阵乘法:\n{b @ b}")       # @ 就是矩阵乘
print(f"全零张量: {torch.zeros(3)}")
print(f"随机张量: {torch.rand(2,3)}")

print("\n" + "=" * 50)
print("PART 2: 自动求导（Autograd）——最重要的概念")
print("=" * 50)

# 2-1. 简单导数：y = x^2，dy/dx = 2x
x = torch.tensor(3.0, requires_grad=True)   # 告诉 PyTorch：追踪这个变量
y = x ** 2
y.backward()                                 # 自动求导
print(f"x=3, y=x²={y.item():.1f}, dy/dx={x.grad.item():.1f}  (期望: 6.0)")

# 2-2. 多变量：z = x^2 + 2y，∂z/∂x=2x, ∂z/∂y=2
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = x**2 + 2*y
z.backward()
print(f"z=x²+2y: ∂z/∂x={x.grad:.1f} (期望4.0), ∂z/∂y={y.grad:.1f} (期望2.0)")

print("\n" + "=" * 50)
print("PART 3: 训练一个 MLP 分类 Iris 数据集")
print("（这个训练循环的5步结构，和 Day2 音频完全一样）")
print("=" * 50)

# 3-1. 数据准备
iris = load_iris()
X, y = iris.data, iris.target                     # X: (150,4), y: (150,)

scaler = StandardScaler()
X = scaler.fit_transform(X)                        # 标准化：均值0方差1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 转成 PyTorch tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test,  dtype=torch.long)

print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

# 3-2. 定义模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),    # 4个特征 → 32个神经元
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)     # 3类输出（对应3种鸢尾花）
        )

    def forward(self, x):
        return self.net(x)

model = nn.Sequential(
    nn.Linear(4, 32), nn.ReLU(),
    nn.Linear(32, 16), nn.ReLU(),
    nn.Linear(16, 3)
)

# 打印模型结构
print(f"\n模型结构:")
for name, layer in model.named_modules():
    if isinstance(layer, nn.Linear):
        print(f"  Linear: {layer.in_features} → {layer.out_features}")

# 3-3. 训练循环 ★★★ 记住这5步，以后永远是这个模式 ★★★
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

losses = []
print("\n开始训练（100 epochs）:")
print(f"{'Epoch':>6} {'Loss':>8} {'Train Acc':>10}")
print("-" * 30)

for epoch in range(100):
    # ★ Step 1: 清空上一步的梯度（每次必须做）
    optimizer.zero_grad()

    # ★ Step 2: 前向传播（输入 → 输出）
    outputs = model(X_train)

    # ★ Step 3: 计算 loss（预测有多差）
    loss = criterion(outputs, y_train)

    # ★ Step 4: 反向传播（计算每个参数的梯度）
    loss.backward()

    # ★ Step 5: 更新参数（沿梯度反方向走一步）
    optimizer.step()

    losses.append(loss.item())

    if epoch % 20 == 0:
        # 计算训练准确率
        preds = outputs.argmax(dim=1)
        acc = (preds == y_train).float().mean().item()
        print(f"{epoch:>6}   {loss.item():>7.4f}   {acc*100:>8.1f}%")

 # 3-4. 测试集评估
model.eval()
with torch.no_grad():              # 测试时不需要梯度，节省内存
    test_out = model(X_test)
    test_preds = test_out.argmax(dim=1)
    test_acc = (test_preds == y_test).float().mean().item()

print(f"\n测试集准确率: {test_acc*100:.1f}%  (目标: >95%)")

# 3-5. 画 loss 曲线并保存
plt.figure(figsize=(8, 4))
plt.plot(losses, color='steelblue', linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training Loss — Iris Classifier")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("day1_loss_curve.png", dpi=120)
print("\n图表已保存: day1_loss_curve.png")
print("\n✓ Day 1 上午完成！你已经掌握: 张量、自动求导、训练循环")

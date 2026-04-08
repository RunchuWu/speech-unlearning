# Speech Unlearning Research-First Plan

## 1. 研究定位

这个项目的主目标不是先做一个可展示的产品，而是先回答一个清楚的研究问题：

在小规模 celebrity speech identification 场景下，不同 machine unlearning 方法能否有效遗忘目标人物，同时尽量保留非目标人物的识别能力；而在 multimodal 条件下，单独遗忘 audio 是否仍会被 face 信息“补回”身份。

因此接下来的工作顺序应当是：

1. 先定义研究问题和实验协议。
2. 再把 benchmark 跑到可复现。
3. 再补 multimodal 分析和 ablation。
4. 最后才做一个最小 demo 用于展示结果。

## 2. 当前基础

仓库已经具备可作为研究起点的代码骨架：

- `benchmark.py`
  LibriSpeech 上的最小 speech unlearning benchmark。
- `celebrity_benchmark.py`
  已有 celebrity audio benchmark 主流程。
- `unlearning/methods.py`
  已有 GA / RL / FT / SCRUB / SSD / SISA 的实现框架。
- `multimodal_benchmark.py`
  已有 audio-only unlearning 和 joint unlearning 两个 scenario。
- `evaluation/`
  已有 Forget/Retain/Utility、MIA、EER、weight distance、t-SNE 可视化。

所以当前最重要的问题已经不是“缺模块”，而是：

- 数据是否足够干净
- 实验协议是否严谨
- 结果是否可复现
- 结论是否能站住

## 3. 核心研究问题

### RQ1

在 celebrity speech classification 场景下，GA / RL / FT / SCRUB / SSD / SISA 这些方法，谁能最好地平衡：

- Forget effectiveness
- Retain utility
- Privacy leakage

### RQ2

这些结论在不同随机种子、不同样本划分下是否稳定，还是只在单次 run 中成立。

### RQ3

在 multimodal 模型中，只对 audio branch 做 unlearning 是否足够，还是 face branch 仍会保留目标人物身份信息。

### RQ4

不同指标之间是否一致：

- forget accuracy
- retain accuracy
- MIA AUC
- EER
- embedding / parameter distance to oracle

如果这些指标不一致，需要明确解释冲突而不是只汇报单一最好结果。

## 4. 研究假设

- H1: `fine_tune` 会较好保留 retain utility，但 forgetting 不会最彻底。
- H2: `gradient_ascent` 和 `random_label` 会更激进地下压 forget accuracy，但更容易损伤 retain performance。
- H3: `scrub` 和 `ssd` 更可能给出稳定的 forgetting-retention 折中。
- H4: multimodal 中仅遗忘 audio branch 时，face branch 仍可能泄漏目标身份。
- H5: 单纯看 forget accuracy 不足以判断 unlearning 是否成功，MIA 与 oracle distance 必须同时看。

## 5. 实验协议

### 5.1 数据范围

第一阶段固定为 5 个 speaker：

- Forget target: `trump`
- Retain speakers: `biden`, `obama`, `harris`, `sanders`

数据规模分两档：

- quick run: 每人 10 clips
- full run: 每人 50 clips

如果 full run 结论仍不稳定，再扩展到每人 80-100 clips，但不在一开始扩数据。

### 5.2 数据采样规则

必须先定义数据协议，再开始跑结论：

- 每条音频统一截成 2 秒
- 每个 speaker 至少来自 3 个不同 source
- 避免从同一视频连续切出大量高度相似样本
- 尽量覆盖 speech、interview、debate、rally 等不同场景
- 尽量避免背景音乐、强掌声、多人串音片段

建议为每个 clip 保存最小 metadata：

- `speaker`
- `source_url`
- `source_type`
- `start_time`
- `end_time`
- `duration`
- `quality_note`
- `duplicate_group`

这一步的目标不是先凑数量，而是先得到一份可研究的数据集。

### 5.3 划分协议

audio benchmark 采用固定 forget / retain / holdout 逻辑：

- `forget_train`
- `forget_test`
- `retain_train`
- `retain_test`
- `mia_holdout`

要求：

- train/test/holdout 之间样本不重叠
- 尽量避免同一 source video 的高度相邻片段同时落入 train 和 test
- 每个 seed 的划分规则可复现

### 5.4 模型与方法

audio benchmark 作为主研究线，先聚焦以下方法：

- No Unlearning
- Retrain Oracle
- Gradient Ascent
- Random Label
- Fine Tune
- SCRUB
- SSD
- SISA

说明：

- `Retrain Oracle` 是主要参考上界
- 当前 `SISA` 实现更接近 approximate SISA，需要在文档和结果中明确标注

### 5.5 指标

主表保留以下指标：

- Forget Acc ↓
- Retain Acc ↑
- Test Utility ↑
- Loss-threshold MIA AUC ↓
- Label-only MIA AUC ↓
- EER ↓
- Weight Distance vs Oracle ↓

判断标准：

- forget 降低但 retain 全崩，不算成功
- forget 降低但 MIA 仍高，不算成功
- 与 oracle 更接近的方法更值得信任，但不能替代性能指标

### 5.6 复现要求

为了避免只看单次最好结果，最低要求如下：

- 每个主要方法至少跑 3 个 seeds
- 保存每次运行的 config、seed、结果 csv、checkpoint
- 汇总 mean / std
- 所有图和表都能从保存的 artifacts 重新生成

## 6. 研究主线与阶段目标

### Phase 1: 建立可信 audio benchmark

这是最核心的阶段，也是整个项目的 P0。

研究目标：

- 得到一套足够干净的 celebrity audio 数据
- 在 audio-only 场景下比较各 unlearning 方法
- 明确是否存在稳定优于 baseline 的方法

具体任务：

- 补齐 `CELEBRITY_MANIFEST`
- 先人工筛选一版高质量小数据集
- 跑通 `python3 celebrity_benchmark.py --quick`
- 跑至少 1 次 full benchmark
- 对结果做 sanity check

Phase 1 的验收标准：

- benchmark 能稳定跑通
- 结果可重复生成
- 能给出一张主结果表和初步结论

### Phase 2: 补足研究可信度

研究目标：

- 让结论不依赖单次 lucky run
- 明确方法间 tradeoff

具体任务：

- 做 3 seed 重复实验
- 比较 mean / std
- 检查每种方法的失败模式
- 明确哪些方法只是在压 accuracy，哪些方法是真的降低 membership signal

Phase 2 的验收标准：

- 有聚合结果表
- 能说清不同方法的优缺点
- 能明确指出当前结论的边界条件

### Phase 3: multimodal 研究

multimodal 是研究扩展，不应在 audio benchmark 还不稳定时提前抢主线。

研究目标：

- 测试 audio-only unlearning 是否足够
- 测试 joint unlearning 的代价和收益

具体任务：

- 准备对齐的 audio + video 数据
- 跑两个 scenario：
  Scenario A: freeze face branch，只遗忘 audio
  Scenario B: audio + face jointly unlearn
- 补 branch-sensitive 分析：
  audio-only prediction
  face-only prediction
  fusion prediction

Phase 3 的验收标准：

- 至少能回答一次 “只删语音是否足够” 的问题
- 至少有一张 audio-only vs joint 的比较图

## 7. Benchmark 具体计划

### 7.1 Audio benchmark

这是第一优先级。

必做项：

- 数据清洗和 manifest 补齐
- `--quick` smoke test
- full benchmark 单次跑通
- 多 seed 汇总
- 输出主表和主图

建议新增输出：

- `artifacts/celebrity/results_by_seed.csv`
- `artifacts/celebrity/results_aggregate.csv`
- 每种方法的配置快照

### 7.2 Multimodal benchmark

这是第二优先级。

必做项：

- 对齐 audio/video 样本
- 先跑 quick，再跑 full
- 新增 branch-level 结果表

建议重点分析：

- audio-only unlearning 后，fusion 是否仍识别 target
- joint unlearning 对 retain speakers 的损伤是否显著

### 7.3 Ablation

如果前两项稳定，再补以下 ablation：

- forget set 大小变化
- unlearning epoch / lr 敏感性
- SCRUB temperature / alpha
- SSD fisher sample 数量
- 是否使用更严格的数据去重策略

ablation 不是当前 P0，但它是把结果写成研究结论的重要部分。

## 8. 两周里程碑

### Week 1

只做研究主线，不做 UI 打磨。

目标：

- 完成 celebrity 数据协议
- 做出一版高质量小数据集
- 跑通 audio quick
- 跑第一次 audio full

Week 1 结束时必须有：

- 一份可用 manifest
- 一次完整 audio benchmark 结果
- 一版初步研究结论

### Week 2

在 audio 结果初步稳定后，继续增强研究可信度。

目标：

- 完成 3 seed audio benchmark
- 汇总 mean / std
- 启动 multimodal quick/full
- 形成主表、主图和结论摘要

Week 2 结束时必须有：

- audio 聚合结果表
- 至少一张 multimodal 对比图
- 一页研究结论摘要

## 9. 优先级

### P0

- 制定 celebrity 数据协议
- 补齐 `CELEBRITY_MANIFEST`
- 跑稳 `celebrity_benchmark.py`
- 做多 seed 汇总

### P1

- 跑 multimodal benchmark
- 加入 branch-level 分析
- 做关键 ablation

### P2

- 完善 approximate SISA 或升级为更完整实现
- 优化可视化和图表叙事
- 准备最小展示 demo

## 10. 风险与边界

### 风险 1: 数据 shortcut

如果 speaker 与 source/channel 绑定过强，模型可能学到的是录制条件而不是说话人特征。

应对：

- 增加 source 多样性
- 做人工去重
- 尽量打散来源

### 风险 2: 结果不稳定

如果不同 seeds 下方法排序频繁翻转，说明当前数据规模或协议还不够支撑强结论。

应对：

- 固定 3 seeds 起步
- 统一汇报 mean / std
- 先给弱结论，不给过度结论

### 风险 3: 指标冲突

forget acc、MIA、EER 可能不会一致变化。

应对：

- 明确区分“分类性能下降”和“隐私遗忘成功”
- 结果汇报必须并列展示多个指标

### 风险 4: SISA 解释风险

当前实现不是严格意义上的 full SISA unlearning。

应对：

- 在结果中明确标注 approximate SISA
- 不把它包装成强结论来源

## 11. Demo 的位置

demo 不是主线任务，只是研究结果的展示层。

当前策略：

- 只保留一个最小 demo
- 不为 demo 新增额外研究负担
- 不因为 UI 需求改变实验设计

demo 最早也应该在以下条件满足后再投入：

- audio benchmark 已经稳定
- 至少有一组可复现结果
- 已经知道要展示哪 1-2 个方法

如果前面的研究工作还没有完成，demo 不应抢占时间。

## 12. 下一步立即执行

接下来按这个顺序做：

1. 先整理 celebrity 数据协议和采样标准。
2. 补齐 `CELEBRITY_MANIFEST` 或至少做一版高质量手工样本集。
3. 跑 `python3 celebrity_benchmark.py --quick`。
4. 跑第一次 full audio benchmark。
5. 基于结果决定 multimodal 是否进入下一阶段。

当前最重要的不是“展示起来像不像产品”，而是先把第一张能支撑研究结论的表做出来。

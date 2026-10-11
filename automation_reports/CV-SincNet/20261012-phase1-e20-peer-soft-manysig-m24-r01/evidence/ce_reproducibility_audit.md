# 两批 CE 续训的可复现性核对

核对时间：2026-10-12（Asia/Hong_Kong）。范围仅限三 seed CE 的代码、源域训练日志、源 V 诊断、实际配置和采样器状态。没有读取目标成绩或预测，没有启动训练、改变后端或重跑任何 row。

**结论：两批 CE 的主配方、父权重、数据、采样规则和训练预算一致，但实际源轨迹不是逐位复现。初次损失差异出现在第 2、2、3 次更新，量级为 1.5e-8 至 3.0e-8；实际后端允许 cuDNN 非确定性算法。该现象与浮点微差在迭代优化中累积相符，但本次只读证据没有唯一证明 CUDA 非确定性就是原因。** 未做受控确定性复跑，不能据此宣称已排除所有运行时差异。

## 证据范围

| 项目 | 先前批次 | 本批次 |
|---|---|---|
| Run | `20261012-phase1-e20-routed-fishr-manysig-m24-r01` | `20261012-phase1-e20-peer-soft-manysig-m24-r01` |
| 实际执行 commit | `7010b7393dcd91d70366fd50fc18639667aea400` | `e5a134db7794b3b259eb99621966fdfe29ef1bb9` |
| 本次核对 row | `CE-s2026092701/02/03` | `CE-s2026092701/02/03` |
| 每 row 完整记录 | 20 个 epoch、4,440 个 step | 20 个 epoch、4,440 个 step |

读取并比较两批既有 `source_full_analysis.json` 的三个 CE 行；该分析明确记录 `source_only=true`、`target_prediction_or_scores_read=false`，覆盖完整已关闭的逐步日志、epoch JSONL/CSV、暖头及训练 stdout。另通过短连接只读解析六行原始 `source/step_metrics.jsonl.gz` 全部 26,640 条记录、`source/resolved_config.json` 和 `source/auxiliary_state.pt`。没有以日志末尾或抽样 epoch 代替完整轨迹比较。

本地完整分析：

- [先前 source_full_analysis.json](../../20261012-phase1-e20-routed-fishr-manysig-m24-r01/evidence/source_full_analysis.json)
- [本批 source_full_analysis.json](source_full_analysis.json)

远端补充只读证据均位于上述两个 run 的 `CE-s{seed}/source/`，文件名为 `resolved_config.json`、`step_metrics.jsonl.gz`、`auxiliary_state.pt`。没有改写远端产物。

## 配方、状态与代码路径

六份实际配置逐 seed 比较：父 checkpoint 路径、`preparation`、`optimizer`、`model_seed`、`unlabeled`、`batch_size`、`updates`、`parameter_groups`、`semi`、`losses` 和 `multilevel` 均相同。三个 seed 都继承同 seed 原生 E20，重置 AdamW，追加 20×222=4,440 次更新，L batch=128；学习率由 2e-4 cosine 衰减至 1e-6，weight decay=1e-4。逐步学习率 4,440/4,440 项完全相同。

CE 不加载教师概率库：`bank=None`，`teacher.source=none`，U iterator、EMA、辅助网络、Fishr、MixStyle 和星地增强均关闭。两批 `head_warmup` 文件代码相同，CE 直接走 `NOT_APPLICABLE`，没有暖头前向或更新。实际计数均为 L packet occurrences=568,320，U、辅助及教师执行计数为 0。参数分组名称和顺序一致。

两份 `model.py` 的训练初始化路径一致：验证原 E20、严格加载权重、建立 `OriginalIdentity`。新增教师来源校验仅在本批加载自己完成 checkpoint 的分支生效，不改变 CE 的父权重初始化。两实际 commit 间共享训练依赖没有代码修改；新增文件集中在 peer audit/peer soft 包。

两份 `source.py` 都在建立优化器后再次执行相同 `seed_all(model_seed)`，随后构造相同 `DomainStream(..., model_seed+101)`。该采样器使用独立 CPU `torch.Generator`。三对最终 `L_sampler` 的 keys、count、pools、positions、pass_index、domain_cursor 和 generator 状态全部逐项相同；均为 pass_index=91、domain_cursor=0。U sampler、teacher、multilevel 保存状态均为 null。这些证据支持采样路径一致；没有逐步保存全局 CPU/CUDA RNG 状态，因此不能把末态采样器一致扩大成所有运行时随机状态都已逐步验证。

主目标在 CE 路径均为相同 `F.cross_entropy(logits, y)` 加六项固定顺序权重和，其他五项为图内零。旧版 `sum(weighted.values())` 的插入顺序与新版 `BASE_WEIGHTED_KEYS` 一致，未发现新增的非零 CE 主目标。

本批确实多了审计计算：`KL_same`、`KL_disagree` 两个零项会构造标量运算，并在每 444 步执行额外 `autograd.grad(..., retain_graph=True)`，但不加入 total loss，不做 optimizer 更新，也没有新增随机数抽样。该额外审计不能解释第 2/3 步已出现的差异，因为第一次审计发生在第 444 步；不过两批 GPU 运算调用序列并非完全相同，不能据数学目标相同就声称整个执行过程逐位等价。

## 完整源轨迹中的实际差异

三 seed 的第 1 步 CE、总损失、梯度范数和 L accuracy 完全相同。下表比较的是后续第一个 **CE 标量不相同** 的 step，而不是第一个分类预测不同的 step。

| seed | 首个 CE 差异 step | 先前 CE | 本批 CE | 初始绝对差 | 4,440 步中 CE 不同数 | 首个 L accuracy 差异 step |
|---|---:|---:|---:|---:|---:|---:|
| 2026092701 | 2 | 0.12670589983463287 | 0.12670592963695526 | 2.98023224e-8 | 4,434 | 95 |
| 2026092702 | 2 | 0.13075204193592072 | 0.13075201213359833 | 2.98023224e-8 | 4,430 | 117 |
| 2026092703 | 3 | 0.1626364290714264 | 0.1626364141702652 | 1.49011612e-8 | 4,435 | 117 |

后续迭代中的单 step CE 最大绝对差分别为 0.03291199、0.03957365、0.03226022；20 个 epoch 的均值 CE 最大绝对差分别为 0.00091148、0.00070823、0.00103328。不能把“初差很小”表述成“整段训练逐位相同”。

| seed | 先前 E1 均值 CE | 本批 E1 均值 CE | 先前 E20 均值 CE | 本批 E20 均值 CE | 先前最终源 V 准确率 | 本批最终源 V 准确率 | V 差（百分点） |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026092701 | 0.09607051 | 0.09629553 | 0.01933717 | 0.01918457 | 98.025926% | 98.037037% | +0.011111 |
| 2026092702 | 0.10204110 | 0.10203320 | 0.02496370 | 0.02490887 | 98.029630% | 98.048148% | +0.018519 |
| 2026092703 | 0.09446645 | 0.09452920 | 0.01881367 | 0.01899121 | 97.970370% | 97.951852% | −0.018519 |

源 V 为每行固定 27,000 条合法验证样本；该表仅描述复现差异，不用于候选选择，也不包含目标测试结论。

## 实际数值后端与检查边界

六行记录的后端配置一致：

```json
{
  "cudnn_allow_tf32": false,
  "cuda_matmul_allow_tf32": false,
  "cudnn_benchmark": false,
  "cudnn_deterministic": false,
  "matmul_precision": "highest"
}
```

硬件均记录 RTX 3090。seed 2026092701 两次均 GPU 0，seed 2026092702 两次均 GPU 7；seed 2026092703 先前 GPU 0、本批 GPU 2。前两 seed 同卡仍出现差异，不能只用换卡解释。关闭 TF32 和 benchmark 并不等于已要求所有 CUDA 运算确定性。当前记录没有逐 step kernel 选择、全局 RNG 快照或同时负载明细，无法直接定位最早产生差异的具体 kernel 或参数张量。

已有 `cvs_e20_peer_soft/checks.py` 的等价检查明确先设置 `torch.backends.cudnn.deterministic=True`，用同一有界输入、重置模型状态和 RNG，比较旧 `SSL` 与新 `SELF_GATE` 的 loss、gradient、RNG。这证明所测输入和所测后端下该目标实现等价；它不是 CE 三 seed 的 4,440 步端到端确定性续训试验，也没有复刻正式运行的 `cudnn_deterministic=false` 设置、完整采样器、AdamW 更新和实际并发条件。

现有 integration/preflight 验证了父权重精确继承、own checkpoint roundtrip、固定预算、隐藏 U 标签及有界真实源 IQ 梯度路由。这些检查不承诺两批完整训练得到逐位相同轨迹。

因此，本批分析应保留并使用本批同预算 CE 对照；不能将它与先前 CE 声称为同一次运行、互换权重或选择其中分数更高者。跨批配对可以注明数学配方相同但数值轨迹存在差异。**本次结论停在证据支持的范围：未发现新的非零 CE 主目标或显式 RNG 消耗差异；有实际浮点轨迹差异，后端非确定性是合理解释之一，尚未完成唯一归因。**

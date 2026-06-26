# BEX02 集中式域泛化训练与联邦训练可行性报告

生成时间：2026-05-26  
项目范围：CV-SincNet / CVS-RFFI / WiSig BEX02 风格训练  
核心问题：如果希望联邦训练最大限度接近集中式 BEX02 域泛化训练，应该如何理解集中式训练流程、联邦训练会失去什么、有哪些替代方案、代价和局限是什么。

---

## 1. 结论摘要

集中式 BEX02 域泛化训练的核心不是某一个单独损失函数，而是一个完整训练语义：

```text
全局多 receiver/day 域 batch
+ 身份/域解耦模型
+ GRL 去域
+ Fishr 跨域梯度统计对齐
+ MixStyle 跨域风格扰动
+ same-TX cross-domain consistency
+ hard-domain / group-domain loss
+ satellite clean+sat supervised view expansion
+ OOD split checkpoint selection
```

其中最不可替代的是：

```text
同一个训练 step 内同时看到多个 receiver/day 域。
```

如果联邦训练只是普通 FedAvg：

```text
每个 client 本地训练若干 step/epoch -> 上传模型 delta -> 服务器平均
```

那么它并不等价于集中式 BEX02。原因是 Fishr、GRL、MixStyle、hard-domain CE 和 same-TX consistency 都依赖 batch 内多域样本或全局域统计。客户端本地如果只有一个 receiver 或一个 receiver-day，这些机制会明显退化。

要“原汁原味”，从最接近到最实用的路线是：

1. 共享或流式上传 IQ，在服务器直接集中式训练。
2. Split Learning：客户端保留前端，服务器拼接全局多域 feature batch 并执行 BEX02 loss。
3. 同步 FedSGD / distributed centralized training：每个 global step 同步跨 client 梯度，服务器维护 optimizer。
4. StyleBank / ProtoBank / 分解梯度：不共享 IQ，但用统计和虚拟域逼近集中式多域训练。
5. 普通 FedAvg + 本地 BEX02 objective：可作为工程基线，但不能称为严格复刻。

---

## 2. 集中式域泛化训练是什么

### 2.1 任务目标

WiSig / BEX02 风格训练中，模型要预测的是发射机身份：

```text
input IQ -> TX label
```

但每个样本还带有采集域信息：

```text
receiver
day
receiver x day
channel / satellite scenario
```

集中式域泛化的目标不是让模型完全不知道这些域，而是让模型学会：

```text
TX identity = 稳定因素
receiver/day/channel = 可变因素
```

也就是：

```text
z_id  应该保留发射机身份，并尽量去除 receiver/day 域信息
z_dom 应该承接 receiver/day/channel 域信息
```

### 2.2 训练域与测试域

典型划分：

```text
训练域：
  seen receiver + seen day

测试域：
  test_unseen_day_seen_rx
  test_seen_day_unseen_rx
  test_unseen_day_unseen_rx
```

最严格的泛化场景是：

```text
test_unseen_day_unseen_rx
```

即测试时 receiver 和 day 都未出现在训练组合里。

### 2.3 集中式训练的关键优势

集中式 dataloader 拥有全局训练集，因此可以在一个 batch 里采到多个 receiver/day 域：

```text
batch = [
  TX0 @ RX0 day0
  TX0 @ RX3 day1
  TX1 @ RX2 day0
  TX1 @ RX6 day1
  ...
]
```

这使得下面这些操作有真实语义：

- Fishr：比较不同域的梯度方差。
- GRL：让主身份特征 `z_id` 难以预测 receiver/day。
- MixStyle：跨域混合风格。
- same-TX consistency：同一个 TX 在不同域的表征应该接近。
- hard-domain CE：识别当前难域并提高其权重。
- satellite clean+sat expansion：在同一个监督 batch 中加入 satellite view。

如果 batch 内只有一个域，这些机制要么失效，要么只能变成局部正则。

---

## 3. 集中式 BEX02 一次训练 step 怎么进行

### 3.1 输入 batch

训练 loader 输出：

```text
x      : IQ samples
y      : TX labels
d_raw  : raw receiver/day domain labels
d      : mapped training domain labels
```

集中式训练知道每条样本来自哪个 receiver/day 域。

### 3.2 普通 RF 增强

训练时先对 clean IQ 做常规物理/链路增强，例如：

```text
PA distortion
gain / phase variation
noise
frequency / timing perturbation
receiver-chain variation
```

这些增强用于扩大训练分布，但仍然围绕 TX identity 保持监督。

### 3.3 satellite view 增强

BEX02 / baseline-style 星地增强最关键的方式是：

```text
x_clean -> TX
x_sat   -> same TX
```

并把两者拼成一个监督 batch：

```text
x_view = concat(x_clean, x_sat)
y_view = concat(y, y)
d_view = concat(d, d)
```

这叫 clean+sat supervised view expansion。它和“辅助 satellite consistency loss”不同。前者直接把 satellite view 当作同标签监督样本参与主分类和下游域泛化损失；后者只是额外算一个弱一致性/辅助损失。

### 3.4 模型前向

模型不是一个普通分类器，而是身份分支和域分支并存：

```text
IQ
 ├─ ID branch      -> z_id  -> TX classifier
 ├─ domain branch  -> z_dom -> domain classifier
 └─ GRL branch     -> z_id  -> adversarial domain classifier
```

含义：

```text
z_id 负责识别 TX，但应该尽量不含 receiver/day 信息。
z_dom 负责承接域信息，帮助模型分离 identity 与 domain。
```

### 3.5 分类损失

最基础目标：

```text
loss_cls = CE(tx_logits, y)
```

它保证模型仍然在做 RFFI / TX 分类。

### 3.6 domain branch 损失

域分支预测 receiver/day：

```text
loss_dom = CE(domain_logits_from_z_dom, d)
```

作用是让 `z_dom` 显式学习域因素。

### 3.7 GRL 对抗去域

GRL forward 不改变输入：

```text
GRL(z_id) = z_id
```

但 backward 时反转梯度：

```text
grad = -lambda_adv * grad
```

训练语义是：

```text
domain classifier 想从 z_id 预测 domain
encoder 想让 domain classifier 预测失败
```

所以最终希望：

```text
z_id 能预测 TX
z_id 不能预测 receiver/day
```

注意：GRL 必须发生在本次前向/反向计算图里。它不是服务器拿到普通模型 delta 后可以事后补上的操作。

### 3.8 same-TX cross-domain consistency

对同一个 TX，在不同 receiver/day 下的 `z_id` 做一致性约束：

```text
TX0 @ RX0 day0 的 z_id
TX0 @ RX3 day1 的 z_id
应该接近
```

目的：

```text
同一发射机的身份特征跨域稳定。
```

这个 loss 强依赖 batch 中存在同 TX 的跨域样本。

### 3.9 hard-domain CE / GroupDRO

集中式训练可以估计当前 batch 或一段训练中哪些域更难：

```text
domain A loss 低
domain B loss 高
domain C accuracy 差
```

hard-domain CE 会对高损失域加权，避免模型只优化容易域：

```text
loss_group_ce = CE on hard domains
```

这有助于改善最弱 receiver/day 和严格 OOD 泛化。

### 3.10 Fishr

Fishr 的直觉：

```text
不同域上，为了完成 TX 分类所需的梯度统计应该一致。
```

它通常对 logit-gradient variance 做跨域对齐：

```text
domain A gradient variance
domain B gradient variance
domain C gradient variance
...
尽量一致
```

如果某个域的梯度统计与其他域差异很大，说明模型可能在该域依赖了域特有偏差。

BEX02 风格中常见配置：

```text
lambda_fishr = 0.02
fishr_min_domains = 4
```

这意味着 Fishr 至少需要多个有效域。batch 内域数不足时，Fishr 会弱化或变成 0。

### 3.11 MixStyle

MixStyle 在中间特征层混合不同样本/域的风格统计。

在 BEX02 中，`same_tx_crossdomain` 的语义更强：

```text
同一个 TX
跨不同 receiver/day 域
混合风格
```

目的：

```text
身份内容保持
receiver/day 风格扰动
```

模型因此不容易把 receiver/day 风格误当作 TX identity。

### 3.12 总损失

集中式 BEX02 风格总损失可以抽象为：

```text
total_loss =
    loss_cls
  + lambda_dom      * loss_dom
  + lambda_adv      * loss_adv
  + lambda_orth     * loss_orth
  + lambda_cons     * loss_cons
  + lambda_group_ce * loss_group_ce
  + lambda_fishr    * loss_fishr
  + satellite-related terms
```

然后：

```text
backward
optimizer.step
```

集中式训练由单一 optimizer 维护全局动量、AdamW 二阶矩、学习率调度和 checkpoint 选择。

---

## 4. 为什么普通联邦训练不等价

### 4.1 FedAvg 的训练语义不同

普通 FedAvg 是：

```text
server sends global model
client 1 trains locally
client 2 trains locally
...
clients upload model deltas
server averages model deltas
```

而集中式 BEX02 是：

```text
global multi-domain batch
single forward graph
single BEX02 loss
single backward
single optimizer step
```

这两个语义不同。

### 4.2 客户端 batch 通常是单域或少域

如果 `fl_client_key=receiver`：

```text
client rx0 sees samples from receiver 0
client rx1 sees samples from receiver 1
...
```

如果 `fl_client_key=receiver_day`：

```text
client rx0_day0 sees only receiver 0 day 0
client rx0_day1 sees only receiver 0 day 1
...
```

这会破坏集中式 BEX02 最重要的前提：

```text
同一个 batch 内有多个 receiver/day 域。
```

### 4.3 Fishr 退化

Fishr 需要多个有效域比较梯度方差。

联邦客户端本地如果只有一个 receiver 或一个 receiver-day：

```text
fishr_domain_count < fishr_min_domains
```

则 Fishr 要么不激活，要么只能在非常狭窄的局部域上计算，不能代表全局 receiver/day 泛化。

### 4.4 GRL 语义变弱

GRL 本来要让 `z_id` 去除全局 receiver/day 域信息。

但如果某客户端只见一个 receiver：

```text
本地 domain classifier 没有足够多 receiver 可分
```

它就很难学习“去 receiver 信息”。即使本地有多个 day，也只是去本地 day 或局部 receiver-day 组合，不等于全局去 receiver/day。

### 4.5 MixStyle 跨域配对受限

`same_tx_crossdomain` 需要：

```text
同 TX
不同域
```

联邦客户端本地可能没有同 TX 的跨 receiver 样本。因此 MixStyle 会退化为：

```text
同客户端内部风格扰动
```

而不是集中式的全局 receiver/day 风格混合。

### 4.6 same-TX consistency 受限

集中式可以拉近：

```text
TX0 @ RX0 day0
TX0 @ RX3 day1
```

但普通 FL 客户端本地拿不到其他 receiver 的样本，无法直接构造这个对比。

### 4.7 hard-domain CE 缺少全局 hard domain 视角

客户端只能知道自己的局部难样本或局部难 day，服务器只看到聚合指标。

因此普通 FL 很难做到：

```text
在同一个 step 里识别全局最难 receiver/day 域并加权优化。
```

### 4.8 服务器事后做 GRL 不等价

如果客户端上传的是普通模型 delta：

```text
delta = update from CE + domain + Fishr + sat + regularization + optimizer state
```

服务器已经无法分辨其中哪一部分来自：

```text
CE gradient
domain head gradient
GRL adversarial gradient
Fishr gradient
satellite loss gradient
```

所以服务器不能在聚合前简单地“做 GRL”或“把某部分反号”。GRL 必须作用在 domain loss 对 encoder 的那条梯度路径上，必须在反向传播图里完成。

---

## 5. 若要最大限度接近集中式，有哪些方案

### 方案 A：共享或流式上传 IQ

做法：

```text
clients upload IQ samples or stream mini-batches
server builds global multi-domain batch
server runs original BEX02 training
```

等价性：

```text
最高，几乎就是集中式训练。
```

优点：

- Fishr、GRL、MixStyle、same-TX consistency、hard-domain CE 都保持原始语义。
- optimizer、scheduler、checkpoint 选择完全一致。
- 实验解释最干净。

代价：

- 隐私最弱。
- 通信成本高。
- 可能违反 FL 场景初衷。

适用：

- 只追求严格复刻集中式 BEX02。
- 数据可集中或可在可信服务器临时流式使用。

### 方案 B：Split Learning

做法：

```text
client: IQ -> early feature
server: collect features from multiple clients -> global multi-domain feature batch
server: run BEX02 heads/losses
server: send feature gradients back to clients
```

关键：

```text
split 点必须足够早，使服务器仍能执行 MixStyle/Fishr/GRL/域损失。
```

等价性：

```text
高，但取决于 split 点。
```

优点：

- 不直接上传原始 IQ。
- 服务器可以拼接跨 receiver/day 的全局 batch。
- GRL/Fishr 可以保留较强语义。

代价：

- 上传 activation 仍有隐私泄露风险。
- 通信量大。
- 训练系统复杂，需要同步前向和反向。
- 客户端掉线会影响 step。

适用：

- 想最大化集中式等价性，但不能上传原始 IQ。

### 方案 C：同步 FedSGD / Distributed Centralized Training

做法：

```text
server controls global sampler
each global step selects samples from multiple clients
clients compute gradients for their shard
server aggregates gradients for this single step
server maintains optimizer state
server updates global model
server broadcasts new model
```

这不是普通 FedAvg，而是：

```text
data-distributed centralized training
```

关键要求：

- `local_epochs` 不能大于 1。
- 最好是每个 global step 同步一次。
- optimizer state 必须在服务器统一维护。
- server 需要知道或构造全局 batch 的 domain composition。

等价性：

```text
中高。CE 类梯度可以非常接近集中式；复杂跨样本 loss 需要额外设计。
```

优点：

- 不必上传原始 IQ。
- 比 FedAvg 更接近集中式 optimizer 语义。
- 可以减少 client drift。

困难：

- Fishr 和 hard-domain CE 需要跨客户端域统计。
- MixStyle 和 same-TX consistency 需要跨客户端样本配对，普通梯度聚合不足以完成。
- 如果服务器只聚合最终梯度，不能直接执行需要同一计算图的跨样本操作。

适用：

- 要比 FedAvg 更接近集中式，但不想做完整 Split Learning。

### 方案 D：分解梯度上传

做法：

客户端分别上传不同 loss 分量的梯度或统计：

```text
g_cls
g_dom
g_adv_encoder
g_fishr
g_sat
g_proto
```

服务器按 BEX02 权重组合：

```text
g_server =
    g_cls
  + lambda_dom   * g_dom
  - lambda_adv   * g_adv_encoder
  + lambda_fishr * g_fishr
  + lambda_sat   * g_sat
```

注意：

```text
这不是服务器事后对普通 delta 做 GRL。
```

它要求客户端保留分量梯度，使服务器能区分哪条梯度该反向、哪条该正向。

等价性：

```text
中等。比普通 FedAvg 清楚，但仍缺少真实全局 batch。
```

优点：

- 可以让服务器更精确控制 BEX02 各损失项权重。
- 可分析性强。

代价：

- 通信量增加。
- 梯度泄露风险增大。
- 实现复杂。
- 如果客户端本地仍是单域，Fishr/GRL 的数据基础仍弱。

适用：

- 研究型实验，用来验证“损失分量聚合”是否比 FedAvg delta 更接近集中式。

### 方案 E：StyleBank / ProtoBank / 统计共享

做法：

客户端不上传 IQ，而上传：

```text
RF style statistics
receiver/channel centroids
class prototypes
domain prototypes
gradient/statistical summaries
```

服务器聚合成全局 StyleBank / ProtoBank 后下发。

客户端本地构造虚拟多域 batch：

```text
local clean sample
+ remote receiver style view
+ satellite view
+ constructed d_style domain label
```

然后本地运行：

```text
BEX02 local objective
GRL/Fishr over d_style
baseline satellite view
prototype pull
```

等价性：

```text
中等偏低，但工程上最实用。
```

优点：

- 不共享 IQ。
- 通信成本可控。
- 能给 Fishr/GRL/MixStyle 构造有意义的多域标签 `d_style`。
- 与现有 federated CVS-RFFI/StyleBank 结构契合。

代价：

- 虚拟域不等于真实域。
- StyleBank 质量决定上限。
- 需要成熟度门控：早期 StyleBank 不稳定时不能贸然开 Fishr/GRL。
- 难以宣称严格复刻集中式，只能说是 BEX02-inspired FL-DG。

适用：

- 最推荐的实际 FL 路线。

### 方案 F：普通 FedAvg + 本地 BEX02 objective

做法：

```text
每个 client 本地跑 BEX02 风格 loss
server 做 FedAvg/FedProx
```

可使用：

```text
--train_mode fedavg
--fl_client_key receiver
--fl_local_objective receiver_agnostic_bex02
--fl_sat_aug_mode baseline_view
--use_sat_consistency
--lambda_fishr 0.02
```

等价性：

```text
低到中等，取决于客户端本地域多样性。
```

优点：

- 最容易落地。
- 不共享数据。
- 与现有 FedAvg/FedProx 框架兼容。

缺点：

- 多域 batch 语义最弱。
- Fishr/GRL/MixStyle 可能退化。
- local training drift 明显。
- 不能称作集中式 BEX02 严格复刻。

---

## 6. 各方案对比

| 方案 | 是否共享 IQ | 接近集中式程度 | 实现难度 | 隐私风险 | 通信成本 | 主要问题 |
|---|---:|---:|---:|---:|---:|---|
| 服务器集中式训练 | 是 | 最高 | 低 | 最高 | 高 | 不符合 FL 初衷 |
| Split Learning | 否，上传 activation | 高 | 高 | 中高 | 高 | 系统复杂，同步要求高 |
| 同步 FedSGD | 否 | 中高 | 中高 | 中 | 中高 | 跨客户端 loss 仍难 |
| 分解梯度上传 | 否 | 中 | 高 | 中高 | 高 | 需拆分梯度，仍缺真实多域 batch |
| StyleBank/ProtoBank | 否 | 中 | 中 | 中低 | 中低 | 虚拟域质量决定效果 |
| FedAvg + 本地 BEX02 | 否 | 低到中 | 低 | 低 | 低 | 多域 loss 退化 |

---

## 7. 如果要“原汁原味”，推荐实现路线

### 7.1 最严格路线

如果论文或实验目标是：

```text
最大程度复现集中式 BEX02
```

推荐新建一个训练模式：

```text
train_mode = fedsgd_bex02_sync
```

它应该避免普通 FedAvg 的 local epoch 语义，改成：

```text
global step 级同步训练
```

### 7.2 关键设计

服务器负责：

```text
1. 维护全局模型。
2. 维护 AdamW optimizer state。
3. 维护全局 domain sampler。
4. 每个 step 指定来自哪些 clients 的样本组成 global batch。
5. 聚合当前 step 的梯度或 activation-level loss。
6. 更新模型并广播。
```

客户端负责：

```text
1. 保存本地 IQ。
2. 按服务器请求取样。
3. 执行本地前向/部分前向/梯度计算。
4. 上传 activation、分解梯度或必要统计。
```

### 7.3 必须保持的集中式语义

要最大限度接近 BEX02，必须保住：

```text
global multi-domain batch
fishr_min_domains 真实满足
GRL 作用在 domain loss -> encoder 的梯度路径上
MixStyle 能跨 receiver/day 风格配对
same-TX consistency 能跨 client 找同 TX
hard-domain CE 有全局 hard domain 统计
satellite clean+sat view 参与同一 supervised objective
server-side single optimizer state
```

### 7.4 最小可行实现

第一阶段可以不做 full Split Learning，而做同步 FedSGD 近似：

```text
1. server 每轮选择所有 receiver clients。
2. 每个 client 只跑一个 local step。
3. client 上传分解 loss/gradient stats：
   - CE
   - domain/GRL
   - Fishr proxy statistics
   - hard-domain loss statistics
4. server 聚合并更新一次。
5. 每若干 round 做全局 OOD eval。
```

但必须在报告里说明：

```text
这仍是集中式 BEX02 的梯度级近似，不是完全等价。
```

### 7.5 更强实现

第二阶段做 Split Learning：

```text
1. client 前端提取 early feature。
2. server 拼接来自多个 clients 的 feature batch。
3. server 执行 MixStyle、GRL、Fishr、domain/group losses。
4. server 反传 feature gradient。
5. client 更新前端，server 更新后端。
```

这更接近集中式，因为跨域操作发生在同一个服务器计算图中。

---

## 8. 联邦训练中各 BEX02 组件的处理建议

### 8.1 CE 分类

普通 FedAvg 可以处理。

问题：

```text
client label distribution 可能 non-IID。
```

建议：

```text
按 num_samples 聚合，必要时加 FedProx。
```

### 8.2 GRL

不能服务器事后对普通 delta 做。

建议：

```text
在客户端本地计算图中做；
或 Split Learning 中在服务器计算图做；
或上传分解梯度让服务器组合。
```

局限：

```text
如果本地没有多域，GRL 去域目标弱。
```

### 8.3 Fishr

普通 FL 最容易退化。

建议：

```text
使用 StyleBank 构造 d_style；
或同步 FedSGD 收集跨 client 梯度统计；
或 Split Learning 在服务器直接计算。
```

必须监控：

```text
fishr_domain_count
fishr_active
```

### 8.4 MixStyle

普通客户端本地 MixStyle 只能混本地风格。

建议：

```text
用 StyleBank 下发远端 receiver 风格；
本地构造 same-TX cross-style view；
或 Split Learning 在服务器 feature batch 上混。
```

局限：

```text
StyleBank 的虚拟风格可能不等价真实 IQ 风格。
```

### 8.5 hard-domain CE

普通客户端只能知道本地 hard domain。

建议：

```text
服务器聚合 per-domain loss/accuracy statistics；
下一轮下发 domain weights；
客户端按全局 domain weights 加权本地 loss。
```

局限：

```text
延迟一轮，且统计是聚合估计，不是同 step hard-domain selection。
```

### 8.6 same-TX cross-domain consistency

普通 FL 无法直接跨 client 拉近同 TX 样本。

建议：

```text
上传 class/domain prototypes；
或用 StyleBank 生成同 TX 虚拟跨域 view；
或 Split Learning 服务器持有 feature-level batch。
```

局限：

```text
prototype 是均值约束，弱于样本级 consistency。
```

### 8.7 baseline-style satellite view

联邦中可以比较好地本地实现：

```text
local clean sample
local satellite-transformed sample
concat and supervised CE
```

建议：

```text
fl_sat_aug_mode=baseline_view
sat_train_scenario=mixed_orbit
sat_view_prob=1.0
```

局限：

```text
它只能扩展 channel/satellite 视角，不能弥补跨 receiver/day 样本缺失。
```

---

## 9. 推荐实验路线

### 9.1 基线 1：普通 FedAvg CE

目的：

```text
看纯 FL 在 ratio=0.1 下的下限。
```

配置方向：

```bash
--train_mode fedavg
--fl_client_key receiver
--fl_local_objective ce
--wisig_train_ratio 0.1
```

### 9.2 基线 2：FedAvg + receiver_agnostic_bex02

目的：

```text
验证本地 BEX02 objective 在 FL 中能带来多少收益。
```

配置方向：

```bash
--train_mode fedavg
--fl_client_key receiver
--fl_local_objective receiver_agnostic_bex02
--fl_sat_aug_mode baseline_view
--use_sat_consistency
--sat_train_scenario mixed_orbit
--sat_view_prob 1.0
--lambda_fishr 0.02
--fishr_min_domains 4
```

关键监控：

```text
diag_fishr_domain_count
diag_fishr_active
diag_rx_adv_active
diag_baseline_sat_view_active
```

### 9.3 候选 3：FedAvg + StyleBank-BEX02

目的：

```text
用虚拟远端 receiver style 重建多域 batch 语义。
```

配置方向：

```bash
--use_fed_style_bank
--fl_style_replay_start_round 20
--fl_style_phys_start_round 20
--fl_style_dg_start_round 40
--fl_style_dg_min_domains 4
--fl_style_max_views 1
```

判断标准：

```text
style_dg_ready 是否变成 1
diag_fishr_active 是否显著提升
strict UDU 是否超过普通 BEX02-FedAvg
```

### 9.4 候选 4：同步 FedSGD-BEX02

目的：

```text
最大限度逼近集中式 optimizer 和 global step 语义。
```

关键设置：

```text
fl_local_epochs = 1 step
clients_per_round = 1.0
server optimizer state
per-step gradient aggregation
global domain sampler
```

需要新代码支持，不应混同于普通 FedAvg。

### 9.5 候选 5：Split-BEX02

目的：

```text
在不上传 IQ 的情况下，让服务器真正构造 global multi-domain feature batch。
```

这是最像集中式但工程代价最高的方案。

---

## 10. 最终建议

如果目标是工程上尽快提升 FL 表现：

```text
FedAvg/FedProx + receiver clients + receiver_agnostic_bex02 + baseline_view SAT + StyleBank d_style
```

如果目标是论文上严谨地说“最大限度复刻集中式 BEX02”：

```text
不要用普通 FedAvg 表述为复刻。
应该实现 fedsgd_bex02_sync 或 Split-BEX02。
```

如果目标是完全原汁原味：

```text
共享或流式 IQ 到服务器，直接集中式训练。
```

最重要的判断标准不是参数名是否相同，而是日志和训练机制是否证明：

```text
1. 同一训练 step 内存在多 receiver/day 域。
2. Fishr 的有效域数满足 fishr_min_domains。
3. GRL 作用在正确的 domain-loss -> encoder 梯度路径上。
4. MixStyle 能跨真实或构造域配对。
5. hard-domain CE 使用全局或近似全局的难域信息。
6. satellite view 是 clean+sat supervised expansion，而不是误用成弱辅助一致性。
```

只有这些成立，联邦训练才是在逼近集中式 BEX02；否则只是“使用了 BEX02 的若干 loss 名称”。

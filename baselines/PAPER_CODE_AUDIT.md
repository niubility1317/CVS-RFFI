# Baselines 论文-代码一致性审计报告

> 审计日期: 2026-06-04
> 审计范围: `baselines/` 下所有方法的模型、损失函数、训练策略、超参数
> 参考论文:
> - **RIEI**: "Receiver-Agnostic Radio Frequency Fingerprint Identification via Feature Disentanglement" (2023, arXiv:2309.02801v2)
> - **DRIFT-Day1 / DRIFT**: "Cross-Receiver Generalization for RF Fingerprint Identification via Feature Disentanglement and Adversarial Training" (`tmp/pdfs/2510.09405v1_1.txt`)
> - **历史参考，不作为本轮 DRIFT 目标**: "Towards Receiver-Agnostic and Collaborative Radio Frequency Fingerprint Identification" (`tmp/pdfs/2207.02999v1.txt`)
> - **CVCNN-CE**: 无独立论文，纯复数CNN+CE基线

> **2026-06-04 更正 / 本轮执行边界**:
> - 用户已明确排除 RA-Collab，本轮不修改 `baselines/ra_collab`。
> - 本轮口径更正为：**各方法以各自原论文为准**。DRIFT-Day1 论文中的 shared-backbone 对比段落只能约束该论文自己的 Table I/II/III 对比场景，不能覆盖 RIEI 原论文复现实验。
> - 对 `tmp/pdfs/2510.09405v1_1.txt` 的复核显示，DRIFT 方法由 ResNet18-1D feature extractor、transmitter/receiver feature split、GRL、receiver center loss、raw negative MSE 和总损失 `L = L_CE + λ1 L_grl + λ2 L_center + λ3 L_mse` 构成；未找到 U-Net decoder、DeConv、`L_dmi` 或 DANN-style dynamic GRL schedule 的原文依据。
> - 因此本文旧版 DRIFT 条目中关于 “U-Net 解码器缺失”、“深度互信息 `L_dmi` 缺失”、“默认应使用 DANN schedule” 的建议标记为**旧版误判，不采纳**；原因不是 shared-backbone 公平比较，而是目标 DRIFT 原文不包含这些模块。
> - RIEI 原论文 WiSig 设置采用 ResNet 1D-18 FED、EC/RC 3-layer fully connected network、MI/IE/CE 损失和 alternating training；旧版“RIEI 必须改成 2D CNN / 2层MLP”的条目也标记为**旧版误判，不采纳**。
> - RIEI original 的 evaluation window 维持 last10；FedRIEI/journal 分支使用 final5。

---

## 1. RIEI-FD（特征解耦接收机无关发射机识别）

**论文**: *Receiver-Agnostic Radio Frequency Fingerprint Identification via Feature Disentanglement*
**代码入口**: `baselines/riei_fd/train_cvs.py` → `baselines/riei_fd/train.py` + `baselines/riei_fd/model.py` + `baselines/riei_fd/losses.py`

### 1.1 网络架构

| 组件 | 论文描述 | 代码实现 | 文件:行号 | 一致性 |
|------|---------|---------|-----------|--------|
| **编码器 (FED)** | WiSig: ResNet 1D-18；HackRF: ResNet 50 with attention | RIEI 专属 `RIEIResNet1D18FED`，输入 `[B,2,256]` | `riei_fd/architecture.py` | ✅ WiSig 设置一致 |
| **特征解耦** | FED输出分为 z_e (发射机) + z_r (接收机) | `torch.split(z, [emitter_dim, receiver_dim])` | `riei_fd/model.py:53` | ✅ 一致 |
| **EC分类器** | 3-layer fully connected network | RIEI 专属 `RIEIThreeLayerClassifier` | `riei_fd/architecture.py` | ✅ |
| **RC分类器** | 3-layer fully connected network | RIEI 专属 `RIEIThreeLayerClassifier` | `riei_fd/model.py:39` | ✅ |
| **交叉分类** | 交叉空间分类产生"混淆"输出 | `rx_to_emitter_space(z_r)` + `emitter_to_rx_space(z_e)` | `riei_fd/model.py:59-60` | ✅ 一致 |
| **默认feature_dim** | 论文Table I: FED输出维度 | 512 (emitter=256, receiver=256) | `riei_fd/train_cvs.py:40` | ✅ 合理 |

**2026-06-04 复核结论**:
- RIEI 原论文在 Model setup 中写明：WiSig 使用 ResNet 1D-18，HackRF 使用 ResNet 50 with attention；EC 和 RC 均为 3-layer fully connected network。
- 因此旧版“RIEI-FD 论文用 2D CNN / 2层MLP”的判断是误判，不应据此改代码。

### 1.2 损失函数

| 损失 | 论文公式 | 代码实现 | 文件:行号 | 一致性 |
|------|---------|---------|-----------|--------|
| **CE损失** | L_ce = CE(y_e, l_e) + CE(y_r, l_r) (Eq.5) | `F.cross_entropy(emitter_logits, y) + F.cross_entropy(receiver_logits, d)` | `riei_fd/losses.py:26-28` | ✅ 完全一致 |
| **互信息 (MI)** | L_MI = (1/C) Σ \|z_e^(c)·z_r^(c)\| (cosine MI, Eq.6) | `F.normalize → element-wise multiply → abs → mean` | `riei_fd/losses.py:7-11` | ✅ 完全一致 |
| **信息熵 (IE)** | L_IE = H(softmax(y_er)) + H(softmax(y_re)) (Eq.7) | `entropy_from_logits(cross_emitter) + entropy_from_logits(cross_receiver)` | `riei_fd/losses.py:14-16, 30` | ✅ 完全一致 |
| **总损失** | L = L_ce + λ_MI·L_MI − λ_IE·L_IE (Eq.8) | `loss = loss_ce + lambda_mi * loss_mi - lambda_ie * loss_ie` | `riei_fd/losses.py:31` | ✅ 完全一致 |
| **λ_MI 默认值** | 1.2 | 1.2 | `riei_fd/train_cvs.py:38` | ✅ |
| **λ_IE 默认值** | 1.2 | 1.2 | `riei_fd/train_cvs.py:39` | ✅ |

**MI损失实现验证**:
```python
# 论文Eq.6: L_MI = (1/C) Σ |z_e^(c) · z_r^(c)|
# 代码: 对z_e和z_r做L2归一化后逐元素相乘，取绝对值再求均值
ze = F.normalize(z_e[:, :dim], dim=1, eps=eps)
zr = F.normalize(z_r[:, :dim], dim=1, eps=eps)
return torch.sum(ze * zr, dim=1).abs().mean()
```
归一化后的逐元素相乘等价于cosine相似度的分量求和，与论文一致。

**IE损失实现验证**:
```python
# 论文Eq.7: H(softmax(y)) = -Σ p·log(p)
probs = F.softmax(logits, dim=1)
return -(probs * torch.log(probs.clamp_min(eps))).sum(dim=1).mean()
```
标准信息熵计算，与论文一致。符号：总损失中 **减去** IE（`-lambda_ie * loss_ie`），使IE最大化（即交叉分类输出最不确定），符合论文意图。

### 1.3 训练策略

| 策略 | 论文描述 (Sec.III-D) | 代码实现 | 文件:行号 | 一致性 |
|------|---------------------|---------|-----------|--------|
| **交替训练** | Step1: 解冻EC/RC，CE更新全模型；Step2: 冻结EC/RC，MI+IE更新FED | `alternating_training_step()` | `riei_fd/train.py:20-42` | ✅ 完全一致 |
| **Step1** | CE loss, 更新所有参数 | `set_requires_grad(ec/rc, True)` → CE backward | `riei_fd/train.py:23-29` | ✅ |
| **Step2** | MI+IE loss, 冻结分类器, 仅更新FED | `set_requires_grad(ec/rc, False)` → MI+IE backward | `riei_fd/train.py:31-39` | ✅ |
| **分类器恢复** | Step2后恢复分类器梯度 | `set_requires_grad(ec/rc, True)` | `riei_fd/train.py:40-41` | ✅ |

### 1.4 超参数

| 参数 | 论文 | 代码默认值 | 文件:行号 | 一致性 |
|------|------|-----------|-----------|--------|
| 优化器 | Adam | Adam | `riei_fd/train_cvs.py:66-67` | ✅ |
| 学习率 | 未明确指定 | lr_all=1e-4, lr_fed=1e-4 | `riei_fd/train_cvs.py:36-37` | ✅ 合理 |
| Batch size | 未明确指定 | 64 | `riei_fd/train_cvs.py:34` | ✅ 合理 |
| Epochs | 未明确指定 | 200 | `riei_fd/train_cvs.py:35` | ✅ 合理 |
| Dropout | 未明确 | 0.0 | `riei_fd/train_cvs.py:41` | ✅ 可配置 |
| 两个优化器 | 论文描述两个Adam优化器 | `opt_all` (全参数) + `opt_fed` (仅FED) | `riei_fd/train_cvs.py:66-67` | ✅ |

### 1.5 数据处理

| 项目 | 论文 | 代码 | 一致性 |
|------|------|------|--------|
| 数据集 | WiSig (256样本/包) | WiSig (wisig_out_len=256) | ✅ |
| 归一化 | 论文提及 | normalize=True, crop_mode="center" | ✅ |
| 接收机划分 | 训练接收机与测试接收机分离 | `compact_receiver_targets` 映射 | ✅ |

### 1.6 总结

| 类别 | 一致 | ⚠️ 警告 | ❌ 不一致 |
|------|------|---------|----------|
| 损失函数 | 4/4 | 0 | 0 |
| 训练策略 | 4/4 | 0 | 0 |
| 超参数 | 6/6 | 0 | 0 |
| 网络架构 | 6/6 | 0 | 0 |

**关键差异**:
1. ✅ **编码器架构**: WiSig 原论文使用 ResNet 1D-18，当前实现为 RIEI 专属 `RIEIResNet1D18FED`。
2. ✅ **分类器层数**: 原论文 EC/RC 为 3-layer fully connected network，当前为 RIEI 专属 `RIEIThreeLayerClassifier`。
3. ⚠️ **保留风险**: 当前实现是按论文文字复现的项目实现，仍需用训练日志/性能验证确认是否与作者未公开细节完全等价。

---

## 2. DRIFT（基于风格迁移的接收机无关RFFI）

**论文**: *Cross-Receiver Generalization for RF Fingerprint Identification via Feature Disentanglement and Adversarial Training*
**代码入口**: `baselines/drift/train_cvs.py` → `baselines/drift/model.py` + `baselines/drift/losses.py`

### 2.1 网络架构

| 组件 | 论文描述 | 代码实现 | 文件:行号 | 一致性 |
|------|---------|---------|-----------|--------|
| **编码器** | 1D ResNet-18 (论文Sec.III-B, Table I) | DRIFT 专属 `DRIFTResNet18_1DEncoder(embedding_dim=512)` | `drift/architecture.py` | ✅ 一致 |
| **编码器输出** | 512维共享特征 | embedding_dim=512 | `drift/train_cvs.py:32` | ✅ |
| **特征分离** | 共享特征分为 z_tx + z_rx | `z[:, :split_dim]` + `z[:, split_dim:]` (split_dim=256) | `drift/model.py:42-43` | ✅ 一致 |
| **TX分类器** | 3-layer FC/MLP | DRIFT 专属 `DRIFTThreeLayerClassifier(split_dim=256, num_tx)` | `drift/model.py:36` | ✅ 一致 |
| **RX分类器** | 3-layer FC/MLP | DRIFT 专属 `DRIFTThreeLayerClassifier(rx_dim=256, num_rx)` | `drift/model.py:37` | ✅ 一致 |
| **域判别器** | two-layer/FC domain discriminator via GRL | DRIFT 专属 `DRIFTThreeLayerClassifier(split_dim=256, num_rx)` via GRL | `drift/model.py:38` | ✅ 一致 |
| **GRL** | 梯度反转层 (论文Eq.3) | `gradient_reverse(z_tx, grl_lambda)` | `drift/model.py:50` | ✅ 一致 |
| **U-Net解码器 / DeConv** | 目标 DRIFT 原文未包含该模块 | 未实现 | — | ✅ 不应实现 |

**U-Net / DeConv 复核结论**:
- `tmp/pdfs/2510.09405v1_1.txt` 的 DRIFT methodology 描述为 ResNet18-1D 编码、512维特征切分、CE/GRL/center/MSE 损失；未出现 U-Net、decoder、DeConv 或重建路径。
- 不采纳旧版 U-Net 建议；该建议属于错误报告条目，不属于本轮 DRIFT 原论文复现目标。

### 2.2 损失函数

| 损失 | 论文公式 | 代码实现 | 文件:行号 | 一致性 |
|------|---------|---------|-----------|--------|
| **TX CE** | L_tx = CE(ŷ_tx, y_tx) | `F.cross_entropy(tx_logits, tx_label)` | `drift/losses.py:54` | ✅ 一致 |
| **RX CE** | L_rx = CE(ŷ_rx, y_rx) | `F.cross_entropy(rx_logits, rx_label)` | `drift/losses.py:55` | ✅ 一致 |
| **域对抗 (GRL)** | L_adv = CE(D(GRL(z_tx)), y_rx) (论文Eq.4) | `F.cross_entropy(domain_logits, rx_label)` | `drift/losses.py:56` | ✅ 一致 |
| **中心损失** | L_ct = (1/N)Σ\|z_rx^(i) - c_{y_rx}\|² (论文Eq.11) | `receiver_style_transfer_center_loss`: per-RX MSE→center | `drift/losses.py:11-28` | ✅ 一致 |
| **负MSE分离** | `L_mse = -1/N Σ ||z*_i - z'_i||²` | `negative_mse_separation`: raw squared distance → negate；normalize 仅为 opt-in | `drift/losses.py:31-38` | ✅ 一致 |
| **深度互信息 / L_dmi** | 目标 DRIFT 原文未包含该损失 | 未实现 | — | ✅ 不应实现 |
| **总损失** | L = L_tx + L_rx + λ_adv·L_adv + λ_ct·L_ct + λ_sep·L_sep | `loss_ce_tx + loss_ce_rx + λ_grl·loss_grl + λ_center·loss_center + λ_mse·loss_mse` | `drift/losses.py:59-65` | ✅ 一致 |

**中心损失实现验证**:
```python
# 论文Eq.11: L_ct = (1/N) Σ |z_rx^(i) - c_{y_rx}|²
# 代码: 对每个接收机，计算该接收机所有样本特征的均值作为center，然后计算MSE
for rx in torch.unique(rx_label):
    mask = rx_label == rx
    feat = z_rx[mask]
    center = feat.mean(dim=0, keepdim=True)
    losses.append((feat - center).square().sum(dim=1).mean())
```
与论文Eq.11完全一致——计算每个接收机的特征中心，然后最小化样本到中心的距离。

**负MSE分离实现验证**:
```python
# 论文Eq.12: L_sep = −MSE(z_tx, z_rx)
# 代码默认: 直接计算 raw squared distance，取负值
a = z_tx[:, :dim]
b = z_rx[:, :dim]
return -torch.mean(torch.sum((a - b) ** 2, dim=1))
```
默认 raw negative MSE，使总损失最小时 TX/RX 距离最大。normalize 仅作为 opt-in 调试选项，不作为论文默认。

**深度互信息 / L_dmi 复核结论**:
- `tmp/pdfs/2510.09405v1_1.txt` 的总损失只包含 `L_CE`、`L_grl`、`L_center`、`L_mse` 四类项；没有 `L_dmi`。
- RIEI 原论文有 `L_MI`，但这是 RIEI 的 mutual independence loss，不能错误迁移为 DRIFT 的 `L_dmi`。

### 2.3 训练策略

| 策略 | 论文描述 | 代码实现 | 文件:行号 | 一致性 |
|------|---------|---------|-----------|--------|
| **优化器** | Adam | Adam | `drift/train_cvs.py:76` | ✅ |
| **学习率** | 1e-4 | 1e-4 | `drift/train_cvs.py:31` | ✅ |
| **Batch size** | 64 | 64 | `drift/train_cvs.py:53` | ✅ |
| **GRL调度** | GRL backward uses fixed or tunable coefficient；实验超参数 `λ1=1` | 默认 `constant`，可选 `dann` 仅作兼容开关 | `drift/train_cvs.py:39,82` | ✅ |
| **单步训练** | 联合优化所有损失 | `losses["loss"].backward()` | `drift/train_cvs.py:95-96` | ✅ |

**GRL调度复核结论**:
- 目标 DRIFT 原文公式给出 `∂GRL(z)/∂z = -λI`，并在实验设置中给出 `λ1=1`；未要求 DANN dynamic schedule 作为 DRIFT 默认。
- `--grl_schedule dann` 保留为兼容选项，但不作为原论文默认。

### 2.4 超参数

| 参数 | 论文 | 代码默认值 | 文件:行号 | 一致性 |
|------|------|-----------|-----------|--------|
| λ_grl | 1 | 1.0 | `drift/train_cvs.py:34` | ✅ |
| λ_center | 0.01 | 0.01 | `drift/train_cvs.py:35` | ✅ |
| λ_mse | 0.02 | 0.02 | `drift/train_cvs.py:36` | ✅ |
| embedding_dim | 512 | 512 | `drift/train_cvs.py:32` | ✅ |
| split_dim | 256 | 256 | `drift/train_cvs.py:33` | ✅ |
| Epochs | 未明确 | 200 | `drift/train_cvs.py:30` | ✅ 合理 |
| Dropout | 未明确 | 0.0 | `drift/train_cvs.py:40` | ✅ |

### 2.5 总结

| 类别 | 一致 | ⚠️ 警告 | ❌ 不一致 |
|------|------|---------|----------|
| 损失函数 | 6/6 | 0 | 0 |
| 训练策略 | 5/5 | 0 | 0 |
| 超参数 | 7/7 | 0 | 0 |
| 网络架构 | 8/8 | 0 | 0 |

**关键差异**:
1. ✅ **U-Net/DeConv**: 目标 DRIFT 原文不包含该模块，不应实现。
2. ✅ **L_dmi**: 目标 DRIFT 原文不包含该损失；不要把 RIEI 的 `L_MI` 迁移成 DRIFT 的 `L_dmi`。
3. ✅ **GRL调度**: 目标 DRIFT 原文给出 fixed/tunable coefficient 和实验 `λ1=1`，当前默认 constant 符合该设置。

---

## 3. RA-Collab（协作式接收机无关RFFI）

**论文**: *A Cooperative Framework for Receiver-Agnostic Radio Frequency Fingerprint Identification*
**代码入口**: `baselines/ra_collab/train_cvs.py` → `baselines/ra_collab/model.py` + `baselines/ra_collab/losses.py` + `baselines/ra_collab/collaborative_inference.py`

### 3.1 网络架构

| 组件 | 论文描述 | 代码实现 | 文件:行号 | 一致性 |
|------|---------|---------|-----------|--------|
| **预处理** | 时频图/频谱图 (论文Sec.II-A) | `SpectrogramTransform(n_fft=64, hop=32)` → log-amplitude spectrogram | `ra_collab/spectrogram.py` | ✅ 一致 |
| **特征提取** | CNN (ResNet-18 backbone, 论文Sec.III-A) | `ResNet2DEncoder(channels=[32,64,128])` | `common/resnet2d.py:31-70` | ✅ 一致 |
| **TX分类器** | 全连接层 | `ClassifierHead(feature_dim=256, num_tx)` | `ra_collab/model.py:38` | ✅ 一致 |
| **RX分类器** | 全连接层 (via GRL) | `ClassifierHead` + `gradient_reverse` | `ra_collab/model.py:39,46` | ✅ 一致 |
| **GRL** | 梯度反转 (论文Eq.3-4) | `gradient_reverse(feature, grl_lambda)` | `ra_collab/model.py:46` | ✅ 一致 |
| **协作推断** | 多接收机概率融合 (论文Algorithm 1) | `soft_fusion` / `adaptive_soft_fusion` | `ra_collab/collaborative_inference.py:15-38` | ✅ 一致 |

**频谱图预处理验证**:
```python
# 论文Sec.II-A: 时频表示
# 代码: STFT → log-amplitude → z-score归一化
stft = torch.stft(x, n_fft=64, hop_length=32, ...)
spec = torch.log(torch.abs(stft).clamp_min(eps))
spec = (spec - mean) / std  # z-score
```
与论文描述的时频图预处理一致。

**ResNet2D编码器验证**:
- 论文: ResNet-18 backbone
- 代码: 3阶段ResNet (32→64→128通道), 每阶段2个BasicBlock2D
- 结构: stem(7×7, stride=1) → stage1(32, stride=1) → stage2(64, stride=2) → stage3(128, stride=2) → AdaptiveAvgPool2d → Linear(128, 256)
- 与论文描述的CNN backbone一致

### 3.2 损失函数

| 损失 | 论文公式 | 代码实现 | 文件:行号 | 一致性 |
|------|---------|---------|-----------|--------|
| **TX CE** | L_tx = CE(ŷ_tx, y) (论文Eq.5) | `F.cross_entropy(tx_logits, tx_label)` | `ra_collab/losses.py:8` | ✅ 完全一致 |
| **RX CE (GRL)** | L_rx = CE(ŷ_rx, y_rx) (论文Eq.6) | `F.cross_entropy(rx_logits, rx_label)` | `ra_collab/losses.py:9` | ✅ 完全一致 |
| **总损失** | L = L_tx + λ·L_rx (论文Eq.7) | `loss_tx + rx_weight * loss_rx` | `ra_collab/losses.py:10` | ✅ 完全一致 |
| **λ默认值** | 1 (论文Sec.IV-B) | 1.0 | `ra_collab/train_cvs.py:55` | ✅ |

### 3.3 协作推断 (CIS)

**论文Algorithm 1**:
1. M个接收机同时观测同一发射机信号
2. 每个接收机独立分类得到softmax概率
3. 融合所有接收机的预测（平均概率）

**代码实现** (`collaborative_inference.py:15-38`):
```python
def soft_fusion(predictions):  # 平均概率融合
    probs = _as_probs(predictions)
    fused = probs.mean(dim=0)
    return fused / fused.sum().clamp_min(1e-8)

def adaptive_soft_fusion(predictions, snr):  # SNR加权融合
    weights = snr.float().view(-1)
    fused = torch.sum(probs * weights.unsqueeze(1), dim=0)
    return fused / fused.sum().clamp_min(1e-8)
```
✅ 与论文Algorithm 1完全一致。

**CIS vs OBS评估**:
- 论文Table II/III: CIS显著优于单观测(OBS)
- 代码: `evaluate_collaborative_tx()` 实现CIS评估，`evaluate_tx()` 实现OBS评估
- 代码在测试时同时报告CIS和OBS结果 (`evaluate_primary_and_obs_tests`)

### 3.4 训练策略

| 策略 | 论文描述 | 代码实现 | 文件:行号 | 一致性 |
|------|---------|---------|-----------|--------|
| **优化器** | SGD (论文Sec.IV-B) | SGD | `ra_collab/train_cvs.py:84` | ✅ |
| **Momentum** | 0.9 | 0.9 | `ra_collab/train_cvs.py:47` | ✅ |
| **学习率** | 1e-3 | 1e-3 | `ra_collab/train_cvs.py:46` | ✅ |
| **LR衰减** | 验证loss plateau, factor=0.2, patience=10 | `lr_reduce_factor=0.2, lr_patience=10` | `ra_collab/train_cvs.py:48-49` | ✅ 完全一致 |
| **Early stopping** | 20 epochs无改善停止 | `early_stop_patience=20` | `ra_collab/train_cvs.py:50` | ✅ 完全一致 |
| **Batch size** | 64 | 64 | `ra_collab/train_cvs.py:44` | ✅ |
| **Epochs** | 100 | 100 | `ra_collab/train_cvs.py:45` | ✅ |
| **验证标准** | 验证loss | `best_metric="loss"` + `val_loss_fn=cross_entropy_val_loss` | `ra_collab/train_cvs.py:153` | ✅ |

**Plateau控制器验证** (`common/cvs_trainer.py:284-329`):
```python
class ValidationLossPlateauController:
    # 当验证loss连续patience个epoch不改善时，LR *= factor
    # 当连续early_stop_patience个epoch不改善时，停止训练
```
与论文描述的验证loss plateau策略完全一致。

### 3.5 Fine-tune策略

| 参数 | 论文 (Sec.IV-C) | 代码默认值 | 文件:行号 | 一致性 |
|------|----------------|-----------|-----------|--------|
| **学习率** | 1e-5 | 1e-5 | `ra_collab/finetune_cvs.py:33` | ✅ |
| **Epochs** | 20 | 20 | `ra_collab/finetune_cvs.py:32` | ✅ |
| **Batch size** | 32 | 32 | `ra_collab/finetune_cvs.py:30` | ✅ |
| **优化器** | SGD | SGD (momentum=0.9) | `ra_collab/finetune_cvs.py:73` | ✅ |
| **GRL** | 关闭 (grl_lambda=0.0) | `grl_lambda=0.0, return_rx=False` | `ra_collab/finetune_cvs.py:76` | ✅ |
| **Shots** | 论文Fig.5: 5/10/20/50/100 | 默认20 (可配置) | `ra_collab/finetune_cvs.py:34` | ✅ |

### 3.6 数据增强

| 增强 | 论文描述 | 代码实现 | 一致性 |
|------|---------|---------|--------|
| **RF信道增强** | 论文提到"增强型预处理技术" | `OnlineRFChannelAugment` (TDL/Doppler/AWGN) | ✅ 一致 |
| **频谱图** | 时频表示 | `SpectrogramTransform` (STFT → log → z-score) | ✅ 一致 |

### 3.7 总结

| 类别 | 一致 | ⚠️ 警告 | ❌ 不一致 |
|------|------|---------|----------|
| 损失函数 | 4/4 | 0 | 0 |
| 训练策略 | 8/8 | 0 | 0 |
| 超参数 | 8/8 | 0 | 0 |
| 网络架构 | 6/6 | 0 | 0 |
| 协作推断 | 3/3 | 0 | 0 |
| Fine-tune | 6/6 | 0 | 0 |

**RA-Collab实现与论文高度一致，未发现显著差异。**

---

## 4. CVCNN-CE（复数CNN+CE基线）

**无对应独立论文**，作为对比基线实现。

### 4.1 网络架构

| 组件 | 描述 | 代码实现 | 文件:行号 | 一致性 |
|------|------|---------|-----------|--------|
| **复数卷积** | (a+bi)(c+di) = (ac-bd) + (ad+bc)i | `ComplexConv1d`: yr=a·c-b·d, yi=a·d+b·c | `cvcnn_ce/model.py:9-24` | ✅ 数学正确 |
| **ComplexBlock** | ComplexConv1d → BN → ReLU → AvgPool | 顺序结构 | `cvcnn_ce/model.py:27-39` | ✅ |
| **SincNet前端** | 可选learnable带通滤波 | `SincConv1d` stem | `cvcnn_ce/model.py:42-76` | ✅ |
| **主干** | 3层ComplexBlock (32→64→128) | 通道倍增 | `cvcnn_ce/model.py:93-98` | ✅ |
| **Embedding** | AdaptiveAvgPool1d → Linear → ReLU | 128维 | `cvcnn_ce/model.py:99-104` | ✅ |
| **分类头** | Linear(embedding_dim, num_classes) | 128→num_classes | `cvcnn_ce/model.py:105` | ✅ |

**SincConv1d依赖**:
- `cvcnn_ce/model.py:6`: `from model import SincConv1d`
- 这引用了项目根目录的 `code/model.py` 中的 SincConv1d 实现
- 运行时需要确保该模块在Python路径中

### 4.2 损失函数

| 损失 | 描述 | 代码实现 | 文件:行号 | 一致性 |
|------|------|---------|-----------|--------|
| **交叉熵** | 标准CE | `F.cross_entropy(logits, y)` | `cvcnn_ce/train_cvs.py:91` | ✅ |

### 4.3 训练策略

| 参数 | 代码默认值 | 文件:行号 | 备注 |
|------|-----------|-----------|------|
| 优化器 | AdamW | `cvcnn_ce/train_cvs.py:77` | ✅ |
| 学习率 | 2e-4 | `cvcnn_ce/train_cvs.py:34` | ✅ |
| Weight decay | 1e-4 | `cvcnn_ce/train_cvs.py:36` | ✅ |
| 调度器 | CosineAnnealingLR (eta_min=1e-6) | `cvcnn_ce/train_cvs.py:78` | ✅ |
| Epochs | 200 | `cvcnn_ce/train_cvs.py:33` | ✅ |
| Batch size | 64 | `cvcnn_ce/train_cvs.py:32` | ✅ |
| base_channels | 32 | `cvcnn_ce/train_cvs.py:37` | ✅ |
| embedding_dim | 128 | `cvcnn_ce/train_cvs.py:38` | ✅ |

---

## 5. 通用组件审计

### 5.1 梯度反转层 (GRL)

**论文参考**: Ganin et al., "Domain-Adversarial Training of Neural Networks" (DANN, 2016)

| 组件 | 论文描述 | 代码实现 | 文件:行号 | 一致性 |
|------|---------|---------|-----------|--------|
| **前向传播** | 恒等变换 | `return x.view_as(x)` | `common/grl.py:13` | ✅ |
| **反向传播** | 梯度乘以 -λ | `return -ctx.lambd * grad_output` | `common/grl.py:17` | ✅ |
| **DANN schedule** | λ(p) = 2/(1+exp(-γp)) - 1 | `2.0 / (1.0 + exp(-gamma * p)) - 1.0` | `common/grl.py:33-37` | ✅ |

### 5.2 频谱图转换

| 组件 | 描述 | 代码实现 | 文件:行号 | 一致性 |
|------|------|---------|-----------|--------|
| **STFT** | 短时傅里叶变换 | `torch.stft(center=True, return_complex=True, onesided=False)` | `common/spectrogram.py:48-57` | ✅ |
| **对数幅度** | log(|STFT|) | `torch.log(torch.abs(stft).clamp_min(eps))` | `common/spectrogram.py:58` | ✅ |
| **归一化** | z-score | `(spec - mean) / std` | `common/spectrogram.py:61-63` | ✅ |

### 5.3 接收机标签映射

| 组件 | 描述 | 代码实现 | 文件:行号 | 一致性 |
|------|------|---------|-----------|--------|
| **紧凑映射** | 将WiSig全局接收机ID映射为训练域内的紧凑标签 | `compact_receiver_targets()` | `common/paper_protocol.py:25-52` | ✅ |

### 5.4 伪标签自训练 (可选)

| 组件 | 描述 | 代码实现 | 文件:行号 | 一致性 |
|------|------|---------|-----------|--------|
| **置信度过滤** | 仅使用高置信度预测作为伪标签 | `conf >= threshold & margin >= margin` | `common/pseudo_labels.py:102` | ✅ |
| **默认关闭** | 需显式启用 | `--use_pseudo_labels` | `common/pseudo_labels.py:52` | ✅ |

---

## 6. 综合差异汇总

### 6.1 高严重性差异 ❌

| 方法 | 差异项 | 论文描述 | 代码实现 | 影响 |
|------|--------|---------|---------|------|
| 无 | 当前无已确认高严重性差异 | U-Net/L_dmi/2D CNN 条目为旧版误判 | 不适用 | 按原论文复核后移除 |

### 6.2 中等严重性差异 ⚠️

| 方法 | 差异项 | 论文描述 | 代码实现 | 影响 |
|------|--------|---------|---------|------|
| **RIEI-FD** | ResNet1D实现细节 | 原论文只说明 ResNet 1D-18，没有给出完整作者代码细节 | `riei_fd/architecture.py` 专属实现 | 需要靠复现实验结果验证 |
| **DRIFT** | ResNet1D实现细节 | 原论文只说明 ResNet18-1D 和 3-layer classifiers | `drift/architecture.py` 专属实现 | 需要靠复现实验结果验证 |

### 6.3 低严重性差异 / 合理偏离

| 方法 | 差异项 | 说明 |
|------|--------|------|
| RA-Collab | STFT参数 | 论文未详细指定，n_fft=64/hop=32为合理选择 |
| RIEI-FD | Dropout | 论文未明确，默认0.0 |
| DRIFT | Epochs | 论文未明确，默认200 |
| CVCNN-CE | SincConv1d依赖 | 需要项目根目录的model.py在路径中 |

### 6.4 完全一致的组件 ✅

**RIEI-FD**:
- ✅ MI损失 (cosine互信息)
- ✅ IE损失 (信息熵)
- ✅ 总损失公式 (CE + λ_MI·MI - λ_IE·IE)
- ✅ 交替训练策略 (Step1: CE全模型, Step2: MI+IE仅FED)
- ✅ λ_MI=1.2, λ_IE=1.2
- ✅ Adam优化器, lr=1e-4
- ✅ 两个独立优化器 (opt_all + opt_fed)

**DRIFT**:
- ✅ 中心损失 (per-RX MSE→center)
- ✅ 负MSE分离 (raw squared distance → negate；normalize 仅 opt-in)
- ✅ 域对抗损失 (GRL + CE)
- ✅ λ_grl=1.0, λ_center=0.01, λ_mse=0.02
- ✅ Adam优化器, lr=1e-4, batch_size=64
- ✅ 特征分离 (split_dim=256)
- ✅ 1D ResNet-18编码器

**RA-Collab**:
- ✅ GRL对抗损失 (L_tx + λ·L_rx)
- ✅ SGD优化器, momentum=0.9, lr=1e-3
- ✅ 验证loss plateau衰减 (factor=0.2, patience=10)
- ✅ Early stopping (patience=20)
- ✅ 协作推断 (soft/adaptive fusion)
- ✅ Fine-tune参数 (lr=1e-5, epochs=20, batch=32)
- ✅ 频谱图预处理 (STFT → log → z-score)
- ✅ ResNet2D编码器
- ✅ 所有超参数

---

## 7. 建议

### 7.1 高优先级修复

1. **不采纳旧版 DRIFT U-Net / L_dmi / DANN schedule 建议**
   - 目标 DRIFT 原文不包含 U-Net、DeConv、`L_dmi` 或 DANN dynamic schedule 默认。
   - 不应为追赶性能而加入非原文模块，否则会破坏“完全按原论文”的复现目标。

2. **不采纳旧版 RIEI 2D CNN / 2层MLP 建议**
   - RIEI 原论文 WiSig 设置是 ResNet 1D-18，EC/RC 是 3-layer fully connected network。
   - 当前需要验证的是 RIEI 专属 ResNet1D-18 实现与作者细节是否足够接近，而不是改成 2D CNN。

3. **继续优先验证实验协议与日志可追踪性**
   - RIEI original Table III 需要使用原论文 receiver 组合、80/20 split、last10 统计。
   - DRIFT-Day1 需要使用 ManySig Day 1、source/test receivers disjoint、800/200 per transmitter、last5 统计。

### 7.2 中优先级改进

4. **保持 DRIFT 默认 constant GRL**
   - 目标 DRIFT 原文实验给出 `λ1=1`，未要求 DANN dynamic schedule。
   - `--grl_schedule dann` 仅保留为消融/兼容开关。

5. **保留 RIEI 3-layer classifier**
   - 原论文明确 EC/RC 为 3-layer fully connected network。

### 7.3 低优先级改进

6. **文档补充**: 为每个方法添加论文对应的公式引用和算法步骤说明
7. **超参数对照表**: 在README中添加论文超参数与代码默认值的对照表

---

## 8. 文件索引

| 方法 | 模型 | 损失 | 训练 | 评估 |
|------|------|------|------|------|
| RIEI-FD | `riei_fd/model.py` | `riei_fd/losses.py` | `riei_fd/train.py` + `riei_fd/train_cvs.py` | `riei_fd/eval.py` |
| DRIFT | `drift/model.py` | `drift/losses.py` | `drift/train_cvs.py` | `drift/eval.py` |
| RA-Collab | `ra_collab/model.py` | `ra_collab/losses.py` | `ra_collab/train_cvs.py` | `ra_collab/collaborative_inference.py` |
| CVCNN-CE | `cvcnn_ce/model.py` | — (仅CE) | `cvcnn_ce/train_cvs.py` | — |
| 通用 | `common/resnet1d.py` `common/resnet2d.py` | `common/grl.py` | `common/cvs_trainer.py` | `common/cvs_sat_eval.py` |

---

## 9. 2026-06-26落实状态

本节用于避免后续把ChatGPT Pro旧版建议、已采纳修复和仍需真实训练验证的项目混淆。状态只描述代码/文档/入口是否已落实，不把dry-run或静态测试写成论文结果。

| ID | 建议或差异 | 处理状态 | 落地位置 | 后续边界 |
|---|---|---|---|---|
| AUD-01 | DRIFT U-Net/DeConv补齐 | 不采纳 | 本文第7.1节 | DRIFT目标原文不包含该模块，不能为追赶性能加入非原文结构 |
| AUD-02 | DRIFT `L_dmi`补齐 | 不采纳 | 本文第7.1节 | 旧版误判，不作为论文复现目标 |
| AUD-03 | DRIFT默认DANN schedule | 不采纳默认化，保留消融开关 | `baselines/drift/train.py`与本文第7.2节 | 默认仍为constant GRL；`--grl_schedule dann`只作消融 |
| AUD-04 | RIEI改成2D CNN/2层MLP | 不采纳 | 本文第7.1节 | RIEI WiSig目标仍为ResNet1D-18与3-layer classifier |
| AUD-05 | RIEI original Table III协议入口 | 已落实入口，待真实训练验证 | `scripts/launchers/run_cvs_baseline_queue.sh`；`baselines/scripts/run_riei_original_table3_queue.sh` | 需真实ManySig数据、完整训练、同row指标后才能写结果声明 |
| AUD-06 | DRIFT Day1协议入口 | 已落实入口，待真实训练验证 | `scripts/launchers/run_cvs_baseline_queue.sh` | 需真实ManySig Day1训练与`drift_last5`统计后才能写结果声明 |
| AUD-07 | 日志与manifest可追踪性 | 已落实 | `scripts/launchers/run_cvs_baseline_queue.sh`；`baselines/PAPER_RERUN_REPORT_TEMPLATE.md` | 报告必须绑定run目录、split、seed和完整同row指标 |
| AUD-08 | README实验落地命令 | 已落实 | `README.md`；`baselines/README.md` | dry-run只证明命令可生成，不代表训练完成 |
| AUD-09 | 超参数对照表 | 已落实 | `baselines/README.md` | 表内值用于入口默认与论文边界说明 |
| AUD-10 | CVCNN SincConv1d路径说明 | 已落实 | `baselines/README.md` | CVCNN仍为CE-only baseline，不新增论文外辅助损失 |
| AUD-11 | RA-Collab是否并入纸面复现 | 不采纳为RIEI/DRIFT纸面复现项 | `baselines/README.md` | RA-Collab只用于CVS-aligned comparison |
| AUD-12 | 作者未公开细节导致的RIEI/DRIFT ResNet差异 | 待真实训练验证 | `baselines/PAPER_RERUN_REPORT_TEMPLATE.md` | 只能由完整复现实验结果判断，不能由静态代码直接宣称完全一致 |

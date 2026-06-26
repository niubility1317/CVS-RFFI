# 优化星地信道增强设计报告

生成时间：2026-06-22  
工作区：`E:\type10-7`  
输入材料：`AGENTS.md`、`项目.md`、`C:\Users\lh594\Downloads\deep-research-report (3).md`、本地星地增强代码/实验分析、联网论文检索、四个只读子 agent 审查  
状态：方案设计，不包含代码修改，不包含 N607 启动

## 1. 结论

新的星地信道增强主线应收敛为：

```text
物理参数化 satellite view
+ 分场景 curriculum
+ CE-only 隔离训练
+ late weak z_id identity alignment
+ domain DSQ / receiver leakage audit
+ clean 与 satellite 双指标门控
```

不建议回到 full concat，也不建议把 StyleBank、生成式补样或目标接收机样本直接推成主线。当前本地证据显示：`SA16` 的成功点不是“更强卫星增强”，而是 `clean 主路径完整 CVS/DG + satellite view 只做 TX CE + domain DSQ`。参考报告和外部论文进一步支持：单纯加噪不是关键，关键是物理变量受控、同一 TX 在 clean/satellite 视图下身份一致、receiver/day 捷径受控。

本报告提出的首选路线命名为：

```text
CVS-SAT-PAIC
Physics-Aligned Identity-Consistent Satellite Augmentation
物理对齐、身份一致的星地信道增强
```

它不是新的科研场景定义，不改变 `项目.md`。它只是 Phase1 source-domain DG 内的源域派生视图训练策略，以及 Stage2 satellite/LEO target view 的评估/适配准备。

## 2. 协议边界

必须遵守以下边界：

- WiSig/ManySig 是 terrestrial proxy / ground-accessible source-domain family，不是真实卫星训练集。
- satellite augmentation / satellite stress 是 physics-informed deployment stress，不是真实在轨 IQ 验证。
- Phase1/source-only DG 只能使用 `R_s`；不得使用 `R_t` 的样本、统计、BN、prototype、adapter、阈值、伪标签、验证或 early stopping。
- clean view 只能是 control/reference。deployment-primary 必须按 satellite/LEO target view 报告。
- Stage2-A/B/C 的 `R_t`、`Y_old/Y_new/Y_unknown`、support/query 权限不能被增强策略改变。
- unknown query 永远 eval-only，不能用于阈值拟合、伪标签、adapter、prototype 或 verifier。

因此，本报告所有“优化”只允许落在三处：

```text
Phase1:
  源域 clean IQ -> 源域派生 satellite view

Evaluation:
  OOD split -> five-scenario satellite stress

Stage2:
  合规 R_t support/query -> satellite/LEO target view 下的 prototype/gate/adapter 评估
```

## 3. 子 Agent 协同结果

| 子 agent | 任务 | 关键结论 |
|---|---|---|
| A：参考报告提炼 | 读取参考研究报告并映射 CVS | 报告主张物理一致增强、课程学习、双视图一致性、receiver/day 去捷径、真实/合成混编；提升幅度只是工程估计，不能写成 CVS 已验证结果。 |
| B：本地实验与代码审计 | 审计代码和历史结果 | 当前支持物理信道、CE-only、full concat、fed baseline_view；`SA16` 是 central clean/UDU anchor，`FSDG49` 是 FL anchor；full concat 和非 CE-only baseline_view 容易污染 DG。 |
| C：文献检索映射 | 联网找论文并映射 hook | 3GPP NTN、Shadowed Rice、RF 数据增强、GAN-RXA、RF contrastive、MixStyle/DANN/SupCon 都支持“物理扰动 + 身份一致 + 去域捷径”的方向。 |
| D：协议红队 | 检查 Stage2 和声明边界 | 新方案必须继续区分 source-only DG、target-old calibration、seen-new enrollment、unknown rejection；报告/validator 需要硬字段防误放行。 |

四个 agent 的交集很清楚：低风险主线是 CE-only satellite view + curriculum + late weak alignment + DSQ；高风险路径是 full concat、目标域泄漏、把 satellite stress 写成在轨验证。

## 4. 外部文献依据

| 类别 | 文献/来源 | 对本设计的作用 |
|---|---|---|
| 数据与问题定义 | [WiSig, arXiv:2112.15363](https://arxiv.org/abs/2112.15363) | WiSig 明确面向 receiver/channel-agnostic RF fingerprinting，且 receiver/day 变化会显著降低性能；支持 CVS 的 receiver/day DG 主线。 |
| NTN 信道边界 | [3GPP NTN overview](https://www.3gpp.org/technologies/ntn-overview), [TR 38.811](https://www.3gpp.org/dynareport/38811.htm) | 提供 NTN 部署场景、轨道、Doppler、delay、channel modeling 的标准化背景；用于约束 satellite scenario 参数而不是替代真实数据。 |
| NTN 仿真工具 | [Implementation of a Channel Model for NTN in ns-3](https://arxiv.org/abs/2305.05544) | 说明 ns-3 已实现 3GPP NTN channel/antenna model 并对照校准结果验证；未来可作为更真实物理上下文来源。 |
| 卫星衰落模型 | [Abdi et al., Shadowed Rice LMS channel](https://web.njit.edu/~abdi/RiceNakagamiSatellite-Revised.pdf) | 支持当前 LOS/LOO/Rayleigh、Rician/Shadowed-Rice 方向；建议后续把 K 因子和 fade-duration 统计做得更可审计。 |
| RF 增强 | [More Is Better: Data Augmentation for Channel-Resilient RF Fingerprinting](https://genesys-lab.org/papers/DataAugmentationMagazine.pdf) | 支持在 raw IQ 上注入 channel/noise，而不是只做普通 AWGN；对应 `sat_channel.py` 和 `RFFIAugmentor`。 |
| Receiver-agnostic RFFI | [GAN-RXA, arXiv:2303.14312](https://arxiv.org/abs/2303.14312) | 支持 receiver-agnostic feature extractor、closed-set 与 open-set 同时评估；对应 `z_id/z_dom` 和 Stage2 gate。 |
| 自监督/对比 | [Unsupervised Contrastive Learning for Robust RF Device Fingerprinting](https://arxiv.org/abs/2403.04036) | 支持同一 transmission 不同视图构造正样本，缓解 time/channel domain shift；对应 clean-sat positive pair。 |
| 监督对比 | [Supervised Contrastive Learning, NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html) | 支持用同类样本聚合、异类样本分离；可作为小改代码后的 TX-aware identity alignment。 |
| 域去偏 | [Domain-Adversarial Training of Neural Networks, JMLR 2016](https://jmlr.org/papers/v17/15-239.html) | 支持 GRL/DANN 类方法，但在 CVS 中必须只用于抑制 `z_id` receiver/day 泄漏，不能抹掉 TX 指纹。 |
| 域泛化风格混合 | [MixStyle, ICLR 2021](https://arxiv.org/abs/2104.02008) | 支持混合源域 feature statistics 增加域多样性；在 CVS 中应限定为 receiver-balanced / same-TX 合理混合。 |
| 生成式 RF | [RF-Diffusion, MobiCom 2024](https://arxiv.org/abs/2404.09140) | 可作为后期尾部补样探索；不应作为第一优先级，也不能把生成样本写成真实星地样本。 |
| Channel-resilient RFFI | [DeepCRF, arXiv:2411.06925](https://arxiv.org/html/2411.06925v1) | 支持“提取稳定微弱指纹、对抗 channel 条件变化”的方向；对 `z_id` 稳定性和 residual channel 诊断有启发。 |
| 数据元信息 | [SigMF](https://sigmf.org/) | 支持记录 sample rate、frequency、time、annotation、geolocation 等元数据；对应未来 satellite meta audit。 |

## 5. 本地证据约束

已有分析文件给出三个关键事实。

第一，星地增强不是没有注入。`sat_channel.py` 已包含轨道/仰角/路径损耗、天气、大气复衰落、LOS/LOO/Rayleigh、Doppler+CFO、相位噪声、多径、AGC、AWGN、IQ imbalance。问题在于它主要增加 nuisance，不增加 TX identity 信息。

第二，full concat 污染 CVS 的 domain/DG 语义。它复制 `d_raw` 给 satellite view，使已经叠加 satellite style 的样本继续参与 domain CE、GRL、Fishr、group CE 等完整损失。这个路径适合做 baseline-style 对照，不适合作默认主线。

第三，CE-only 更干净但约束不够强。已有 `SA16` 说明 CE-only + domain DSQ 能保 clean/UDU，但 satellite avg/min 只小幅改善；`SA14/SA17/SA24` 更偏 satellite robustness，但 clean strict UDU 会掉。这说明下一步不是加粗 satellite CE，而是补一条轻量身份一致性约束。

核心 anchor：

| anchor | clean/UDU | satellite | 解释 |
|---|---:|---:|---|
| `SA16_ceonly_domain_dsq_r010` | strict UDU 82.78 | SAT avg/min 43.66/39.56 | central clean/UDU 主线，星地小幅改善 |
| `SA14_ID_phase_DSQ` | strict UDU 79.80 | SAT avg/min 47.17/40.99 | satellite robustness 更强，但主指标下降 |
| `FSDG49_fedprox_receiver_ra_bex02_cvs_sat` | best strict UDU 76.295 | mixed/cvs consistency | 当前稳定 FL anchor |
| `FSDG50_baseline_sat` | final strict UDU 70.5167 | baseline_view 粗搬 | 说明非 CE-only baseline_view 不稳 |

## 6. 新设计：CVS-SAT-PAIC

### 6.1 组件一：物理参数化 satellite curriculum

目标：把当前固定/循环 satellite scenarios 改成“从易到难”的训练视图，而不是 epoch 1 起全强度 all-five。

推荐阶段：

| 阶段 | epoch | view prob | scenario | 目的 |
|---|---:|---:|---|---|
| warm-up | 1-40 | 0.20-0.30 | `mixed_orbit` 或 `clear_leo` | 先学 TX identity，避免强扰动早期洗掉指纹 |
| transition | 41-90 | 0.50-0.65 | `mixed_orbit*2,low_elev_leo,rain_leo` | 引入低仰角和雨衰，保留 mixed_orbit 训练密度 |
| robust | 91+ | 0.70-0.85 | `mixed_orbit,low_elev_leo,rain_leo,storm_mp` | 加入 storm/multipath，优化 satellite floor |

可直接用现有 `sat_view_schedule` 表达：

```text
1@0.30:mixed_orbit;
41@0.60:mixed_orbit*2,low_elev_leo,rain_leo;
91@0.80:mixed_orbit,low_elev_leo,rain_leo,storm_mp
```

说明：`geo_clear` 可以作为 sensitivity/control，不进入主报告 five-scenario deployment set，除非后续 `项目.md` 修改推荐视图。

### 6.2 组件二：CE-only 隔离

所有训练主线默认：

```text
clean x -> 完整 CVS/DG 主损失
sat x   -> 单独 forward，只加 TX CE
```

总损失：

```text
L = L_clean_CVS_DG + w_sat_ce * CE(tx_logits_sat, y_tx)
```

不允许主线使用：

```text
x_cat = concat(clean, sat)
d_cat = concat(d_raw, d_raw)
satellite view 进入 domain/GRL/Fishr/group CE
```

full concat 只保留为对照组，用来证明污染风险。

### 6.3 组件三：late weak identity alignment

CE-only 的不足是“不要求 clean/sat 的 `z_id` 接近”。因此 PAIC 增加晚启动、弱权重、stop-gradient 的身份一致性：

```text
z_clean = z_id(clean)
z_sat   = z_id(sat)

L_align = 1 - cosine(z_sat, stopgrad(z_clean))
```

推荐：

```text
sat_cons_start_epoch in {60, 90}
lambda_sat_cons in {0.01, 0.03}
lambda_sat_cls = 0.10 or CE-only branch weight = 1.0
```

原则：

- 必须晚启动，不能在 TX identity 尚未稳定时压一致性。
- 必须小权重，避免把 TX 相关相位/频谱细节也洗掉。
- 只用源域派生 satellite view，不用 `R_t`。
- 成功标准不是单看 satellite avg，而是 clean strict UDU 不明显掉、satellite floor 上升。

若后续小改代码，可把 cosine consistency 升级为 supervised contrastive：

```text
positive:
  same sample clean/sat
  same TX, different source receiver/day, when batch permits

negative:
  different TX

forbidden:
  R_t samples
  Y_new/Y_unknown query
```

### 6.4 组件四：z_dom 吸收 channel nuisance

`z_id` 负责 TX identity，`z_dom` / sidecar 负责 receiver/day/channel/satellite style。PAIC 不追求“模型完全不知道信道”，而是：

```text
z_id:
  不直接泄漏 receiver/day/satellite style
  保留 TX hardware fingerprint

z_dom:
  记录 receiver/day/channel/noise/RCN statistics
  服务 gate、diagnostic、adapter
  不参与 TX prototype 距离
```

推荐保留：

- `domain_freq_stability_mode=dsq`
- `domain_enhancer=rcn_stats`
- `z_id -> receiver/day` leakage probe
- per-scenario satellite floor

### 6.5 组件五：satellite metadata audit

当前 `apply_sat_gnd_channel_batch(return_meta=True)` 可返回：

```text
orbit, h_km, theta_deg, d_km, state, pl_db, fD_hz, cfo_hz, snr_db, K_db
```

建议后续报告记录每个 run 的 meta 分布摘要：

```text
scenario_count
theta p10/p50/p90
snr p10/p50/p90
state ratio: LOS/LOO/Rayleigh
fD_hz p10/p50/p90
cfo_hz p10/p50/p90
storm_mp fraction
```

这能防止只写 scenario name 却不知道实际增强强度。

## 7. 实验矩阵

### 7.1 Centralized 小矩阵

| ID | 名称 | 目的 | 关键配置 | 成功判据 |
|---|---|---|---|---|
| C0 | clean/no-sat baseline | 固定无 satellite train 对照 | 关闭训练时 satellite view，保留 satellite eval | clean strict UDU 与 SAT avg/min 作为参考 |
| C1 | SA16 anchor repeat | 复核现有最稳主线 | `--use_concat_sat_channel_aug --concat_sat_ce_only --concat_sat_ce_weight 1.0 --domain_freq_stability_mode dsq --sat_train_scenarios all5` | 接近 `SA16`：strict UDU 约 82+，SAT avg/min 约 43.66/39.56 |
| C2 | PAIC curriculum CE-only | 验证课程是否优于固定 all5 | C1 + `sat_view_schedule` 三阶段 | clean strict UDU 不低于 C1 0.5 pp；SAT min 或 storm/low_elev floor 上升 |
| C3 | PAIC + late cosine | 验证身份一致性是否补 CE-only 不足 | C2 + `sat_cons_start_epoch={60,90}` + `lambda_sat_cons={0.01,0.03}` | SAT avg/min 上升，clean strict UDU 降幅 <= 1 pp |
| C4 | PAIC-SupCon proposal | 小改代码后验证 supervised contrastive | same-TX clean/sat positives + different TX negatives | 比 C3 更稳，且 leakage probe 不升 |
| C5 | robustness branch | 明确追 satellite floor | ID phase/DSQ + low_elev/rain/storm emphasis | SAT floor 明显高于 C1；若 clean strict 大跌，只作 robustness branch |

### 7.2 Federated 小矩阵

硬约束：

```text
wisig_train_ratio = 0.1
fl_rounds = 200
fl_client_key = receiver
wisig_domain = rx_day
```

| ID | 名称 | 目的 | 关键配置 | 成功判据 |
|---|---|---|---|---|
| F0 | FSDG49 anchor | 稳定历史对照 | FedProx + receiver-client + RA-BEX02 + `cvs_consistency` | strict UDU 对齐 76.295 / final 75.9167 附近 |
| F1 | FL82_16 CE-only DSQ | 复制 SA16 核心因素 | `--fl_sat_aug_mode baseline_view --fl_baseline_view_ce_only --fl_baseline_view_ce_weight 1.0 --domain_freq_stability_mode dsq --sat_train_scenarios all5` | clean strict > F0；SAT avg/min 不低于 SA16 |
| F2 | FL-PAIC curriculum | 联邦 CE-only 加课程 | F1 + 三阶段 `sat_view_schedule` | 不降低 F1 clean strict，提升 SAT floor |
| F3 | FL-PAIC late align | 联邦弱一致性探索 | F2 + late small consistency，仅在诊断确认本地目标不污染时跑 | 若 clean strict 下跌或 Fishr/diag 异常，回退 |
| F4 | StyleBank diagnostic only | 验证 style 是否有额外价值 | F0/F1 + StyleBank-conditioned physical vs random physical | 只有优于 random physical 才可进入下一轮，否则停用 |

必须记录日志字段：

```text
diag_baseline_sat_view_active
diag_sat_cls_active
diag_fishr_domain_count
diag_rx_adv_active
fl_baseline_view_ce_only
fl_baseline_view_ce_weight
sat_train_scenarios
sat_view_schedule
```

### 7.3 Stage2 准备矩阵

Stage2 只做合规设计，不把 Phase1 增强结果写成 Stage2 成功。

| ID | 阶段 | 目的 | 必填字段 |
|---|---|---|---|
| S2-A | zero-label deploy | 旧类识别 + unknown rejection | `R_t`, `Y_old`, `Y_unknown`, empty support, satellite/LEO query |
| S2-B | old-class calibration | target-old K-shot 校准 | `K`, target-old support/query, unknown query eval-only |
| S2-C | old + seen-new enrollment | seen-new 注册 + unknown 拒识 | target-old support/query, target-new support/query, unknown query, `Y_new ∩ Y_old = empty` |

Stage2 报告必须同时给：

```text
clean control
clear_leo
low_elev_leo
rain_leo
storm_mp
mixed_orbit
```

但 deployment-primary 只看 satellite/LEO view。

## 8. 放行门控

### 8.1 Centralized 放行

一个 central 候选要进入下一轮，必须同时满足：

```text
clean strict UDU >= SA16 - 0.5 pp
SAT avg >= SA16 SAT avg + 0.5 pp 或 SAT min >= SA16 SAT min + 0.5 pp
storm_mp / low_elev_leo 不低于 anchor
z_id -> receiver leakage 不升高
训练曲线没有 random-level collapse
```

若只提升 satellite floor 但 clean strict UDU 下跌超过 1 pp，应标为 robustness branch，不替换主线。

### 8.2 Federated 放行

一个 FL 候选要进入下一轮，必须同时满足：

```text
clean strict UDU > FSDG49 best 76.295 或至少 final > 75.9167
SAT avg/min 不低于 SA16 reference 43.66 / 39.56
diag_baseline_sat_view_active = 1
diag_sat_cls_active = 1
fl_baseline_view_ce_only = true
diag_fishr_domain_count >= fishr_min_domains when Fishr is claimed active
```

如果 `fl_baseline_view_ce_only=false`，该 run 只能归类为 legacy full-objective diagnostic。

### 8.3 Stage2 放行

Stage2 row 只有在以下条件均满足时才能写成 launchable：

```text
R_t ∩ R_s = empty
Y_old, Y_new, Y_unknown mutually disjoint
target-old and target-new samples available under R_t when Stage2-C
support/query split verified
unknown query eval-only
target_channel_view contains satellite/LEO
clean view role = control_only
```

否则只能是：

```text
NON_LAUNCH_DIAGNOSTIC
LOCAL_PROTOCOL_REPAIR_REQUIRED
LOCAL_DATASET_EXTENSION_REQUIRED
```

## 9. 需要的代码改动分层

### 9.1 不改代码即可做

- 用现有 `--use_concat_sat_channel_aug --concat_sat_ce_only` 跑 CE-only。
- 用现有 `--sat_view_schedule` 跑 curriculum。
- 用现有 `--use_sat_consistency --lambda_sat_cons` 跑 late weak cosine alignment。
- 用现有 federated `--fl_baseline_view_ce_only` 复刻 SA16 语义。
- 报告 clean strict UDU、SAT avg/min、per-scenario floor。

### 9.2 小改代码

- 把 satellite meta 聚合写入 report。
- 加 TX-aware supervised contrastive head，正样本限定 same TX / clean-sat / source-only。
- 增加 `satellite_view_id` 或 `channel_view_label`，让 diagnostic 知道 satellite style 不是原始 receiver/day。
- 在 validator/report 中新增 Stage2 协议字段。

### 9.3 探索性改动

- 接入 STK/ns-3 NTN 输出作为 scenario 参数先验。
- 引入 residual-channel view 或 light canonicalizer。
- RF-Diffusion/VQ-VAE/GAN 只用于尾部补样，带嵌入距离筛选、置信度降权和每类补样上限。

## 10. 不应采用的路线

| 路线 | 原因 | 处理 |
|---|---|---|
| full concat 作为主线 | satellite view 复制 `d_raw` 进入完整 DG/domain loss，污染语义 | 只作 baseline-style 对照 |
| 早期强 consistency | 模型尚未学稳 TX identity，容易把指纹也当 nuisance 洗掉 | 改成 epoch 60/90 后小权重 |
| 只训 clear_leo | 对 low_elev/rain/storm 泛化不足 | 作为 control，不作主线 |
| all-five epoch 1 全强度 | 稀释场景密度，早期扰动过强 | 改成 curriculum |
| StyleBank 主线 | style 是否等于 receiver 域无法自证 | 只做 diagnostic |
| RF-Diffusion 主线 | 生成样本可能引入 generator artifact | 放到第二阶段尾部补样 |
| 使用 `R_t` 做 Phase1 训练 | 违反 source-only DG | 禁止，或改标 DA/TTA/few-shot |

## 11. 推荐执行顺序

第一轮，只做最小高价值矩阵：

```text
C0 no-sat
C1 SA16 repeat
C2 PAIC curriculum CE-only
C3 PAIC curriculum + late cosine
F0 FSDG49 anchor
F1 FL82_16 CE-only DSQ
```

第二轮，根据第一轮结果扩展：

```text
C4 SupCon
C5 robustness branch
F2 FL curriculum
F4 StyleBank diagnostic
S2-A/B/C satellite-view protocol check
```

第三轮才考虑：

```text
STK/ns-3 parameter prior
residual-channel view
RF-Diffusion / VQ-VAE tail completion
student distillation
```

## 12. 论文与报告表述

推荐写法：

> CVS-SAT-PAIC uses a physics-informed satellite-channel stress view over terrestrial proxy IQ data. The satellite view is isolated from the full domain-generalization objective and contributes transmitter-supervised CE plus a late, weak identity-consistency regularizer. Clean performance is reported as a control, while satellite/LEO scenarios are reported separately as deployment-oriented stress tests.

中文写法：

> CVS-SAT-PAIC 在 WiSig/ManySig 地面代理 IQ 上构造物理启发的 satellite/LEO 压力视图。该视图不进入完整 domain-generalization 损失，而是以 CE-only 的方式提供同一发射机身份监督，并在后期用弱权重约束 clean/satellite `z_id` 一致。clean view 只作为对照，satellite/LEO 分场景结果作为部署压力测试报告，不能等同真实在轨 IQ 验证。

不能写：

- “已经完成真实星地部署验证”。
- “clean strict UDU 成功等价 deployment success”。
- “Stage2-A/B 完成 seen-new identity recognition”。
- “生成式补样等价真实卫星样本”。
- “full concat 是当前最稳主线”。

## 13. 最短执行摘要

当前星地增强的下一步不应是“更强扰动”或“更多生成样本”，而应是“更干净的训练语义”。最优先路线是：保留 `SA16` 的 CE-only 隔离和 domain DSQ，加入三阶段 satellite curriculum，再用晚启动、小权重的 clean-sat `z_id` 对齐补足 CE-only 的不变性不足。中央训练先对齐 `SA16`，联邦训练先对齐 `FSDG49` 与 `FL82_16`，Stage2 只在严格 `R_t/Y_old/Y_new/Y_unknown` 合规后进入 satellite/LEO target view。所有结论必须分开报告 clean control、LEO-only、all-scenario SAT 和 Stage2 open-set 指标。


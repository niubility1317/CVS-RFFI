# CVS-RFFI 地面训练到星上小样本适应审计报告

审计日期：2026-06-13  
审计方式：本地只读代码、配置、报告、日志解析结果和既有实验 artifact；未重新训练；未访问 N607。  
总判定：**部分达到**。

## 1. 执行摘要

1. 当前工作已经实现了地面多源域/DG 训练框架、CVS `z_id/z_dom` 解耦 backbone、卫星信道仿真 stress/eval、部署期 SFE prototype/open-set gate、FTRC adapter/LoRA sidecar 和 rollback gate。
2. 当前工作不能判定为完整达到“地面训练、星上部署、少量星上标签样本同时完成旧类对齐、新类识别、open-set 拒识和轻量化部署”的总体目标。
3. 地面 DG 方面有明确实现和阶段性结果：highratio 当前 best completed `R030_K225_RATIO_STRICT` 的 strict UDU 为 85.17、overall 为 86.19；但已有显式目标 `strict_udu >= 86.00` 或论文草稿 `UDU 86.90`，当前 r010 证据尚未完全达到，且 4 个候选未完成。
4. 星地信道仿真方面已实现 Doppler/CFO/相位/多径/IQ/AWGN 等扰动，并有 38-45% / sat avg 42.1 的论文草稿目标；现有报告可支撑“仿真压力测试”，不能外推为真实在轨 IQ 验证。
5. 星上新类注册方面，Card8 最佳 SFE `COMBINED_K20` 达到 full 50.56%、accepted 79.82%、coverage 63.33%、old 92.50%、new 17.00%，说明旧类保持较强但新类识别弱，生命周期仍为 quarantine。
6. 旧类目标域对齐方面，FTRC feature adapter 和 logit LoRA 均未产生可部署提升：target_tx 从 42.20% 变为 42.11% / 42.16%，rollback 拒绝所有 best checkpoint。
7. open-set/unknown 当前未达成：Card8 `UNKNOWN_TX_IDS=[]`，unknown 指标 n/a；Card9 已审计 unknown TX 但未启动。
8. 轻量化方面有论文草稿参数量证据：训练约 1.05M，部署约 0.63M；但缺少星上端侧 latency、峰值显存、功耗、FLOPs 和 adapter/prototype 存储开销实测。
9. 最大问题是实验闭环不完整：SFE 缺真实 unknown 和显式 satellite support/query 证据，FTRC 适配未通过 rollback，多 seed 和复现锁定不足。
10. 下一步最优先：启动 Card9 true open-set，完成 highratio 36/36 最终解析，并对 `COMBINED_K20` 做多 seed + threshold sweep。

## 2. 研究场景确认

本工作采用地面训练、星上部署的两阶段 CVS-RFFI 架构。地面阶段使用多源域数据进行域泛化训练，以获得跨域鲁棒特征；星上阶段使用少量带标签、叠加星地信道干扰的接收样本进行小样本学习/轻量适应，目标是在消除旧类域偏移的同时，保留或识别新类别，并提升星地信道下的识别性能。

当前代码和文档**部分对应**这个场景。设计文档明确把训练阶段定义为 low-data source-only DG，把部署阶段定义为 few-shot/new-class/target calibration，证据见 `analysis/spaceborne_rffi_dg_fsl_recalibration_20260611.md:489-499`。同一文档也明确：目标样本一旦用于 prototype、adapter、阈值或 early stopping，就不能放入 strict DG 主表，证据见 `analysis/spaceborne_rffi_dg_fsl_recalibration_20260611.md:554-559,594`。

不匹配点：

- 新类注册不是主模型分类器扩容训练，而是 frozen `z_id` feature 上的 prototype/open-set gate。
- 旧类适配不是 `train.py` 默认主路径，而是 `train_target_adapt.py` sidecar。
- SFE Card8 缺少 completed unknown；Card9 只完成 TX audit，未运行。
- SFE support/query 是否显式经过星地信道扰动缺少可验证 evidence；FTRC 目标域适配有 satellite target view 证据。
- 真实卫星链路验证仍在论文讨论中被列为未来工作，见 `paper/discussion_cn.md:15-17`。

## 3. 方法实现审计

| 目标环节 | 应有能力 | 当前实现 | 是否启用 | 证据 | 结论 |
|---|---|---|---|---|---|
| 地面域泛化 | 多源域、domain id、DG loss、未见域评估 | WiSig 支持 `day/rx/rx_day` 域，CVS 有 `z_id/z_dom`，loss 含域对抗/一致性/GroupCE/Fishr/SupCon | 已接入，具体启用依赖 launcher/lambda | `code/dataset_wisig.py:107,165-183`; `code/model_dual_cvsincnet.py:390,596-631`; `code/cvsrffi/losses.py:567` | 部分达到 |
| RF 信道增强 | AWGN、Doppler、CFO、相位、多径、IQ imbalance | `sat_channel.py` 实现物理启发仿真；训练/评估有接入点 | 已接入，真实运行取决于 profile | `code/sat_channel.py:95,123,243-289`; `code/train.py:1956-1970,2138-2161` | 仿真达到，真实在轨未证实 |
| CVS 表征 | embedding、归一化、projection、closed-set 之外可迁移 | `z_id/z_dom` 输出，SFE 使用 frozen `z_id` prototype | SFE/FTRC 已用 | `code/export_spaceborne_features.py:158-179`; `code/cvsrffi/spaceborne_fewshot.py:171,212` | 部分达到 |
| 星上 few-shot support/query | K-shot support，query 独立，old/new split | SFE payload 构造 source/new/unknown TX；FTRC 排除 adaptation samples 后评估 | Card8 SFE/FTRC 已跑；Card9 未跑 | `code/cvsrffi/wisig_fewshot_payload.py:46-58,107-145`; `code/train_target_adapt.py:296-325,628-629` | 部分达到 |
| 旧类域对齐 | 少量目标样本校准旧类 | FTRC adapter/LoRA/calibration + rollback | 已实验 K2，但未通过 rollback | `code/target_domain_adaptation.py:77-143,228-264`; Card8 audit lines 163-186 | 未达到可部署提升 |
| 新类注册/保留 | new prototype、类增量统一分类 | SFE prototype 合并 source/new prototypes，new lifecycle quarantine | 已实验 K5/K10/K20 | `code/cvsrffi/spaceborne_fewshot.py:289,341,422-458`; Card8 report 161-168 | 部分实现，性能未达 |
| Prototype / gate | cosine、Mahalanobis、OpenMax、combined | `OpenSetGateConfig` 和 gate CLI 已实现 | Card8 已跑多个 gate | `code/cvsrffi/spaceborne_fewshot.py:25`; `code/eval_spaceborne_fewshot.py:35-55,88-94` | 实现达到，open-set未验证 |
| Adapter/LoRA | 冻结 backbone 小参数更新 | logit LoRA、feature residual adapter | Card8 K2 已跑，rollback 拒绝 | `code/target_domain_adaptation.py:77-143`; `code/train_target_adapt.py:421-432,613-641` | 实现存在，实验未达 |
| 防遗忘 | old-class guard、rollback、replay/EWC | rollback 有 old-class 和 unknown false accept 规则；未见 EWC/replay 主算法 | 已接入安全门控 | `code/cvsrffi/adaptation_safety.py:40-50,91-127` | 部分达到 |
| open-set / unknown | unknown buffer、AUROC/FPR95、拒识 | CLI 和 metrics 支持 unknown；Card8 unknown 空；Card9 未运行 | 未完成实验 | `code/eval_spaceborne_fewshot.py:53-55`; Card8 report 217; Card9 report 7-10 | 未达到 |
| 星上部署约束 | 小参数、短适应、低存储、可回滚 | 参数量草稿 0.63M；adapter sidecar 和 rollback 有实现 | 缺端侧实测 | `paper/experiments_cn.md:135`; `train_target_adapt.py:668` | 部分达到 |

当前实现最准确归类为：**地面 low-data source-only DG + CVS `z_id/z_dom` 解耦 backbone + 卫星信道仿真 stress/eval + 部署期 frozen-embedding prototype SFE + 轻量 adapter/LoRA FTRC sidecar**。它不是端到端星上持续学习系统，也不是主训练阶段 few-shot learning。

## 4. 实验协议审计

| 实验项 | 当前设置 | 是否符合目标 | 风险 | 证据 |
|---|---|---|---|---|
| source domains | WiSig `rx_day`，train days/rxs 与 test days/rxs 可分离 | 基本符合 | `post_stage_cli.py` 默认 ratio 0.2，与项目硬约束 0.1 有口径风险 | `code/train.py:1092-1101`; `code/post_stage_cli.py:28-36` |
| target/satellite domain | FTRC 使用 `test_unseen_day_unseen_rx` 和 satellite target view | 符合旧类目标域适配 | 合成信道，不是真实在轨 IQ | `code/train_target_adapt.py:57-63,531-535` |
| old classes | SFE source TX 作为 old/base；FTRC 类集合等于源集合 | 部分符合 | SFE source 导出不等价于完整地面 train split 审计 | Card8 report 12-19 |
| new classes | SFE new TX support/query | 部分符合 | new_acc 低，且 support/query 显式 satellite 证据不足 | Card8 report 161-168 |
| unknown classes | Card9 planned `UNKNOWN_TX_IDS=6,7` | 未符合 | Card8 unknown 空，Card9 未启动 | Card8 report 217; Card9 report 7-10 |
| K-shot | SFE K5/K10/K20；FTRC K2；历史 CEN51 K5/K10/K20/K30/K50 | 部分符合 | SFE/FTRC 主结果缺多 seed 和完整 K 曲线 | Card8 report 161-168; comprehensive report 42-59 |
| SNR/channel profile | `sat_channel.py` 有多场景仿真；FTRC eval 有 clear/low/rain/storm/mixed | 部分符合 | SFE payload 没有证明 support/query 都经星地扰动 | `sat_channel.py`; Card8 result_audit 177-186 |
| train/val/test split | FTRC eval 排除 support；SFE split_overlap audit 为 0 | 基本符合 | SFE 证明样本索引不重叠，但未证明 day/rx 域隔离 | `train_target_adapt.py:296-325`; Card8 result_audit 96,156 |
| baseline | SFE 有 source-only baseline；FTRC 有 before baseline | 部分符合 | 缺同协议 source-only、DG-only、prototype-only、adapter sweep 的完整对照 | Card8 result_audit 69-76,163-186 |
| multi-seed | CEN51 comprehensive 有多 seed；Card8 SFE/FTRC 不是同 candidate 多 seed | 不符合论文级稳定性 | 单 seed 不足以宣称稳定达到 | comprehensive report 221-255 |
| ablation | 有 gate 比较和历史 DG ablation | 部分符合 | open-set threshold、satellite SFE、adapter K 曲线缺失 | Card8 report 161-168 |

## 5. 预期性能 vs 实际性能

| 目标 | 指标 | 预期值来源 | 预期值 | 实际值 | 差距 | 是否达标 | 证据 |
|---|---|---|---|---|---|---|---|
| 未见域泛化性能 | strict UDU | `docs/cvs_rffi_staged_experiment_groups.md:29`; `paper/experiments_cn.md:51-52` | `>=86.00`；草稿声称 86.90 | highratio best completed 85.17 | -0.83 pp vs 86.00 | 部分达成 | status report 35-45 |
| 地面 overall | overall | `paper/experiments_cn.md:51-52` | 91.01 | highratio best completed 86.19 | -4.82 pp vs 草稿 | 未达当前草稿口径 | stage analysis 17,27 |
| 星地信道 no-adapt | sat avg / sat floor | `paper/abstract_cn.md:3`; `paper/experiments_cn.md:127` | 38-45；sat avg 42.1 | 历史/当前报告有 41.78-47.17 等仿真指标 | 仿真达成 | 部分达成 | central status and paper evidence |
| 星上 few-shot 后性能 | SFE full acc | 项目未定义显式目标；需补充 | 建议 `>=60%` | best K20 50.56% | -9.44 pp vs 建议 | 未达建议 | Card8 report 214 |
| 旧类准确率 | SFE old_acc | 项目未定义显式目标；需补充 | 建议 `>=90%` | best K20 92.50% | +2.50 pp | 达到该单项 | Card8 report 214 |
| 新类准确率 | SFE new_acc | 项目未定义显式目标；需补充 | 建议 `>=50%` | best K20 17.00% | -33.00 pp | 未达 | Card8 report 214-215 |
| H-mean | old/new harmonic mean | 项目未定义显式目标；需补充 | 建议 `>=64%` | best K20 28.72% | -35.28 pp | 未达 | `metrics_summary.csv` 派生 |
| forgetting | old before/after | 项目未定义显式目标；需补充 | 建议 old drop <=2 pp | SFE 无完整 before/history；FTRC rollback拒绝 | 无法判断 | 无法判断 | Card8 audit |
| domain gap recovery | source clean vs target before/after | 项目未定义显式目标；需补充 | 建议 positive 且有意义 | SFE 缺 source clean；FTRC 42.20->42.11/42.16 | FTRC 负增益 | 未达/无法完整计算 | result_audit 163-186 |
| open-set AUROC/FPR95 | AUROC/FPR95 | 项目未定义显式目标；需补充 | 建议 AUROC >=0.90、FPR95 <=5% | 未找到完成结果 | 缺失 | 未达/无法判断 | Card8 report 217; Card9 report 7-10 |
| 适应开销 | params/latency/memory | `paper/experiments_cn.md:135` | 部署约 0.63M 参数 | 参数量有草稿，latency/memory 缺失 | 工程证据不足 | 部分达成 | paper experiments 135 |

建议验收门槛不是项目已有目标。建议先采用：FTRC target 提升 >=1pp、overall drop <=0.5pp；SFE full >=60%、old >=90%、new >=50%、coverage >=70%；open-set unknown rejection >=95%、false accept <=5%；部署报告 P95 latency、峰值显存、adapter/prototype state size。

## 6. 是否达到目标的逐项判断

### 6.1 地面训练是否实现了域泛化？

- 结论：**部分达到**。
- 使用方法：CVS 双骨干 `z_id/z_dom`、domain id、域对抗/一致性/GroupCE/Fishr/SupCon、卫星仿真 stress。
- 预期性能：显式目标包含 `strict_udu >=86.00`，论文草稿声称 UDU 86.90。
- 实际性能：当前 highratio best completed strict UDU 85.17，overall 86.19；4 个候选仍未完成。
- 证据：`code/train.py:1092-1101`; `code/cvsrffi/losses.py:567`; `automation_reports/CV-SincNet/status_fewshot_deploy_20260613_2033/report.md:35-45`。
- 是否达标：当前 r010 口径未完全达成 86+ 目标。
- 问题：协议口径混用、late drop、未完成候选。

### 6.2 星上部署是否实现了小样本学习？

- 结论：**部分达到**。
- 使用方法：SFE frozen `z_id` prototype + gate；FTRC frozen backbone adapter/LoRA + rollback。
- K-shot 设置：SFE K5/K10/K20；FTRC K2；历史 CEN51 anchor K5/K10/K20/K30/K50。
- 预期性能：项目未定义显式 SFE/FTRC 性能目标。
- 实际性能：SFE best K20 full 50.56、accepted 79.82、coverage 63.33；FTRC K2 42.20->42.11/42.16。
- 证据：Card8 report 161-168, 214-217；result_audit 163-186。
- 是否达标：SFE 流程实现但新类弱；FTRC 未产生可部署提升。
- 问题：缺多 seed、缺完整 K 曲线、缺 open-set实跑。

### 6.3 少量星上标签样本是否叠加了星地信道干扰？

- 结论：**部分达到/无法完全判断**。
- 信道模型：`sat_channel.py` 支持 Doppler、CFO、phase noise、multipath、IQ imbalance、AWGN、path loss/weather。
- 扰动类型：FTRC 有 target satellite view 和 per-scenario eval；SFE Card8 没有找到 support/query 显式 satellite payload 证据。
- 证据：`code/sat_channel.py:95,123,243-289`; `code/train_target_adapt.py:57-63,333-354`; Card8 result_audit 177-186。
- 是否足够真实：只是物理启发仿真，不是真实在轨 IQ。
- 问题：SFE new-TX 实验可能只是 WiSig feature-level few-shot，需补 satellite SFE payload。

### 6.4 旧类别是否被对齐，域偏移是否被消除或减小？

- 结论：**未达到可部署旧类对齐**。
- old_acc before adaptation：FTRC target no-adapt 42.20%。
- old_acc after adaptation：feature adapter 42.11%，logit LoRA 42.16%。
- domain gap：缺 source clean 同表，无法按定义完整计算。
- gap recovery：FTRC 为负增益，不应声称恢复。
- 证据：`result_audit_20260613.md:163-186`; `train_target_adapt.py:613-641`。
- 是否达标：未达。
- 问题：adapter/LoRA 被 rollback 正确拦截，但没有带来目标域收益。

### 6.5 新类别是否被保留或识别？

- 结论：**未达到研究目标**。
- new_acc：best K20 为 17.00%。
- new-to-old confusion：未找到完整矩阵；从 new_acc 极低推断存在强混淆风险，但需补混淆矩阵验证。
- H-mean：审计派生约 28.72%。
- open-set 指标：未完成；unknown n/a。
- 证据：Card8 report 161-168,214-217；`metrics_summary.csv`。
- 是否达标：未达。
- 问题：prototype/gate 对新 TX 分离不足，且 lifecycle 为 quarantine。

### 6.6 星地信道下总体性能是否提升？

- 结论：**仿真旧类压力测试部分达成；部署期适配未达成**。
- no-adapt baseline：SFE K20 baseline_full 41.11；FTRC 42.20。
- after-adapt result：SFE K20 full 50.56；FTRC 42.11/42.16。
- 提升幅度：SFE +9.45 pp，但这是 new-class prototype enrollment，不是严格 old-class domain gap recovery；FTRC -0.09/-0.04 pp。
- 多 seed 稳定性：SFE/FTRC 主结果不足。
- 证据：Card8 result_audit 69-76,163-186。
- 是否达标：SFE 有阶段性提升信号，FTRC 未达。
- 问题：缺 source clean、缺多 seed、缺 open-set。

## 7. 未达标原因分析

| 未达标目标 | 可能原因 | 证据 | 置信度 | 验证实验 | 修复方向 |
|---|---|---|---|---|---|
| open-set unknown | Card8 unknown 为空，Card9 未启动 | Card8 report 217; Card9 report 7-10 | 高 | Card9 `UNKNOWN_TX_IDS=6,7` | 先补 true open-set |
| 新类识别 | frozen prototype 对新 TX 分离不足，gate 更保护旧类 | new_acc 4-17%，old 可到 92.5 | 高 | K20 多 seed + threshold sweep + confusion matrix | prototype calibration、balanced margin |
| 旧类对齐 | adapter/LoRA 更新太弱或损失目标不匹配，rollback 拦截退化 | 42.20->42.11/42.16，best ckpt missing | 高 | FTRC K1/5/10/20 + lr/alpha sweep | 更弱更新、distillation、old validation buffer |
| 星地 support/query | SFE manifest 未证明 support/query 经过 satellite channel | SFE 是 feature payload；FTRC 才有 target view | 中高 | clean vs satellite SFE payload A/B | feature export 显式 sat profile |
| DG 目标未完全达标 | 当前 r010 best 低于 86+，且 late drop | status report 35-45 | 高 | highratio 36/36 final parse | checkpoint selection、late-drop guard |
| 论文级可信度 | 主结果多为单 seed，环境/commit/hash 缺失 | no git repo，env file缺失，多 seed不足 | 高 | 3-5 seed + env/checkpoint hash | reproducibility bundle |
| 部署轻量化 | 只有参数量草稿，缺端侧实测 | paper experiments 135 | 中 | latency/memory/FLOPs benchmark | edge benchmark script |

## 8. 改进方案

| 优先级 | 改进项 | 解决问题 | 修改模块 | 预期影响指标 | 验证实验 | 风险 |
|---|---|---|---|---|---|---|
| P0 | 启动 Card9 true open-set | unknown 指标缺失 | launcher, `eval_spaceborne_fewshot.py` | unknown rejection/FPR95/AUROC | K20 + unknown 6,7 | new_acc 可能更低 |
| P0 | SFE payload 记录 satellite profile | support/query 信道证据不足 | `export_spaceborne_features.py`, payload manifest | channel 可审计性 | clean vs satellite SFE | 特征导出成本 |
| P0 | highratio 完成后最终解析 | 地面 best 未最终确定 | parser/report | strict/primary/late drop | 36/36 parse | 不能打断训练 |
| P0 | 标准化指标表 | overall 掩盖 trade-off | evaluator/parser | old/new/H-mean/forgetting | Card8/Card9 refresh | 派生指标需标注 |
| P1 | K20 多 seed | 稳定性不足 | SFE matrix | mean/std/CI | 3-5 seed | 成本增加 |
| P1 | prototype calibration | new_acc 低 | `spaceborne_fewshot.py` | new_acc/H-mean | calibration ablation | old_acc 下降 |
| P1 | support quality filtering | support outlier 污染 | payload/evaluator | coverage/new_acc | score/SNR filter | 有效 shots 变少 |
| P1 | FTRC K sweep | K2 不能代表全局 | `train_target_adapt.py` matrix | target gain/rollback pass | K1/5/10/20 | 仍可能无提升 |
| P2 | adapter + distillation/EWC | 旧类遗忘与适配冲突 | target adaptation | target_acc/forgetting | distillation ablation | 算力上升 |
| P2 | unknown buffer clustering | open-set 生命周期不完整 | SFE lifecycle | unknown precision/new enrollment | provisional class test | 错误聚类 |
| P2 | channel-aware adapter | 多卫星场景差异 | adapter/router | worst scenario | descriptor routing | 路由误差 |
| P3 | 完整 open-world FSL | 当前不是持续学习系统 | deployment registry | sequential new_acc/forgetting | class-incremental benchmark | 周期长 |
| P3 | 硬件闭环 | 部署可行性证据不足 | benchmark/export | latency/memory/power | INT8/FP16/edge bench | 精度损失 |

## 9. 复现性与可信度

- 复现性评分：**3/5**。
- 配置完整性：matrix、launcher、report、dry-run 命令较完整。
- 结果完整性：Card8 完整；Card9 缺运行结果；highratio 有 4 个未完成候选。
- 多 seed：CEN51 comprehensive 有多 seed；Card8 SFE/FTRC 主结论缺同 candidate 多 seed。
- checkpoint：报告记录 remote checkpoint；缺本地 checkpoint 归档和 SHA256。
- 日志：Card8 与 highratio 解析结果存在；open-set 缺日志。
- 环境：未找到 `environment.yml`、`requirements.txt`、`Dockerfile`、`pyproject.toml`、`setup.py` 等标准环境锁定文件。
- git：顶层不是 git repository，`git status` 返回 fatal；无法记录 commit hash。
- 一键复现实验命令：Card9 launcher 有 dry-run 和 N607 命令，见 `missing_experiments.md`。
- 当前结论可信度：阶段性审计可信；论文主张证据不足，需补 Card9、多 seed、环境/权重 hash、部署 benchmark。

## 10. 最终结论

基于当前代码和实验结果，本项目已经实现了地面训练、星上部署两阶段 CVS-RFFI 架构的主要工程框架：地面侧有 CVS 解耦表征、多源域/DG loss 与卫星信道仿真 stress；星上侧有 frozen embedding prototype SFE、OpenMax/Mahalanobis/combined gate、new-class lifecycle、rollback gate，以及 FTRC adapter/LoRA sidecar。  

但从研究目标达成看，当前只能判定为**部分达到**：地面 DG 有阶段性结果但当前 r010 best 尚未达到 86+ 显式目标；FTRC 旧类目标域适配没有可部署提升；SFE 新类注册流程可运行但 best new_acc 只有 17.00%，H-mean 约 28.72%；open-set unknown 还没有完成实验；SFE support/query 的星地信道扰动证据不足；轻量化只有参数量草稿，缺端侧开销实测。最优先改进方向是先跑 Card9 true open-set、补 SFE satellite payload 证据、完成 highratio 最终解析，再做 K20 多 seed 和 threshold sweep。

## 11. 目标达成度判定表

| 目标 | 是否达成 | 方法证据 | 预期性能 | 实际性能 | 差距 | 主要原因 | 下一步 |
|---|---|---|---|---|---|---|---|
| 地面多源域训练是否实现域泛化 | 部分达成 | `dataset_wisig.py`, `model_dual_cvsincnet.py`, `losses.py`; highratio report | 显式 `strict_udu >=86.00`；草稿 UDU 86.90 | highratio best completed strict 85.17, overall 86.19 | -0.83 pp vs 86.00 | 4 个候选未完成，late drop，协议口径分裂 | 36/36 最终解析，固定 best checkpoint |
| 星上少量标签样本小样本学习是否实现 | 部分达成 | SFE prototype/gate；FTRC adapter/LoRA | 项目未定义；建议 SFE full >=60 | SFE K20 full 50.56；FTRC 42.20->42.11/42.16 | SFE 低于建议；FTRC 负增益 | 新类 prototype 弱；adapter rollback 拒绝 | K20 多 seed；FTRC K sweep |
| 少量标签样本是否包含星地信道干扰 | 部分达成/无法完全判断 | `sat_channel.py`; FTRC target view | 项目未定义；应 support/query 都有 profile | FTRC 有 satellite eval；SFE 缺显式证据 | SFE 证据缺失 | feature-level payload 未记录 sat profile | clean vs satellite SFE payload |
| 旧类别是否完成域对齐 | 未达成 | `train_target_adapt.py`; Card8 FTRC audit | 项目未定义；建议 target +1pp | 42.20->42.11/42.16 | -0.09/-0.04 pp | adapter/LoRA 无可部署提升 | 降 lr/steps/alpha，distillation，K sweep |
| 新类别是否被保留或识别 | 未达成 | `spaceborne_fewshot.py`; Card8 SFE report | 项目未定义；建议 new >=50 | best new_acc 17.00, H-mean 28.72 | -33 pp vs 建议 new | prototype 分离不足，lifecycle quarantine | calibration、balanced loss、support filtering |
| 星地信道下总体性能是否提升 | 部分达成 | `sat_channel.py`; Card8 SFE/FTRC | 草稿仿真 38-45；FTRC 建议 +1pp | SFE full +9.45 pp vs baseline；FTRC 负增益 | 旧类适配未提升 | SFE 不是完整 domain recovery；FTRC 退化 | 分开汇报 SFE/FTRC，补 source clean |
| 是否避免旧类遗忘 | 部分达成 | rollback rules; SFE old_acc | 建议 old drop <=2 pp | SFE old 92.50；FTRC rollback 拒绝退化 | forgetting 无法完整计算 | 安全门控保守，缺 replay/EWC | old validation buffer、prototype replay |
| 是否满足星上部署轻量化要求 | 部分达成 | paper 参数量；adapter sidecar | 草稿部署约 0.63M；需 latency/memory | 参数量有草稿，端侧开销缺失 | 缺硬件证据 | 无 P95 latency/FLOPs/功耗 | benchmark + checkpoint hash + env lock |

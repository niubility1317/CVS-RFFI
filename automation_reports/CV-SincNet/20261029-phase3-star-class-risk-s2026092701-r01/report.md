# 预测类别条件风险阈值：预登记

当前登记状态：ANALYZED_SOURCE_ONLY；两CPU stage COMPLETE，实测NO_PROMOTION。以下预登记正文保留，结果见末节。

run `20261029-phase3-star-class-risk-s2026092701-r01`，发布前登记，不是实测成功；完整Phase3目标ACTIVE，适用最新v3双90%/原F0同P1闭集到开放损失2pp、宏A_K/R_U/H。

实际code `bb507e2b6f4d66bd808a485d70d64d25c3556da3`，24项相关检查/45subtests PASS，唯一独立下一实跑P0/P1审查PASS，本地runtime validate和一次N607 preflight PASS。实际parent readout checkpoint无query feature smoke是第一步，PASS继续；无新IQ/F0/G smoke要求，不冒称真实新星地样本测试。

方法、Fraction公共尾率q、整ties/42组风险、条件阈值、唯一主与科学停止见[最小设计](../../../docs/PAMF_PHASE3_STAR_CLASS_RISK_PLAN_20261029.md)。固定parent paired1000 rawscore及P1身份，knownV27000只校准一global对照和六predclass阈值，模型predclass路由而非真实TX；全部阈值冻结后才开novel3150，所有两列margin预测complete后独立truth-last。无神经/geometry/normalizer训练、无新L/OE/raw/clean/IQ/F/G/target。父训练成本/数据祖先完整保留。

源known42组C_K≥90%、原F0closed−A_K≤2pp，unknownoverall/5RX/6scene≥90%；primary仅predicted_class，global不得救主，缺必需类N/A，不设H硬门槛。margin=s−τ_pred，在0接收，真TX仍只scorer。source同RX历史novel验证不等于跨RX目标、DA或未来注册，A_N/H_OW=N/A，无conformal保证。

命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_class_risk_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase3-star-class-risk-s2026092701-r01-release/runtime.json`。
环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase3-star-class-risk-s2026092701-r01-release`。
run/output：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase3-star-class-risk-s2026092701-r01`；logs同root的`logs/`；local：`E:/type10-7/local_artifacts/pamf_phase3_star_class_risk_20261029_r01`。
parent预测/实际readout checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase3-star-paired-evidence-s2026092701-r01/prediction`；私有novel truth只独立scorer读取。
CPU-only，CUDA_VISIBLE_DEVICES为空、新增GPUworker0，保留所有其他owner。单immutable root和唯一owner，技术输入/非有限/输出碰撞/无法合法预测评分则保留partial，不盲重发；科学低分仍全评分，不调未知阈值或换主。

预期artifact见experiment.json：root prediction/complete与smoke、thresholds.npz(global1/conditional6)、calibration/complete、predict/known和novel.npz、独立score/results与complete、launch/pipeline/logs及独立PID/CWD/argv/终态观察。只Git交付代码/配置/报告/登记，IQ、权重、原始预测和日志不提交。实际checkpoint smoke首步、单archive一次local/remote SHA、一次compile和启动读回按最小流程完成；REJECTED_EXTRA_GATE：无额外设计审批、data重验、hash/receipt、重复审查或报告完整性发布门槛。

## 2026-10-29 预测类别条件风险实测结果

`20261029-phase3-star-class-risk-s2026092701-r01` 两CPU stage COMPLETE；唯一主predicted_class **NO_PROMOTION**，完整Phase3目标仍 **ACTIVE**。

| 方法 | A_K% | R_U% | H-score% | C_K% | F0损pp | sourcegate |
| --- | --- | --- | --- | --- | --- | --- |
| global_control | 90.3593 | 43.7143 | 58.9227 | 98.4778 | 0.6222 | False |
| predicted_class | 90.0593 | 59.6508 | 71.7668 | 97.8407 | 0.9222 | False |

known42保护通过，按actual group统计未知必需12/12失败；global精确复现冻结parent paired1000。主−global ΔAK−0.3000pp/ΔRU+15.9365pp/ΔH+12.8441pp，净正确−81/净接收−172/未知净多拒502。
公共q=97/4473，26984候选、选中索引583，六预测桶拒前缀[92,91,97,107,97,99]；下一q=92/4241因RX6/lowurban原F0损2.0316pp超2失败。只按模型预测c选六τ，真TX宏指标独立评分，非conformal保证。
曲线用class-specific margin重算，主AUROC96.0749/OSCR88.5425%，不沿用父rawscore排序；较高曲线不代表0运行点RU90。无新L/OE IQ、F/G/geometry/normalizer/读出更新、target/U/注册或新GPUworker，祖先E50/11100和父2000读出更新成本保留。
部署共享F+G+MLP503177参数＋6个FP64τ（48B），global对照另1τ；两row实际parent读出feature smoke PASS，不是新F0/G/IQsmoke。独立live driver2955649/source2955662 CWD/argv/CUDA空，CPUscore2955731完成，log485→607B，终态live空/foreign0；pipeline3.598202s不是逐包时延。
完整42组/真TX/预测桶预算与配对证据见[`PAMF_PHASE3_STAR_CLASS_RISK_RESULTS_20261029.md`](../../../docs/PAMF_PHASE3_STAR_CLASS_RISK_RESULTS_20261029.md)。同源RX与历史novel不是独立target确认，AN/HOW=N/A；不调阈值、不换主、不新增实验。

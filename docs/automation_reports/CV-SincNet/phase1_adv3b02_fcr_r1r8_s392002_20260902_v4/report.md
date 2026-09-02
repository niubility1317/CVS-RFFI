# ADV3B02-FCR R1-R8八卡并行实验v4报告

## 状态

- run_id：`phase1_adv3b02_fcr_r1r8_s392002_20260902_v4`
- 当前状态：`ANALYZED / SCIENTIFIC_NEGATIVE_NO_PROMOTION`
- protocol_scope：Phase1 source-only；不访问Phase2 support/query/truth
- implementation_commit：`9e2b75a4ed70a4d206cbd87a4c9750773d8299c8`
- branch：`codex/adv3b02-fcr-20260901`
- predecessor：v3因CUDA AMP将FCR复数链路降为`ComplexHalf`，在`gather`处确定性失败；v1-v3产物全部保留，本v4使用新的不可覆盖run root。

## 冻结矩阵与GPU映射

| Row | 语义 | GPU | Decoder |
|---|---|---:|---|
| R1 | FCR身份CE+`L_self+L_eta` | 0 | `control` |
| R2 | R1+`L_swap` | 1 | `control` |
| R3 | R2+`L_shared` | 2 | `control` |
| R4 | R3+`L_latent_cycle` | 3 | `control` |
| R5 | R4+同样本basic`L_drop_f`必要性 | 4 | `control` |
| R6 | R5+严格Fingerprint Pair定向移植 | 5 | `control` |
| R7 | R6+完整物理顺序Decoder和Fisher门控`L_phys` | 6 | `full_physics` |
| R8 | R7+严格三轴干预 | 7 | `full_physics` |

R0及旧ADV3B02基线均不启动。所有row固定`seed=392002`、`epochs=200`、`model_variant=lite_d`、Meta-SSL`L_s/U_s/V=0.07/0.63/0.30`、E80卫星辅助CE、三段LEO_WEAK日程和同一source split。严格Fingerprint Pair仍为`blocked`，缺失轴连接零并报告`N/A`。

## 本次修复与本地验证

- 在FCR聚合前向、`ContentGenerator`、Fingerprint响应算子和物理顺序Decoder建立模块级FP32/`complex64`精度边界；未关闭ADV3B02主干AMP，未改变因子、Decoder顺序、损失、数据或矩阵。
- CPU autocast RED复现后修复；本地真实CUDA FP16 AMP下`U_s`和`L_s`完整objective+backward通过，主干输出保持FP16、FCR目标为FP32、33组FCR梯度有限。
- 全量FCR聚焦组90项通过；Python语法和差异检查通过。
- 原P1定点复审：PASS，0项P0/P1。

## 用户资源授权

用户明确授权本次05:00启动忽略既有显卡进程数量限制，每张GPU新增一个本实验row。该授权不允许停止、重启、修改或影响任何既有进程。

## 环境、路径与命令

- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- release root：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_fcr_r1r8_s392002_20260902_v4`
- 输入：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fcr_r1r8_s392002_20260902_v4`
- row输出：`<run root>/jobs/Rk/Rk`
- launcher日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_fcr_r1r8_s392002_20260902_v4.Rk.launcher.out`
- 启动入口：`docs/automation_reports/CV-SincNet/phase1_adv3b02_fcr_r1r8_s392002_20260902_v4/launch_r1r8_remote.sh`
- 启动命令：`bash <release root>/docs/automation_reports/CV-SincNet/phase1_adv3b02_fcr_r1r8_s392002_20260902_v4/launch_r1r8_remote.sh`

## Release归档映射

- archive_source_commit：`d55daeb206f358e9d20d7c19561ac6329383eccb`
- 本地归档：`E:\type10-7\release_archives\phase1_adv3b02_fcr_r1r8_s392002_20260902_v4_d55daeb2.tar.gz`
- 远端归档：`/home/szu2070436088/2510044040/CV-SincNet/releases/archives/phase1_adv3b02_fcr_r1r8_s392002_20260902_v4_d55daeb2.tar.gz`
- SHA256：`45828f1c017cb10d71229f2ac16705a75381ae0ef67e76a1e8734410713df53e`

## 直接技术停止规则

只在数据/query权限越界、错误row/seed/split、输出覆盖、错误checkout、命令不能运行、无prediction闭合、进程归属不明，或至少两个row出现同一确定性pre-prediction异常时，停止本run拥有的精确进程树并保留全部产物。低性能、负收益、严格Fingerprint Pair不可用或既有GPU任务数量不得作为停止理由。

## 预期artifact

每个R1-R8独立保存`best_joint.pth`、`fcr_diagnostics.json`、`fcr_predictions.json`、`train.log`和`status.txt`。完成训练的row必须产生clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`四场景prediction。启动闭合只证明`RUNNING`；独立truth-last评分后才能进入`ARTIFACTS_COMPLETE/ANALYZED`。

## N607发布与启动闭合

- 2026-09-02 10:32 CST完成直接SSH只读preflight；普通账户、项目路径、8张GPU和7.2TB可用磁盘均可见。v4 release、run和archive目标在发布前均不存在。
- release归档本地/远端SHA256一致；远端Python编译和`bash -n`通过。N607 Torch2.1/CUDA12.1下直接运行完整CUDA FP16 AMP`compute_fcr_pair_objective+backward`，结果为`N607_CUDA_AMP_PAIR_OBJECTIVE_PASS`。
- 远端训练环境未安装`pytest`，因此未以pytest runner执行；同一回归函数通过标准库直接加载并执行成功，不安装服务器软件。
- 2026-09-02 10:35 CST启动R1-R8；未启动R0或旧ADV3B02基线。launcher PID：R1=`4166651`、R2=`4166652`、R3=`4166653`、R4=`4166654`、R5=`4166655`、R6=`4166656`、R7=`4166657`、R8=`4166658`。
- 主训练PID：R1=`4166678`、R2=`4166679`、R3=`4166682`、R4=`4166680`、R5=`4166674`、R6=`4166684`、R7=`4166681`、R8=`4166683`。CWD全部绑定v4 release，GPU映射逐一为R1→GPU0至R8→GPU7。
- 10:36 CST，8个`train.log`均增长至约16.5KB并写出E-01 checkpoint事件；`ComplexHalf|Traceback|RuntimeError`扫描为零，原v3确定性故障未复现。

## 完成状态与证据闭合

- 2026-09-02 15:05 CST，R1-R8全部完成E200，8个`status.txt`均为`PREDICTIONS_READY`，对应launcher和训练进程均已退出。
- 每个row均保存`best_joint.pth`、完整`train.log`、200轮`logs.jsonl`、200轮`metrics.csv`、`fcr_diagnostics.json`和`fcr_predictions.json`。
- 每个prediction文件包含100,800条记录，即25,200个样本×4个场景；8个row合计806,400条逐样本预测。
- 四个场景均已闭合：`clean`、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`。每个场景25,200个样本、6个TX类别、每类4,200个样本。
- 本地独立分析进程未导入训练代码，完整解析8份prediction、8份diagnostics、1,600个epoch级CSV记录、1,600个JSONL记录和全部stdout日志，并重新计算混淆矩阵、每类准确率、receiver floor和资源统计。
- 限制：prediction中的样本ID可逆地包含`tx0`至`tx5`，所以本次属于“独立进程评分”，不是严格不透明ID的truth-last封存。数值结果可复核，但不能声称已证明严格隐藏真值的评分隔离。

## 一句话结论

实验已跑完，工程产物完整，v3的CUDA`ComplexHalf`故障已修复；但R1-R8全部出现严重类别塌缩，R2-R8还出现大面积FCR非有限值和安全跳步。最佳LEO三场景均值仅为R3的21.3003%，且该row有143个epoch缺失有效训练loss、6,515次反向跳过、clean类准确率下限为0%。因此本矩阵结论为`NO_PROMOTION`，不能替换或升级ADV3B02。

## 同row最终结果

所有数值均由同一个row的最终选定checkpoint生成；不把不同row的单项最优拼成虚构“最佳模型”。

| Row | Best val/Epoch | clean | clear | low-elev | rain | LEO均值 | LEO最差 | clean类下限 | 缺失loss epoch | 反向跳过 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 | 28.9563/E1 | 28.9563 | 21.6270 | 20.8810 | 21.0159 | 21.1746 | 20.8810 | 0.0000 | 0 | 96 |
| R2 | 28.7738/E1 | 28.7738 | 21.5913 | 20.9444 | 21.1627 | 21.2328 | 20.9444 | 0.0000 | 126（E75起） | 5,834 |
| R3 | 28.4405/E1 | 28.4405 | 21.6865 | 21.0317 | 21.1825 | **21.3003** | 21.0317 | 0.0000 | 143（E58起） | 6,515 |
| R4 | 28.9286/E1 | 28.9286 | 21.5992 | 20.9444 | 21.0516 | 21.1984 | 20.9444 | 0.0000 | 132（E69起） | 6,117 |
| R5 | 20.4921/E21 | 20.4921 | 18.7659 | 18.4802 | 18.1746 | 18.4735 | 18.1746 | 0.0000 | 173（E27起） | 7,901 |
| R6 | 16.8095/E1 | 16.8095 | 18.3135 | 17.9087 | 17.9881 | 18.0701 | 17.9087 | 0.0000 | 174（E27起） | 7,979 |
| R7 | 16.6984/E1 | 16.6984 | 17.6230 | 17.3413 | 17.4206 | 17.4616 | 17.3413 | 0.0000 | 158（E43起） | 7,231 |
| R8 | 16.9484/E1 | 16.9484 | 18.5833 | 18.1587 | 18.1270 | 18.2897 | 18.1270 | 0.0000 | 172（E29起） | 7,921 |

说明：`Best val`与`clean`相同是当前launcher按clean validation选取`best_joint.pth`后再评测所得，不代表训练末轮性能。R3的LEO均值比R1高0.1257个百分点，但R3的数值失稳明显更严重，差异不构成可晋级收益。

## 每类结果与类别塌缩

### clean每类准确率

| Row | TX0 | TX1 | TX2 | TX3 | TX4 | TX5 |
|---|---:|---:|---:|---:|---:|---:|
| R1 | 3.8810 | 0.0000 | 0.0000 | 0.0238 | 98.9286 | 70.9048 |
| R2 | 2.1667 | 0.0000 | 0.0000 | 0.0238 | 99.0238 | 71.4286 |
| R3 | 3.5238 | 0.0000 | 0.0000 | 0.0238 | 98.9286 | 68.1667 |
| R4 | 2.7143 | 0.0000 | 0.0000 | 0.0238 | 98.8810 | 71.9524 |
| R5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 81.5000 | 41.4524 |
| R6 | 0.0238 | 0.0000 | 0.0000 | 0.0000 | 99.7143 | 1.1190 |
| R7 | 0.0238 | 0.0000 | 0.0000 | 0.0000 | 99.7619 | 0.4048 |
| R8 | 0.0238 | 0.0000 | 0.0000 | 0.0000 | 99.5952 | 2.0714 |

### `leo_rain_weak`每类准确率

| Row | TX0 | TX1 | TX2 | TX3 | TX4 | TX5 |
|---|---:|---:|---:|---:|---:|---:|
| R1 | 6.0476 | 0.0000 | 0.0000 | 0.0238 | 43.1190 | 76.9048 |
| R2 | 5.0000 | 0.0000 | 0.0000 | 0.0238 | 45.6905 | 76.2619 |
| R3 | 6.4286 | 0.0000 | 0.0000 | 0.0238 | 45.1429 | 75.5000 |
| R4 | 5.5000 | 0.0000 | 0.0000 | 0.0238 | 43.8095 | 76.9762 |
| R5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 24.7619 | 84.2857 |
| R6 | 0.0476 | 0.0000 | 0.0000 | 0.0238 | 91.7857 | 16.0714 |
| R7 | 0.0714 | 0.0000 | 0.0000 | 0.0238 | 95.4524 | 8.9762 |
| R8 | 0.0476 | 0.0000 | 0.0000 | 0.0238 | 85.4048 | 23.2857 |

所有row在TX1、TX2上均为0%，TX3几乎为0%；预测高度集中于TX4/TX5。以clean为例，R6-R8分别有25,076、25,150、25,001个样本被预测为TX4，占25,200个样本的99%以上。这不是轻微精度下降，而是身份判别空间发生结构性塌缩。

## receiver floor

receiver floor是同一row、同一场景下各receiver准确率的最小值，用于排除总体均值掩盖某个receiver完全失效的情况。

| Row | clean | clear | low-elev | rain |
|---|---:|---:|---:|---:|
| R1 | 19.4167 | 15.8056 | 16.0833 | 16.4167 |
| R2 | 18.1389 | 15.8056 | 15.9167 | 16.5556 |
| R3 | 19.2500 | 15.6111 | 15.8333 | 16.4167 |
| R4 | 18.3889 | 15.8611 | 16.0278 | 16.5278 |
| R5 | 13.3889 | 16.3889 | 15.3889 | 16.1111 |
| R6 | 16.4167 | 16.2778 | 16.3333 | 16.8889 |
| R7 | 16.4722 | 16.3889 | 16.2778 | 16.3889 |
| R8 | 16.3056 | 16.6111 | 16.1667 | 16.6111 |

receiver floor相对接近随机六分类的16.67%，但类别floor全部为0%，说明主要失败轴是TX身份类别，不是单个receiver完全失效。

## 数值稳定性与训练曲线

- 8个row都完整记录四阶段轮数：E1-E40共40轮、E41-E90共50轮、E91-E150共60轮、E151-E200共50轮。
- R1没有epoch级FCR NaN，但有96个batch因loss或梯度非有限而被`safe_backward_step`拒绝。
- R2-R8分别从E75、E58、E69、E27、E27、E43、E29开始出现持续epoch级FCR非有限值；合计1,078个epoch缺失有效训练loss，49,498次batch跳过。连同R1，本矩阵累计49,594次跳过。
- 代码在`sanitize_loss`处把非有限标量loss替换为可反传零项，并在`safe_backward_step`处拒绝非有限总loss或非有限梯度，因此作业没有崩溃，仍完成E200和预测。但这只保证“不中断”，不能把这些epoch视为有效优化。
- 日志能够证明FCR非有限值与大规模跳步同期出现，但不能仅凭聚合日志断言每一次非有限梯度都只由FCR造成；R1的FCR epoch指标有限但仍有96次跳步，就是该因果边界的反例。
- R2-R8在后续阶段目标正式启用前已经失稳，因此R5的necessity、R6的strict transplant、R7的E151物理项和R8的E151三轴项没有得到有效、有限、可归因的训练证据。

## 因子诊断与资源

| Row | z_f域probe | z_n域probe | 有效秩 | Gram条件数 | 峰值VRAM(MB) | 推理延迟(ms) | 训练时间(h) |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1 | 99.2188 | 65.8203 | 1.4667 | 260,038.95 | 11,028.28 | 0.2692 | 4.3800 |
| R2 | 99.4141 | 65.6250 | 1.4667 | 260,250.80 | 11,028.28 | 0.2567 | 3.9210 |
| R3 | 99.2188 | 66.2109 | 1.4670 | 261,266.91 | 11,028.28 | 0.2813 | 3.8129 |
| R4 | 99.2188 | 65.8203 | 1.4688 | 260,674.47 | 11,030.43 | 0.2435 | 4.1673 |
| R5 | 51.9531 | 56.0547 | 2.6435 | 24.92 | 11,027.49 | 0.2126 | 3.8007 |
| R6 | 99.0234 | 65.4297 | 1.4668 | 260,695.13 | 11,030.43 | 0.2368 | 3.9204 |
| R7 | 99.0234 | 68.5547 | 1.5659 | 149,924.05 | 11,036.70 | 0.2315 | 4.2492 |
| R8 | 99.2188 | 66.7969 | 1.5662 | 149,920.53 | 11,039.63 | 0.2845 | 4.2158 |

8个row累计训练时间为32.4673 GPU·h；八卡并行实际墙钟约4.5h。除R5外，`z_f`可被domain probe以99%左右准确率识别，说明宣称的身份因子并未实现域不变性；同时有效秩约1.47-1.57、Gram条件数约15万-26万，和类别塌缩一致。R5改善了秩和条件数并把`z_f`域probe降到51.95%，但clean准确率仅20.49%、类别floor仍为0%，属于诊断解耦而非性能成功。

TX/content probe因本次真实数据诊断覆盖不足为`N/A`；严格Fingerprint Pair及严格移植指标也因FCR-13能力缺失继续为`N/A`，不能写成0或写成“通过”。

## 机制落地、实际启用与可解释边界

| Row | 配置机制 | 实际运行证据 | 判定 |
|---|---|---|---|
| R1 | 身份CE、`L_self`、`L_eta` | E1-E200有有限epoch级FCR指标，但96次反向跳过；最终类别塌缩 | 已启用，负结果 |
| R2 | R1+`L_swap` | E75起非有限，126轮缺loss | 启用后失稳，不可晋级 |
| R3 | R2+`L_shared` | E58起非有限，143轮缺loss | 启用后失稳；LEO均值最高但无有效归因 |
| R4 | R3+`L_latent_cycle` | E69起非有限，132轮缺loss | 启用后失稳 |
| R5 | R4+basic necessity | E27起失稳，早于E91 necessity阶段 | 配置存在，但没有有效necessity证据 |
| R6 | R5+strict transplant | E27起失稳；真实严格pair能力缺失 | 严格指标`N/A`，不能宣称机制验证 |
| R7 | R6+full physics/`L_phys` | E43起失稳，早于E151物理阶段 | Decoder路径存在，但没有有限物理项效果证据 |
| R8 | R7+three-axis | E29起失稳，早于E151三轴阶段 | 三轴真实能力仍blocked，没有有效效果证据 |

## 比较边界

- 按用户要求，本run没有启动R0，也没有启动ADV3B02旧基线。
- 因此不能报告“相对ADV3B02提升/下降多少”，也不能做FCR相对旧模型的因果收益声明。
- 允许的结论只限于R1-R8同一矩阵内部：R1取得最高clean，R3取得最高LEO均值；但所有row类别floor为0，R2-R8存在严重数值失稳，故没有可晋级row。
- 本run是Phase1 source-only训练与四场景评测，不涉及Phase2注册、old/new竞争或`DA0_REG0`等四状态指标。

## 最终裁决

1. 工程状态：`ANALYZED`，训练、checkpoint、四场景prediction、独立进程评分、完整日志和资源诊断均已闭合。
2. 科学状态：`SCIENTIFIC_NEGATIVE_NO_PROMOTION`。
3. 主要失败模式：TX类别塌缩、`z_f`域泄漏、因子低秩/病态、R2-R8持续非有限FCR指标及大规模安全跳步。
4. R1-R8均不得替换ADV3B02，也不得进入多seed或更大矩阵确认。
5. 下一次修复应先用最小同row候选解决“非有限梯度+类别塌缩”，并明确区分FCR标量已被清零与梯度仍非有限的来源；这属于后续研发建议，不改变本run的冻结矩阵和负结果。

## 数据文件

- 完整本地artifact根：`E:\type10-7\local_artifacts\phase1_adv3b02_fcr_r1r8_s392002_20260902_v4`
- 行汇总：`row_summary.csv`（8行）
- 场景汇总：`scenario_results.csv`（32行）
- 每类结果：`per_class_results.csv`（192行）
- 完整结构化分析：`analysis.json`（含全部混淆矩阵、预测直方图、曲线摘要、诊断和完整性检查）
- 可复现独立分析器：`analyze_fcr_v4.py`
- 原始逐样本预测：`R1`至`R8`各自的`fcr_predictions.json`
- 完整训练日志：`R1`至`R8`各自的`train.log`、`metrics.csv`和`logs.jsonl`

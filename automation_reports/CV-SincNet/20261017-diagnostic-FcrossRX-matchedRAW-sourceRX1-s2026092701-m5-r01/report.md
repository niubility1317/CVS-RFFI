# F＋跨 RX：五个同 seed raw 控制的描述性审计结果

**ANALYZED／VERIFIED。** 五个固定历史控制全部保留。随机初始 206 个张量逐对完全一致，实际源角色、全部 L/U 顺序、归一化后的共同配方及记录的普通增强 MSE 也相同；默认头最终 clean 的跨度为 0.952381 个百分点，low urban 为 0.952381 个百分点。它们是一个 model seed 的五个历史控制，不是五 seed，也不是新随机复制实验。不能将跨度当作通用噪声阈值、置信区间或 CUDA 因果结论。

run_id：20261017-diagnostic-FcrossRX-matchedRAW-sourceRX1-s2026092701-m5-r01。本轮 0 新 model epoch、优化更新、网络前向、IQ 或目标访问；读取 5 个随机未训练初始化状态，0 训练后 checkpoint。已独立复算 44,100 个保存源域判决、66,600 个完整 native/view/unit 步、29,970 个 L/U 顺序记录及全部 100 个 epoch。没有晋级候选，也不改变其他正在收尾实验的科学判据。

## 固定来源和执行底座

五个来源在收集前固定，历史分数已暴露：P13 refresh_RAW、P15 timing_RAW、sceneCycle B00、isolatedAWGN B00、boundedPhase B00。精确 run、row、路径、六类 seed 及三阶段命令见 [experiment.json](E:/type10-7/automation_reports/CV-SincNet/20261017-diagnostic-FcrossRX-matchedRAW-sourceRX1-s2026092701-m5-r01/experiment.json)。不按结果删除来源，不训练或重训任何分类头。

共同原配方为 F＋跨 RX 同 TX SupCon、三路 SIG_TIME、统一 Sinc、真实 joint160、NoExplicitCFO、默认 cos30 与骨干原联合训练，scratch E20/4,440，原 13L＋7U/oldgate/ownEMA。完整 base .75raw＋.25ordinary 与 ordinary-only 浅中层原 SupCon 保持，再额外 .25raw 同 IQ RDrop 对；8L 前向、总 pairweight1.25。这里的 raw 控制仍保留共有普通增强，不能冒称完全无增强，也不能和四/六 L 前向的旧控制混为一个预算。

实际 fit L5040/U45360/V21600、源 RX3/4/6/8、day1/2/3、6 TX、split392005；held 源 RX1 原 L1260，仅原 FINAL20 的 clean＋六 practical。U TX 真值始终隐藏；训练 ID 当作 opaque 字符串，只有原 held 源读出使用原五元物理 ID。五组有效角色在随机快照载入前比对一致。无父模型、祖先或目标训练来源，随机快照不用于模型初始化或教师；仅 CPU 张量比较，不实例化 CVS。

## 可比性证据与局限

| 项目 | 实际结果 |
|---|---|
| 随机初始化 | 十对均 206 张量 key/shape/dtype/数值完全相同；最大绝对差与 L2 差均 0；原默认头 6×160 |
| 有效源角色 | 五组全部原 L/U/V role IDs 完全相同，原 source-fold 标记与 scratch/hidden-U/final-only 合法性通过 |
| 完整 L/U 顺序 | 每组 4,440 个 L 与 1,554 个 U 记录、5994 条完整顺序逐项相同 |
| 共同配方 | feature extractor、原 SupCon、精度、base/extra 权重、原 native aug 参数等归一化后十对相同；全部 native 参数不同键只有 run_id/output_dir，部分另有 candidate_id |
| 记录的 ordinary MSE | 全 4,440 步十对逐项相同；不证明已生成 IQ 的逐点相等 |
| 更新健康 | 每组 4,440 步已更新，0 非有限跳过、0 U 真值可用、0 零梯度；所有目标份额分解通过 |
| 正式六场景配置 | 五组保存的 actual_configs 完全相同；没有改 SNR、场景或删包 |
| 已保存逐包物理状态 | refresh_RAW 与 timing_RAW 六景各 1260 条、六景逐项相同；其余三控制未保存这些记录，故相关 54 个 pair×scene 比较为缺失/未知，不是已证实的信道差异 |
| held 源评价配对 | 五组每景同 1260 个物理 ID/labels、每 TX210；所有保存 accuracy/F1 独立重算一致 |

原主分析中的 all_eval_physics_exact=false 表示未获得五组全部逐包状态的相同证据；[覆盖细分](E:/type10-7/local_artifacts/cvs_FcrossRX_RAWcontrol_audit_20261017/final/physics_coverage.json) 明确 AVAILABLE_EQUAL=6、MISSING_BOTH=18、MISSING_SECOND=36。相同配置与存储 physics 也不等于逐点 IQ 相同，当前没有读取或重放 IQ。不能唯一归因于 GPU、CUDA、dropout、loader 或某段实现；各 run 的 unused shadow 构造和代码路径也不同，不能说成严格随机控制的重训复制。

## 逐控制、逐视图结果

准确率，单位 %；全部为一个 model seed2026092701、held 源 RX1L1260，非正式 target。每行均为原 checkpoint 内默认头已固定预测的重算。

| 控制 | clean | practical_high | practical_mid | practical_low_suburban | practical_high_urban | practical_mid_urban | practical_low_urban |
|---|---:|---:|---:|---:|---:|---:|---:|
| refresh_RAW | 80.317460 | 39.126984 | 39.603175 | 36.269841 | 38.492063 | 32.460317 | 27.936508 |
| timing_RAW | 79.841270 | 38.412698 | 39.047619 | 36.031746 | 37.619048 | 32.063492 | 26.984127 |
| cycle_B00 | 80.158730 | 39.920635 | 40.238095 | 36.349206 | 38.968254 | 32.460317 | 27.777778 |
| awgn_B00 | 79.682540 | 38.571429 | 39.206349 | 36.349206 | 38.174603 | 32.222222 | 27.301587 |
| phase_B00 | 80.634921 | 39.126984 | 39.126984 | 35.714286 | 38.174603 | 32.301587 | 27.857143 |

六类 macro F1，单位 %：

| 控制 | clean | practical_high | practical_mid | practical_low_suburban | practical_high_urban | practical_mid_urban | practical_low_urban |
|---|---:|---:|---:|---:|---:|---:|---:|
| refresh_RAW | 79.941358 | 28.304475 | 29.586451 | 28.353370 | 29.072386 | 27.079730 | 24.610874 |
| timing_RAW | 79.343615 | 28.303558 | 29.361850 | 28.450928 | 28.793624 | 26.669836 | 23.660254 |
| cycle_B00 | 79.742937 | 29.049466 | 30.224256 | 28.312996 | 29.583784 | 26.906529 | 24.569258 |
| awgn_B00 | 79.253153 | 27.976096 | 29.319085 | 28.410315 | 28.823502 | 26.839303 | 23.861963 |
| phase_B00 | 80.327323 | 28.736903 | 29.406604 | 27.945410 | 29.147531 | 26.706456 | 24.550875 |

| 视图 | 均值（%） | 最低（%） | 最高（%） | 跨度（百分点） | 描述样本 SD（百分点） |
|---|---:|---:|---:|---:|---:|
| clean | 80.126984 | 79.682540 | 80.634921 | 0.952381 | 0.378963 |
| practical_high | 39.031746 | 38.412698 | 39.920635 | 1.507937 | 0.592321 |
| practical_mid | 39.444444 | 39.047619 | 40.238095 | 1.190476 | 0.492447 |
| practical_low_suburban | 36.142857 | 35.714286 | 36.349206 | 0.634921 | 0.272628 |
| practical_high_urban | 38.285714 | 37.619048 | 38.968254 | 1.349206 | 0.494362 |
| practical_mid_urban | 32.301587 | 32.063492 | 32.460317 | 0.396825 | 0.168359 |
| practical_low_urban | 27.571429 | 26.984127 | 27.936508 | 0.952381 | 0.410863 |

以上 mean/SD 只描述这五个历史控制。未将同 seed 的五次控制当独立模型种子，没有因这组跨度而变更当前或已关闭候选的阈值、排名、配方及种子安排。

## 首次分歧与配对错误

所有十对的梯度记录从第 1 步已有严格浮点差异，幅度约 2.20e−13 至 1.49e−8；超过 1e−5 的差异从第 3 或第 4 步出现。首次 loss 严格差异在第 3 或第 4 步，超过 1e−5 在第 5 至第 10 步。这些是时间先后描述，不是引发最终误差的独立归因。

| 第一→第二控制 | 首次 loss 差异步 | 首次 loss 差异 >1e−5 | 首次梯度差异 >1e−5 | 第 1 步梯度绝对差 | clean 修复/新增错误 | low 修复/新增错误 |
|---|---:|---:|---:|---:|---:|---:|
| refresh_RAW→timing_RAW | 3 | 10 | 3 | 1.1249e-8 | 3/9 | 6/18 |
| refresh_RAW→cycle_B00 | 3 | 5 | 3 | 1.1250e-8 | 3/5 | 11/13 |
| refresh_RAW→awgn_B00 | 3 | 10 | 3 | 1.2873e-8 | 1/9 | 4/12 |
| refresh_RAW→phase_B00 | 4 | 8 | 3 | 2.0698e-9 | 5/1 | 8/9 |
| timing_RAW→cycle_B00 | 3 | 6 | 3 | 2.2027e-13 | 8/4 | 21/11 |
| timing_RAW→awgn_B00 | 3 | 6 | 4 | 1.6232e-9 | 4/6 | 11/7 |
| timing_RAW→phase_B00 | 3 | 8 | 3 | 1.3319e-8 | 12/2 | 14/3 |
| cycle_B00→awgn_B00 | 3 | 5 | 3 | 1.6230e-9 | 5/11 | 15/21 |
| cycle_B00→phase_B00 | 3 | 5 | 3 | 1.3319e-8 | 8/2 | 14/13 |
| awgn_B00→phase_B00 | 3 | 8 | 3 | 1.4942e-8 | 12/0 | 14/7 |

十对 clean hard 判决变化 0.555556% 至 1.587302%，low 为 3.015873% 至 6.031746%。例如 timing 相对 refresh 的 clean 修复3/新增9、low 修复6/新增18；cycle 与 awgn 的 low 判决有76包不同，但正确数仅净差6。总准确率接近不代表逐包身份判断相同，也不能把不同 run 的小增益相加。

## 完整曲线与实际图表

[逐景准确率与 F1](E:/type10-7/local_artifacts/cvs_FcrossRX_RAWcontrol_audit_20261017/final/figures/historical_control_metrics.png)、[十对 hard 判决变化](E:/type10-7/local_artifacts/cvs_FcrossRX_RAWcontrol_audit_20261017/final/figures/paired_hard_decisions.png)、[全部 100 轮训练曲线](E:/type10-7/local_artifacts/cvs_FcrossRX_RAWcontrol_audit_20261017/final/figures/complete_training_curves.png)；对应同名 PDF 已保存，3 PNG＋3 PDF 均实际生成，三张 PNG 已逐张核看。

![完整逐景结果](E:/type10-7/local_artifacts/cvs_FcrossRX_RAWcontrol_audit_20261017/final/figures/historical_control_metrics.png)

![完整曲线](E:/type10-7/local_artifacts/cvs_FcrossRX_RAWcontrol_audit_20261017/final/figures/complete_training_curves.png)

五组 E20 mean total_L 为 0.334819 至 0.335276，mean preclip 梯度 0.685596 至 0.704541，seenV 为 98.004630% 至 98.037037%。E14 起 original U，伪标签选择量按 epoch SUM，不能解释为独立样本数；这些健康及损失统计不是泛化收益或收敛证明。图表使用已有 scalar 聚合，无 IQ、checkpoint、NN 或训练。

## 实现、检查与成本

运行代码 b167b4864ff6bd408a2a87e5dae71da42fceefb8；离线 physics 缺失分类/图表 329e70ee4bd0d4fb5c00d7431f9293b5e7bc0190；直接 forward-counter 口径修正 c8ca5ba4df8f2fd5a271a7a6d202b5d74b6a9199，三提交均 push 后独立 remote OID 比较 identical。仅此 package、登记/报告及相关索引提交，未提交权重、IQ、原始预测、ID、日志、生成图或机器状态。原有无关 CPU.md/coverage.json 变化保留，未暂存。

13 个协议负例、206 合成张量及 3 个状态差异控制 PASS；5 个 physics availability 对照 PASS；所有新增模块 compile PASS。唯一独立 P0/P1 静态审查 PASS/0 findings，未重复完整审查。实际首份合法随机状态结构与有限性检查 PASS，随后完成余四份；0 NN，不能称为真实模型前向 smoke。SSRPython3.10/Torch2.1 CPU2 保留已知 NumPy2 桥警告，直接 tensor 操作成功，没有使用 NumPy 桥或修改环境。

普通 N607 只读收集 135 文件，本地 immutable inputs 与五快照；仅标量与已提交 plot 代码在已有 Matplotlib 环境生成图。第一次图表父目录缺失，远端独立核对无任何产物后仅修正 mkdir parents 并继续，失败 stderr 保留，不涉及训练重跑。

历史成本为 100 model-epochs／22,200 更新。每行保存的 student-train4,942,928、student-eval475,202、teacher-train0、teacher-eval396,370，加 held8820，直接合计 5,823,320，五行合计 29,116,600；保留原计数中的 smoke，不作无依据扣减。这是前向样本量，不是唯一物理样本数。v1 曾每行减2、合计29,116,590，已更正为直接计数；v1 输出保留，v2 全量复核确认除该成本字段与口径说明外所有结果和完整曲线未变，不影响科学判据。

原 row 墙钟分别 1416.171875／2301.330696／1403.310508／1393.016820／1388.088959 s；它们不构成同一次矩阵墙钟或公平吞吐比较。新增随机状态比较主体0.321179s，成本修正后完整元数据复算12.596s；第一次元数据读取与文件传输/绘图未单独计时，不编造整轮墙钟。新增 0 modelE/更新/NN/教师/EMA/头拟合/IQ/target，5 初始 CP 读取、0 训练后 CP 读取。全部源判决来自原冻结读出，没有重预测。

权威成本修正版 [summary](E:/type10-7/local_artifacts/cvs_FcrossRX_RAWcontrol_audit_20261017/final/cost_recount_v2/analysis_summary.json) 与 [复核](E:/type10-7/local_artifacts/cvs_FcrossRX_RAWcontrol_audit_20261017/final/cost_recount_v2/verification.json)；原 [完整曲线](E:/type10-7/local_artifacts/cvs_FcrossRX_RAWcontrol_audit_20261017/final/training_curves.json)、[十对初始化](E:/type10-7/local_artifacts/cvs_FcrossRX_RAWcontrol_audit_20261017/final/initial_states.json)、physics 覆盖及图保持。canonical experiment.json/events 与镜像报告指向同一 run，路径不可覆盖。

## 当前判断和未完成项

本轮证实五个名义同配方控制的实际初始化、role 与顺序一致，同时最终默认头判决仍有差异；没有证明某个新 Ag/As 有效，没有证明低 urban 退化被训练改善。当前四行 SupCon 视图组织实验的两新行已由原进程完成 E20/4440/七源视图；原 owner 独立终态分析仍待其报告，本轮不接管、重复评分或修改它的固定 Q_mix 判据。

当前唯一在等待结论的训练假设仍是：保持完整原 raw/ordinary/As 监督和总预算，仅将原跨 RX 同 TX SupCon 的一半出现位置用于 As，是否比 ordinary-only 与 all-As 同时保留 clean 并提高 low/urban。最小矩阵已预登记为 Q0/Q_ord/Q_sat/Q_mix，失败按原规则关闭，不改比例、层位、损失、物理概率、RX、seed 或预算补救；本描述性审计不是新 gate，也不调整该规则。

新增身份保持 IQ Ag、匹配 B00/B10/B01/B11 与交互、正式 clean＋六 practical 的共同收益、low urban>60%、suburban 接近自身 clean、最差未见 RX 及跨 model seed 重现均未完成。不能以这次历史复算替代最终验证，goal 保持 ACTIVE。

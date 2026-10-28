# PAMF Phase3 源代理联合基线结果

**结论：真实运行技术完成，科学结果为 `NO_PROMOTION`。** 固定四类 CE20 表示上的公共平移＋anchored DP-means 未发现合格匿名组件，D1 还损伤旧类识别。本结果只否定本次表示、几何、尺度和预算的联合候选，不证明所有聚类方法或完整六类 PAMF 不可行。

- run：`20261029-phase3-source-joint-s2026092701-r01`；row：`source-joint-fold0-s2026092701`。
- 实际代码 commit：`6570e891982d72bc317c9219ead3e97221d93951`。
- 实际状态：`VERIFIED / ANALYZED_SOURCE_ONLY`；`eligible_states=[]`；`target_performance_claim=false`。
- 证据来自已取回的完整产物；本次分析未重新拟合、扫参、启动预测或读取新 Full／正式 target。

## 数据、权限与完成情况

父模型是本轮四类 scratch CE20/540 模型，旧模型标签 0/1/2/3 对应原 source labels 2/3/4/5（20-15、20-19、6-15、8-20）。原 source label0（14-10）在本代理中记为可发现类4，label1（14-7）记为后续未知类5；它们不是正式 Full 四个未知 TX。

源训练 L3360 仅建立冻结四类锚点与共享协方差；已见源 RX1/3/4/6 的旧类 V14400 只校准一条 99% higher 分位半径。U1050 是原 F0 L 中 RX8 的 source labels 0/2/3/4/5 隐藏标签代理；RX8 label1 的 L210 未开放、模型前向为0。future 是 RX8 V5400，每个模型真类900条，旧四类共3600条。RX8 对应 Main 接收机 3-19。

原完整 V27000 未转成训练；本次未加载 V IQ，复用冻结分支产物中的特征、logits 和元数据。builder 私下加载完整 L6300 以筛选合法角色，不等于这些记录都参加模型训练或前向。真实前向只有两条 selected L smoke＋U1050，共1052条。原 U_s 未打开，F1/F2 历史 query artifact 未读取；源训练、U、future 物理角色互斥由 builder 完成标记记录。

RX8 曾在其他划分作为 target，存在跨划分历史曝光。本轮属于 F0 源角色开发，不是新盲 target，不是正式六类零标签注册或真实在轨证明。

四阶段完成标记依次为 `SOURCE_STREAM_INPUTS_EXPORTED` → `SOURCE_STREAM_CALIBRATED` → `ALL_STREAM_PREDICTIONS_COMPLETE` → `SCORED_SOURCE_STREAM_COMPLETE`，均为 source-only、target_reads=0。U 状态在读取 future 前冻结；future 逐样本预测不拟合；匿名映射仅可在预测完成后由 U 真值决定，future 不重新匹配。

真实 own-checkpoint smoke 为 PASS：2条合法训练 L，feature [2,160]、logits [2,4]，无 validation/target 读取或优化步。发布前 local_verification 的“smoke 尚未执行”是当时快照，实际完成以 export/smoke.json 和完整 export.log 为准。本 run 的神经 optimizer_updates=0、head_changed=false；D1 仅改变冻结完整 cosine head 的输入，未重训或替换 head。

## 五状态未来评价

百分数保留4位小数。旧类指标始终对旧四类等权宏平均，包含拒识和误注册造成的识别错误；FRR 只统计 -1，错误旧类或匿名接受不算 FRR。R0 未注册，新类 accuracy 为 N/A；R1 注册模式已开启但注册表为空，新类 accuracy 按0/900计。

| 状态 | 同状态闭集旧类宏准确率 | 开放旧类宏准确率 | 旧类宏FRR | 类4匿名准确率 | 类4拒识率 | 后续类5拒识率 |
|---|---:|---:|---:|---:|---:|---:|
| F | 66.4722% | 65.1389% | 3.7778% | N/A | 5.3333% | 2.4444% |
| D0R0 | 66.4722% | 65.1389% | 3.7778% | N/A | 5.3333% | 2.4444% |
| D1R0 | 65.4722% | 63.9167% | 4.0278% | N/A | 6.2222% | 2.1111% |
| D0R1 | 66.4722% | 65.1389% | 3.7778% | 0.0000% | 5.3333% | 2.4444% |
| D1R1 | 65.4722% | 63.9167% | 4.0278% | 0.0000% | 6.2222% | 2.1111% |

原始闭集旧类宏准确率为66.4722%（2393/3600）；D1 校正输入后的同一个冻结 head 为65.4722%（2357/3600），下降1.0000个百分点。D1 的开放准确率从65.1389%降到63.9167%，R0、R1配对收益均为 **-1.2222个百分点**，即少44条旧类正确预测。相对原始闭集的损失为2.5556个百分点，超过2个百分点门槛；相对 D1 自身已校正闭集的损失为1.5556个百分点，二者口径不能混写。

D0 的旧类拒识136/3600，D1 为145/3600。后续未知类5只拒识22/900与19/900，分别为2.4444%和2.1111%，远未达到95%。类4的5.3333%／6.2222%拒识是未注册发现类的辅助诊断，不能替代注册后的持续识别率。

D0R0=D0R1、D1R0=D1R1，注册旧类损伤均为0。原因是两者没有任何匿名 slot；零损伤不构成注册安全性证据。两状态 U 匹配覆盖率为0，未匹配纯度和聚合纯度为 null，不用0替代“无可评价注册”的缺失值。

只有完整 D1R1 可以晋级。该状态失败于：未收敛、触及组件上限、无合格组件、registry 不可晋级、后续未知拒识不足95%、相对原闭集旧类损失超过2pp、新类准确率不足80%，以及域校正旧类收益非正。旧类 FRR≤5% 和注册损伤≤2pp 单独通过不改变 NO_PROMOTION。

## 发现状态与失败位置

冻结平方半径 λ_det=262.830305312，λ_cluster=65.707576328；合格组件 r90²≤16.426894082，两半质心距离平方≤8.213447041。本次不修改这些阈值，也不增大组件上限挽救原 run。

| 诊断 | D0 | D1 |
|---|---:|---:|
| 已执行外轮 | 1 | 1 |
| 自由组件／上限 | 32/32 | 32/32 |
| 收敛 | false | false |
| cap_reached／limit_reached | true／true | true／true |
| 合格匿名组件 | 0 | 0 |
| 自由组件分配记录数 | 428 | 374 |
| 单例组件数 | 14 | 15 |
| 组件大小最小／中位／最大 | 1／2.5／86 | 1／2／88 |
| r90²最小／中位／最大 | 0.0000／170.7602／234.8702 | 0.0000／112.3198／238.1340 |
| 本轮剩余最大最近距离平方 | 234.8702 | 238.1340 |
| b范数 | 0.000000 | 3.179036 |

两状态均在第一外轮的内层最远点出生中产生32个组件、0次合并，仍有最大最近距离平方超过 λ_cluster，按固定规则触顶退出。它们没有进入第二外轮的中心均值更新／合并，不应把这些截断组件称为已收敛身份簇。D0 objective=153227.3933，D1=150138.4491；目标值变小也没有转化为合格注册或身份收益。

资格失败数如下，分母各为32个自由组件；多个条件可在同一组件同时失败，不能相加为组件总数。

| 资格条件 | D0失败数 | D1失败数 |
|---|---:|---:|
| 至少16个不同ID | 26 | 27 |
| 两半各至少8条 | 27 | 27 |
| r90²紧凑性 | 18 | 17 |
| 与旧锚点分离 | 18 | 17 |
| 两半质心一致 | 28 | 29 |

通过紧凑性和分离条件的组件恰好分别只有14、15个单例；这些单例又必然过不了16条与两半样本量要求。D0只有6个、D1只有5个组件达到16条，但它们没有通过完整资格。当前状态同时存在单例碎片和半径很大的组件，没有形成可注册的新身份。

D1 平移范数为3.179036，上限3.182266，达到上限的99.8985%。10轮核心计数按冻结 head 的伪类0/1/2/3排列，从[0,49,63,0]变为[0,69,76,0]；两端伪类始终无核心，平移只由中间两个伪类估计。核心伪标签不是真值，本分析不能据此断言其正确率或未知污染量；但它不支持“公共域偏移已由四类一致估计”的解释。完整核心计数、范数轨迹和逐组件诊断保存在 analysis_summary.json。

这次失败同时受当前弱四类 CE20 表示、共享 Mahalanobis 几何、固定细分尺度及32组件截断预算约束。没有合格组件不等于不存在可辨认 TX；也不能由本次未收敛结果推导增大 cap 一定有效。新设计如改变表示、出生／合并顺序或预算，应另登记候选，不覆盖或选择性重跑本次结果。

## 执行、日志与验证口径

pipeline wall time=6.700397 s：export=3.839293 s，calibrate=0.478137 s，predict=2.218227 s，score=0.162269 s。仅 export 使用物理 GPU2，后三阶段 CPU；foreign_processes_stopped=0。独立 post_state 记录 driver 及四子进程均已退出、所有 job COMPLETE、无 failure。

该时长包含本轮导出与 CPU 几何／评分流程，复用了已有 V/训练特征，且不包含父模型训练；不能当作完整 RFFI 逐窗口部署时延或模型训练速度。

四份日志全部读取，共19行、1281字节，不使用tail代替完整扫描；未发现 Traceback、ERROR、OOM、Killed、NaN/Inf 或 Warning。科学状态中的 converged=False、cap=True 按协议保留，属于已完成实验的失败证据，不是未完成的技术异常。

初始整体验证为68项 pytest＋11项 subtests。P1修复“控制组通过误推进完整联合候选”后，相关48项通过；其余23项代码／测试未变，因此当前唯一测试覆盖为48＋23=71项，并非把68＋48相加，也未声称完整重跑71项。独立审查随后有4项针对复核通过，单独记录，不重复累计为本地完整套件。合成全链与真实 checkpoint smoke 分开陈述，均不能充当真实源性能成功。

## 后续方向与限制

本候选停止晋级，不扩种子，不访问已准备好的 Full future query，不通过改 cap、λ 或资格条件重做本 run。后续可独立比较：在当前满意的完整六类 Tri-SIG-TIME SSL PAMF 上开展开放开发；或在匹配预算下恢复仅用合法 L 的跨 RX 身份约束／DG 表示。两条路线均未由本报告执行，也不承诺可以提升。

既有元数据调查确认 Full 有24个 TX 不在四compact并集中，且与正式四未知和旧六类不交；其中7个在原5源RX×D1–D3的15单元均≥70条。该事实可支持另设计 source-OE 开发／暴露类，但不是已建合法OE数据。源角色、标签可见性、物理排除与新adapter继承仍须明确，正式四未知不能通过源开发选择暴露。若使用新增源标签或“非旧类”状态，应如实称源OE监督，不称零源标签。

本报告未读取新 Full 数据；上述事实仅引用已经完成的[OE元数据调查](E:/type10-7/local_artifacts/pamf_phase3_source_oe_metadata_20261029_r01/report.md)。没有进行目标预测、注册或后续方法试验。

## 完整证据入口

- 冻结方案：[源联合计划](E:/type10-7/docs/PAMF_PHASE3_SOURCE_JOINT_PLAN_20261029.md)。
- 本地核心结果：[score/results.json](E:/type10-7/local_artifacts/pamf_phase3_source_joint_20261029_r01/score/results.json)；[predict/states.json](E:/type10-7/local_artifacts/pamf_phase3_source_joint_20261029_r01/predict/states.json)；[analysis_summary.json](E:/type10-7/local_artifacts/pamf_phase3_source_joint_20261029_r01/analysis_summary.json)。
- 执行和独立后态：[pipeline_state.json](E:/type10-7/local_artifacts/pamf_phase3_source_joint_20261029_r01/pipeline_state.json)；[post_state.json](E:/type10-7/local_artifacts/pamf_phase3_source_joint_20261029_r01/post_state.json)；[landing_readback.txt](E:/type10-7/local_artifacts/pamf_phase3_source_joint_20261029_r01/landing_readback.txt)。
- 配置与发布身份：[configs/export.json](E:/type10-7/local_artifacts/pamf_phase3_source_joint_20261029_r01/configs/export.json)；[runtime.json](E:/type10-7/local_artifacts/pamf_phase3_source_joint_20261029_r01/runtime.json)；[package.json](E:/type10-7/local_artifacts/pamf_phase3_source_joint_20261029_r01/package.json)。
- 源角色与真实smoke：[export/builder_report.json](E:/type10-7/local_artifacts/pamf_phase3_source_joint_20261029_r01/export/builder_report.json)；[export/fold_manifest.json](E:/type10-7/local_artifacts/pamf_phase3_source_joint_20261029_r01/export/fold_manifest.json)；[export/smoke.json](E:/type10-7/local_artifacts/pamf_phase3_source_joint_20261029_r01/export/smoke.json)。
- 四阶段完成标记：[export/complete.json](E:/type10-7/local_artifacts/pamf_phase3_source_joint_20261029_r01/export/complete.json)；[calibration/complete.json](E:/type10-7/local_artifacts/pamf_phase3_source_joint_20261029_r01/calibration/complete.json)；[predict/complete.json](E:/type10-7/local_artifacts/pamf_phase3_source_joint_20261029_r01/predict/complete.json)；[score/complete.json](E:/type10-7/local_artifacts/pamf_phase3_source_joint_20261029_r01/score/complete.json)。
- 四份完整日志：[logs/export.log](E:/type10-7/local_artifacts/pamf_phase3_source_joint_20261029_r01/logs/export.log)；[logs/calibrate.log](E:/type10-7/local_artifacts/pamf_phase3_source_joint_20261029_r01/logs/calibrate.log)；[logs/predict.log](E:/type10-7/local_artifacts/pamf_phase3_source_joint_20261029_r01/logs/predict.log)；[logs/score.log](E:/type10-7/local_artifacts/pamf_phase3_source_joint_20261029_r01/logs/score.log)。
- 本地与独立验证记录：[local_verification.json](E:/type10-7/local_artifacts/pamf_phase3_source_joint_20261029_r01/local_verification.json)；[focused_tests.log](E:/type10-7/local_artifacts/pamf_phase3_source_joint_20261029_r01/focused_tests.log)；[focused_p1_tests.log](E:/type10-7/local_artifacts/pamf_phase3_source_joint_20261029_r01/focused_p1_tests.log)。独立4项复核结论记录于 local_verification 和原run登记报告，未另提供独立stdout日志。
- 原run登记入口：[report.md](E:/type10-7/automation_reports/CV-SincNet/20261029-phase3-source-joint-s2026092701-r01/report.md)；实际完成状态以本次 pipeline/score/post_state 为准。
- N607原始root：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase3-source-joint-s2026092701-r01`。完整特征、calibration/bundle.npz、三份predict NPZ及私有truth保持原root；本地分析读取的是已取回的评分／状态／日志，没有重开这些数据。

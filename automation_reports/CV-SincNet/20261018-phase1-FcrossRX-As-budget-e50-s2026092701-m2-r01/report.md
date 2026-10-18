# 原完整 practical 配方：E30 至 E50 匹配预算验证

当前状态：**已在 N607 启动并独立核实运行。** 实际训练 release 提交为 `0f399c910324e21a8b1a60346d879be5c2964a9c`；实现提交为 `115dc1c9e5d7d392e820d6230bbf66aaae08f8bc`。一次独立 P0/P1 审查 PASS，远端 30 项协议负例、真实父 checkpoint 的 2 NN 无 query smoke、两行真实 F 的 1,536 NN 合成检查均 PASS。E50 训练和正式结果尚未完成。

本次依据用户 2026-10-18 的明确授权“对于表现好的，可以跑 E50 更多预算，看看有没有提升”，研究原完整 As 的预算收益。唯一假设是：在其余配方保持不变时，原完整 As 各自固定 FINAL30 追加 20 轮，可进一步改善 low／mid urban，并在累计 E50 的匹配 NoAs 对照下控制 clean 代价。此处不新增 Ag、结构、损失或门控，不是 satelliteDropout 候选的救跑，也不依赖该候选的分析恢复。

## 候选依据与比较矩阵

候选依据仅为原完整 As 固定三模型种子的合法 source 结果。以下 full V27000 指全源验证集；星地增益取同一 V1800 的两个已有信道 draw，两个 draw 不算两个模型种子。源 panel 只有 low suburban、mid urban、low urban，没有 high urban，不生成“源域 urban 三场均值”。

| model seed | full V clean NoAs | full V clean As | clean 配对变化 | mid urban 配对增益（两 draw） | low urban 配对增益（两 draw） |
|---|---:|---:|---:|---:|---:|
| 2026092701 | 98.359259% | 98.255556% | −0.103704 pp | +29.250000 pp | +23.472222 pp |
| 2026092702 | 98.396296% | 98.325926% | −0.070370 pp | +20.638889 pp | +16.722222 pp |
| 2026092703 | 98.422222% | 98.355556% | −0.066667 pp | +25.027778 pp | +19.333333 pp |

source clean 配对均值为 −0.080247±0.020398 pp；mid／low 均值增益为 +24.972222／+19.842593 pp，两场均值 +22.407407 pp。01 的 clean 下降略超过 0.1 pp，必须保留该事实，不能凭源均值宣称正式 clean 达标。历史 source_As_replicated_all3=true 使用其原判据，不改写成新的 0.1 pp 阶段通过。

来源：`E:/type10-7/local_artifacts/cvs_FcrossRX_As_replication_E30_s0203_20261018_r02/analysis_pipeline_publish_r01/retrieved/source/` 下的 `source_decision.json`、`source_summary.json`、六份 `B00/B01-sSEED_training.json` 和 `training_stream_comparability.json`。full V 绝对值来自最后一条 epoch30.source_val_accuracy_pct；配对差与 source_decision 一致。此依据未读取正式指标／target truth，不把任何既有 formal 成绩用于选择候选或种子。

首轮固定单种子 2026092701 的两行最小预算验证。它是既定种子顺序中的第一项，不是从三种子挑出的最好种子；automatic_seed_extension=false。两个 E30 行及其原冻结预测仅复用作预算参照，不重新训练或重新测试。

| 行 | 正式训练来源 | 新增预算 | 累计预算 | 新增参数／损失 |
|---|---|---:|---:|---|
| B00-s2026092701 | 原 P43 B00 FINAL30 student＋own EMA | E20／4,440 更新 | E50／11,100 更新 | 无；匹配 NoAs |
| B01-s2026092701 | 原 P43 B01 FINAL30 student＋own EMA | E20／4,440 更新 | E50／11,100 更新 | 无；原完整 practical As |

原训练 run：`20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01`，固定运行依赖提交 `19aedc3604fed9c193237dd92366164178af0983`。每行继承自己的 `source/final_ssdg.pth` 中 student 和 EMA，并核对自己的 scratch20 祖先；不交换 As／NoAs 父权重，不使用来源不明、契约不匹配或目标选出的权重。原 E30 在固定源训练冻结后的独立正式评分不等于训练污染，但该评分不是本次晋级依据。

## 固定训练与数据契约

保留 F 的三路 SIG_TIME、统一 Sinc CENTER_UNITS、真实 joint160、6×160 默认 cosine 头（scale30）与正常联合训练。NoExplicitCFO 保持，必要的内部逐包补偿保留。无部署分支、无冻结骨干后另训头、无诊断分类器。普通旧增强在两行共同保留，NoAs 不是“完全无增强”。

源角色始终为 L6300／U56700（TX 真值隐藏）／V27000，split392005；源 RX=[1,3,4,6,8]、day=[1,2,3]，全部 6 TX，无训练域留出，无 L_train／L_dev 内部分折。沿用现有 256 点 IQ、equalized index1；不要求更长上下文。L 在累计全部 50 轮均参与监督；U 沿用既有规则与各自 EMA，不增加 U 监督／星地 U／额外教师前向。V 仅作无状态更新的源验证。

固定正式 target RX=[0,2,5,7,9,10,11]、day=[0,1,2,3]，数组原样来自 P43 B00／B01 `source/actual_native_args.json` 的 wisig_test_rxs／wisig_test_days，不作索引基准转换。具体本地证据为 `E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/runtime_metadata_r01/B00-s2026092701/source/actual_native_args.json` 第 823／825 行，B01 相同。source／target 允许日期重叠，物理 ID 与 RX 权限仍严格分离。

原完整 As 保持 `.5 RD(ordinary128)+.5 RD(mixed128)+.25 RD(raw128)+.1 CROSS_RX(ordinary)`，每步 6×128 次有标签前向，总 L 系数 1.25。mixed 的 64 个交替位置使用原完整 practical；全部合法卫星样本参加 CE，不加置信筛选，保留 satellite 的 KL 与 dropout。RDrop 两次前向复用同一份已生成 IQ。NoAs 使用相同视图数／损失尺度／前向预算。未启用 Ag_mode=NONE 对应的继承 noise／clock 字典。

新增段为 20 轮，AdamW 重置，lr=0.0002、weight_decay=0.0001、cosine20、lr_min=0.000001；不是恢复 E30 优化器轨迹。内部 segment=30 仅为旧 backend 的父权重载入标记，实际 label_epochs=0、pseudo_epochs=20。新段 EMA 成功更新 0 至 4,440，累计计数加上原 6,660 至 11,100；启动平均关闭。普通增强沿用原 scratch20 args／epoch13 冻结安排，CROSS_RX 按累计 epoch+30 使用原系数，不因本段局部计数重新预热。

训练 As 仅使用原 full practical residual／post_sync／noEQ：low_suburban、mid_urban、low_urban；保持传播、接收质量、噪声与补偿的依赖。训练 seed 为 20261016421，validation 为 20261016422，receiver 为 20261016423。固定正式 received IQ、SNR、信道、所有困难包不修改。

## 读出与判定

两个 row 均固定 FINAL50，不按源或正式成绩挑 epoch。每行自动执行源读出、冻结后的 clean＋六 practical 全部预测，全部七视图预测完成后才由独立 exec 的 CPU scorer 连接 truth。任何源科学结果均不跳过本次已登记正式评分；无自动扩种、选种、参数微调或选择性重跑。源检查只处理真实标签／权限错误、数值异常与明显损伤。

source 报告 full V27000 clean 及各源 RX 准确率；固定 V1800 两 draw 的 clean、low suburban、mid urban、low urban、两 urban 场均值、F1、最差 RX、必要的错误修复／新增错误和已有状态分层。full V 未保存逐包预测时不虚构 full V F1／逐包错误重算。正式报告 clean 与全部六 practical 的准确率、宏平均 F1、最差 RX，重点列 low／mid urban 和 high／mid／low 三 urban 均值；同时保留 RX／day／TX 分层与配对错误修复／新增错误。

对每个可比指标 A，同时报告：

- 预算收益：A(As50)−A(As30)、A(NoAs50)−A(NoAs30)。
- E50 As 收益：A(As50)−A(NoAs50)，并列原 E30 差值。
- 预算交互：A(As50)−A(NoAs50)−A(As30)+A(NoAs30)。

阶段目标为正式 clean 配对变化不低于 −0.1 **绝对准确率百分点**，并提高 practical，重点 low／mid urban。最终目标仍为可重复的星地和 clean 双提升。单种子或不足 0.1 pp 的微小正变化仅为初步观察，不能宣称稳定双提升。若新增预算没有产生源域 practical 正收益，或 clean 明显受损，则该固定预算假设不获支持；保留全部已登记读出，不微调／挑种子救跑。

## 成本、命令与执行边界

| 成本范围 | 模型轮数 | 更新 | identity NN |
|---|---:|---:|---:|
| 本次两行新增 | 40 model-epochs | 8,880 | 14,921,048＋共享 smoke 1,538 |
| 两行原父训练及原冻结读出 | 60 model-epochs | 13,320 | 18,298,572（单列历史成本） |
| 累计两条轨迹 | 100 model-epochs | 22,200 | 33,219,620＋共享 smoke（本次 1,538，历史共享另保留原账） |

每新 row 的预计 native 训练／验证为 6,271,922 NN：student train 4,543,920、student eval 594,002、teacher train 0、teacher eval 1,134,000。另计冻结 source 12,600 NN、正式七视图 1,176,002 NN，合计 7,460,524 NN。共享 smoke 为真实父 checkpoint 无 query 2 NN＋双真实 F 合成 1,536 NN，0 更新。实际更新、L／As／U、学生与教师前向、冻结读出、wall time 由日志独立汇总；未产生的成本不写成实测。

公开路径采用 PROJECT 与 CVS-RFFI/python 别名，私有真实路径／argv 保留在 local runtime 原件。

- 不可变 release／CWD：`PROJECT/releases/cvs_FcrossRX_As_budget_E50_20261018_r01`。
- 不可变 run：`PROJECT/runs/20261018-phase1-FcrossRX-As-budget-e50-s2026092701-m2-r01`，日志 `logs/{row_id}.log`。
- 本地发布入口：`CVS-RFFI/python -m experiments.cvs_practical_budget50.publish`。
- 远端主入口：`CVS-RFFI/python -u experiments/cvs_practical_budget50/runner.py`。
- row 命令：`CVS-RFFI/python -u experiments/cvs_practical_budget50/runner.py --mode worker --row B00-s2026092701`（B01 同式）。
- owner：`codex/root/practical-budget50`；普通 N607 账户，CPU2 线程；GPU 动态分配，至少 12,000 MB 空余，同 GPU 所有 owner 总进程最多 3 个。

发布器拒绝覆盖既存 release／archive／run；一次 N607 preflight、一次 archive 本地／远端 SHA 比较、一次远端编译。launcher 首步真实 no-query smoke PASS 后直接继续。启动后独立核对 PID／CWD／argv／GPU／日志增长。仅遇本 run 的契约／父权重／头／数值／输出冲突／非法预测／不能运行等系统性技术错误时保留现场并停止尚未启动的本 run 行，不自动重跑，不停止其他健康任务。

预期产物为父权重来源／args／完整日志与 checkpoint／source_frozen、源 panel 读出、全部七视图不可变预测、独立评分、成本与配对分析。自动汇总只到 SCORED_COMPLETE_PENDING_FULL_LOG_ANALYSIS；完成独立全日志与逐包配对分析后才可写 ANALYZED。当前已有实际训练日志及初始化产物；FINAL50、源读出、正式预测及评分尚未完成。

## 实际运行证据

独立观察时间：2026-10-18T03:56:37.970Z。dispatcher PID 1143533 存活；两行 CWD／argv／GPU 环境与登记一致，GPU 进程可见，日志相对初次观察持续增长。

| row | worker PID | GPU | 累计轮数 | 新段更新 | 累计更新 | 日志增长 |
|---|---:|---:|---:|---:|---:|---:|
| B00-s2026092701 | 1143555 | 1 | 35／50 | 1110／4440 | 7770／11100 | +25774 bytes |
| B01-s2026092701 | 1143703 | 2 | 33／50 | 666／4440 | 7326／11100 | +15466 bytes |

两行均为原有 U 规则，extra_U_NN=0、U_As=0、U_truth_read=false、target_access=false；未出现 failure。新段 L_NN 和 As 计数只包含本次追加段，不能冒充累计 E50 成本。普通账户 preflight、一次 release archive SHA 比较与远端编译已完成；不为报告补值重新做哈希。详见 execution_verification.json。

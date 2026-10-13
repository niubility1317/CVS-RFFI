# Phase1 84% 目标：特征提取与历史旧门控基线

日期：2026-10-13。目标保持 active，现有完整结果未达到目标域 clean 三主种子均值 84%。

用户最新明确要求：提升网络特征提取能力，保留现有分类头；对比基线使用历史“旧伪标签 batch_neighbor 门控＋EMA＋cos”完整配方，不能以纯 CE 替代。用户已经从项目两种“旧门控”定义中明确选择历史配方，此含义不再需要重复确认。此前纯 CE E50 保留为独立历史结果，不追溯改写成该基线。

后续 epoch 由 Agent 自主选择，原则每模型不超过 50。当前固定四组×三种子：OLD_GATE_COS50、TIME_MS50、PA_MS50、JOINT_MS50；主方案 JOINT_MS50，种子 2026092701/02/03。每模型从零 50 轮、222 步/轮、11100 次更新。历史基线原 130＋70 的阶段比例缩为 33 轮有标签阶段＋17 轮伪标签阶段；这属于预先固定的 50 轮版本，不冒称相同 E200 训练成本。所有结构对照采用同一日程和原门控、EMA、增强及优化参数。

实际历史来源代码为 experiments/cvs_legacy_no_sat 与 cvs_phase1_stack。已核对原配方：EMA=0.999、batch_neighbor 时间门控、窗口 2、最低置信度 0.8、强视图一致，rx/day quantile=0.86，lambda_u=0.16、lambda_ent=0.01、label_smoothing=0.01，L batch128/U batch256，AdamW lr0.0002、weight_decay0.0001、clip5、cosine 最低 0.000001。训练不使用卫星拼接或卫星一致性。U_s TX 继续隐藏，不改既有 L_s/U_s/V 的物理角色；冻结最后第 50 轮，逐 row 自动 clean 预测与独立 truth-last 评分。原用户 NoExplicitCFO 身份输入约束继续保留，对全部对照一致应用，不删内部逐包补偿。

特征结构改造仅位于 t1/t2/t3 与 pa_b1/pa_b2/pa_b3。时序块包含 kernel/dilation=(3,1)/(5,1)/(5,2) 三个深度卷积分支；PA 块包含 kernel3/5/7，逐层原 dilation1/2/4。每块增加可学习 1×1 残差映射，保留原 norm、activation、pool、dropout 与输出通道，最终保持 160 维和原 ResidualPhysicalHead cosine6×160/scale30/融合接口。classifier 参数仍按原训练损失正常学习，结构、功能和初始化不改。

本地已验证真实四种模型：classifier 与对应种子的原基线初值及相同 embedding 分类输出逐项一致，所有新卷积分支实际执行且收到有限非零 CE 梯度，特征输出发生变化，全部训练参数纳入优化器，strict state round-trip 与 epoch51 负例通过。可训练参数为 177025 / 237697 / 291585 / 352257；性能优先，真实时间及额外成本另列，尚未声称改造有效。

此前未登记、未发布的 cosine 梯度诊断与自身学生 SSL 特征草案仅保存在工作树 .codex_tmp，按当前用户方向暂缓；没有远端进程、权重或目标访问。新历史配方四组的训练、预测、独立评分和队列已实现并完成本地闭环检查及预登记。独立审查发现并修正了 15 域训练权重与 5 域预测构造不一致的问题；定向复审 PASS，已发布并核实启动。

按最小 CVS 流程完成剩余闭环后发布；不要求额外设计审批、hash/receipt 或重复审查。目标数据不得用于配方、参数或权重选择，不因目标低分停止或选择性重跑。健康旧任务不停止、不重启、不热改，每 GPU 最多 3 个总进程，包括其他 owner。

用户再次明确（2026-10-13）：每次优化探索的实现和对应实验结果都上传 Git。每轮代码、配置、预登记与逐 seed/配对/失败缺项的结果报告提交并推送，独立核对 remote branch OID 与 HEAD；checkpoint、原始预测、完整日志和数据留在原实验目录，用报告路径定位。当前全流程本地闭环与依赖检查已通过，旧配方四组已预登记；首种子的四组均完成 11100 次更新、冻结且各自末 222 次训练 L_s CE 均值 ≤0.5 后，自动扩展任意组的另两 seed，无 V/target 门槛。原生 15 域模型至预测模型的严格加载、四组首种子门槛与分类头不变检查均通过，独立定向 P1 复审 PASS，已发布并核实启动。

当前运行：`20261013-phase1-feature-multiscale-oldgate-cos-e50-clean-manysig-m12-r01`，release `cvs_feature_multiscale50_20261013_r01`，执行 commit `5fa536f9ca2d90e7d30d547d58503a41b612d57f`。首种子四组已到第 39/41/41/40 轮并均进入 pseudo 阶段，实际 EMA 更新与无标签损失执行、U_s 真值隐藏、无非有限跳步 VERIFIED，无失败记录。首种子四组全部 source-only 通过后队列自动扩展，逐行自动 clean prediction 和独立评分；完整结果报告与实际种子数随后提交推送。目标仍 active。

后续扩展已核实：首种子四组完整闭环，终轮源训练 CE 0.288426/0.249913/0.196179/0.193749 全部通过既定条件，剩余 8 行已自动启动并到第 4 至 6 轮，无失败。当前实际完整种子数仅 1，尚无三种子完整均值；未读取目标分数参与扩展。仓库保留 `first_seed_source_summary.json`，全矩阵结果完成后仍须审计和上传。

2026-10-13 完整更新：历史门控E50多尺度12模型全部完成与完整审计 VERIFIED，clean均值 OLD/TIME/PA/JOINT=80.526786%/81.690476%/79.003175%/81.289484%，主目标84%未达。全部seed和负结果已保存于该run final_results.md/result_summary.json/CSV，须推送核对。旧run实际代码5fa536f9c，不追溯修改。

用户新六路线说明已存 docs/design_reports/phase1_feature_six_routes_20261013.md。后续改为归一化信息恢复与分支竞争首轮，BASE/同参数普通残差/NORM/COMP/NORM_COMP；NORM仅t1浅层，恢复当前样本GN前相对mu/std，alpha初值.05有界.2；两种旁路均新增1225 identity参数。COMP p=.1、5epoch warmup后L_s学生随机撤去4种实际证据路径之一，处理频域rho/PA统计侧路及独立reference_projection，U/EMA生成方式和推理全部路径保持。当前只有本地原型和无query检查PASS，冻结骨干普通cos refit与source/RX诊断、实际全流程矩阵及Phase2同support旧新类比较仍待实现/验证，不能称机制有效。四条后续路线分阶段保留，编码率先做几何诊断。新参数在读取现多尺度目标值前已固定，不依据该目标结果调整。

2026-10-13 新路线正式闭环已登记：20261013-phase1-restitution-competition-oldgate-e50-clean-manysig-m15-r01。五条件BASE_REF/ORDINARY/RESTORE/COMP/RESTORE_COMP固定，12新增scratch50＋3同seed历史基线复用。L_s-only冻结普通cos10轮及源RX/部分路径诊断均已实现，成本独立披露；source-only首seed完整屏障、不劣界限及双readout truth-last checks PASS，唯一独立P0/P1 review PASS。下一步提交发布与真实checkpoint smoke，不把代码检查称为识别提升；Phase2仍未验证，84%目标保持active。

# SARD 与无星地增强 F50：固定 DURA v1 三种子七景测试

状态：PARTIAL（技术失败，NoAs三行完整）；代码／配置 commit `13177dbb10b7b6515d87e358e6038e81d5a1ae31`。用户要求对比 SARD 与无星地增强最佳三种子分别加入 DURA 的测试性能，本 run 只评测既有冻结权重，不训练 G、F、EMA 或分类头。

无星地增强参照固定 `20261016-phase1-F-supcon-e50-clean-manysig-m9-r02/CROSS_RX-s2026092701/02/03`，实际 clean 84.676190%、85.357738%、85.307738%，均值 85.113889% ± 样本 SD0.379881pp；原执行 commit `4df7b94d0fd8ce29056bd8236364e45afa92e07f`。它是当前同协议、原默认头、已完整三种子的无星地增强方法参照，不以某个最佳单 seed 代替三 seed 平均。SARD 固定本对话 T1 E30＋20 的 2026092701/02/03 FINAL50 学生；seed01 位于旧 E50 两路径 run，02/03 位于既有三 seed 续训组。

NoAs 从零 E50 为 33L＋17U，SARD 为 E30＋20、累计 13L＋37U；均 50 轮／11100 更新，原训练和继承成本保留，但不是完全匹配日程的纯 As／KL 因果消融。默认分类头随各自骨干原样联合训练，本次冻结直接读出，不另训头。

DURA v1.0 固定 R69 stage2、266951 参数、E20／4940 原 checkpoint，源训练 TX 无标签／F-free；执行 commit `47c264286ca689f060a3f36aafb763fe209a6f06`、FP32／TF32off、reuse=True／graph=True／native=False。本轮复用 `20261021-phase1-restoration-stage2-graph-s2026092701-r01/stage2_residual_graph/target_confirmation/native_map_prior/prediction` 原七景恢复 IQ，不生成新 G 权重、求解预算或精度版本，也不再执行 G。恢复结果仅由当前 IQ 和同一冻结 G 决定，没有 F 或 TX 真值参与；因此可供两个 F 家族及三个 seed 共同读出。本次不声称新的 G+F 时延结果。

完整矩阵唯一维护于 experiment.json：两种 F × 三模型 seed × 原输入／DURA 输出 × clean、practical_high、practical_mid、practical_low_suburban、practical_high_urban、practical_mid_urban、practical_low_urban。每单元沿原 168000 个物理样本／六 TX 和固定 received IQ，split392005，L6300／U56700隐藏／V27000、RX/day/物理角色不变；既有 VALIDATED_ONCE 数据复用。DURA 原 physical_ids 与 T1 原 opaque IDs 在读出前逐项对应，正式观察和恢复 IQ 均不重采样。恢复器也保持原训练数据完整来源。

已有 NoAs seed01 的原输入七景及 DURA 七景、SARD 三 seed 原输入七景、NoAs02／03 原 clean 复用，共37单元；只新增35个DURA-F读出和12个NoAs原practical读出，共47单元／7896000个新F目标读出。每行实际 checkpoint 首步2个合法 L_s 无 query smoke，6行总12个source前向另计。SARD 使用原完整 own_model 来源核对，再严格导出原 encoder 和默认头，以实际 L_s 的 native 默认头逐元素近似等价检查；NoAs 复用严格 F50 metadata/payload/roles/head loader。复用预测不重复模型测试，独立 scorer 在每行全部14单元固定后统一复算指标并作净纠错分析。

每行全部七景原输入／恢复后预测固定，独立 CPU scorer 才读取 truth；输出逐 seed Accuracy、Macro-F1、RX/day/TX、repaired／damaged／net、各组 mean／sampleSD(ddof1)、同 seed DURA 增量及两 F 家族差。三个 seed 只改变 F；G 固定一个 seed2026092701，不声称恢复器跨 seed 稳健性。测试基准和 NoAs 历史排名已暴露，属于用户指定的工程比较，结果不反馈重训、选模、参数或选择性重跑。

环境／CWD：N607 原普通账户 CVS-RFFI Python3.10／Torch2.1、CPU2、完整 FP32／TF32off；`PROJECT/releases/cvs_SARD_NoAs_DURA_E50_m6_20261021_r01`。命令 `CVS-RFFI/python -u experiments/cvs_sard_dura_eval/runner.py`。8GPU按所有 owner 总进程数最少、空闲显存最多排序，空闲≥12000MB、每卡总进程≤3、本run每卡1worker、最多2张卡并行；独立 CPU scorer 最多1。不可覆盖 run root `PROJECT/runs/20261021-phase1-SARD-NoAs-DURA-e50-m6-r01`，唯一 owner codex/root/SARD-DURA，其他 owner 健康任务保持。

3项矩阵／非法seed视图／禁止训练与native回退负测及6文件AST PASS；一次下一真实启动 P0/P1 审查无阻塞。单次普通账户 N607 preflight PASS。Git commit 是代码身份；只对一份发布归档作一次本地／远端 SHA 比较与一次远端编译。launcher 首步实际 checkpoint source-only no-query smoke，PASS即读出，不新增审批或数据重验证。

停止规则仅限自有模型来源、物理角色、种子、默认头、非有限、输出碰撞、无法启动或无法产生合法预测的系统技术失败；停止未启动行，保留已完成和 partial 产物及其他健康任务，不自动重试，不因科学分数低停止。预期 launch／workers／queue、每行smoke、14单元预测及引用、独立scoring、六行complete、results和group complete；缺任一行或场景时保持PARTIAL，不把训练或单seed结果当三seed完成。

发布后独立核验 VERIFIED：归档单次传输比对、6文件远端编译通过；原NoAs seed01完整复用并完成配对评分。NoAs02/03分别在物理GPU7/5启动，实际PID/CWD/完整参数、同seed真实checkpoint source-Ls smoke均已核实，日志已产生七景进度，原输入clean按历史结果复用。其他owner健康任务保持，SARD三行由同一固定队列随后执行。

技术失败独立核验：r01的NoAs三行原输入/DURA七景均完成；SARD01首步source-Ls无query smoke的native对照在原CenteredSinc读取中间张量_version时失败（Inference tensors do not track version counter），SARD目标预测0，02/03未派发。原所有所属PID均已自然退出，无终止其他任务。完整产物保留，不原地重启。仅将native源检查包入inference_mode(False)+no_grad；正式F读出不变。本地回归先复现同一异常，修复后5项检查PASS，一次限定问题复核无遗留P0/P1。新r02复用本run的NoAs三行，只补SARD三行。

# SIG_TIME＋RDrop：A–F 配对验证

当前状态：RUNNING。五个首种子按固定 50 轮训练，已核实进入半监督阶段；新目标汇总尚未读取，B、E、F 的性能结论未形成。用户 2026-10-15 的固定规范为本轮依据。唯一逐行配置、数据与六类 seed 登记位于 [experiment.json](experiment.json)，实现追踪位于 [验收记录](../../../analysis/SIGTIME_RDROP_AF_20261015_traceability.md)。

A＝单路＋RAW 标准；B＝三路＋RAW 标准；C＝单路＋RAW 单位修正；D＝单路＋原 CENTER_25K；E＝单路＋统一居中和单位修正；F＝三路＋统一方案。六者都使用原 RDrop alpha＝1、普通增强、LS＝0.01、旧 batch_neighbor 门控、ownEMA 和 cosine，默认 6×160 分类头随骨干联合训练。两个随机 L 前向复用同一次实际增强后的 IQ。训练不用卫星增强，不拟合任何诊断头。

三路直接复用既有 SIG_TIME_X3 的三个独立局部投影与 SIG_TIME 描述子，共同产生同一个 joint160。单路特征参数 195,041，三路 232,033；默认分类权重均为 960。没有 MEAN_VAR/COV 分支或模型投票。

统一方案 E/F 使用 p＝p0＋25,000q，Fs＝25,000,000 Hz，q 初值 0；实际 AdamW 步前按原 cosine 的 alpha 设置 lr_q＝alpha Fs/s、eps_q＝epsilon s/Fs。原全局 clip5 前只将 Sinc 梯度除以 s，和全部普通参数共同裁剪后再乘回 s。坐标组 AdamW weight_decay＝0，实际自适应步前执行 q−＝alpha lambda(q＋p0/s)。D 保持原坐标 AdamW 和坐标梯度裁剪；因此 E−D 同时涉及单位、裁剪度量、物理衰减，不能只归因于学习率。C/E 在精确算术下针对同一物理优化目标；差异可能主要来自浮点表示。历史涨幅不相加。

A 复用原 RDrop 完整三种子 fixed-final50 结果与预测，发布第一步核实实际 recipe、物理角色、空祖先、默认头、11100 步与原模型代码 b3ac442e。01 来自原 m18，02/03 来自保留的技术恢复 m4，不能伪称同一物理运行。登记检索未找到已完成或健康的匹配 C＋RDrop；此前 C 仅为未发布准备，没有重复训练。

每个新增模型从零 50 轮＝33L＋17U、每轮 222 步、共 11,100 步。固定模型 seed 2026092701/02/03。原物理划分保持：L_s 6,300、U_s 56,700 且 TX 真值隐藏、V 27,000 仅健康诊断。新训练最多 15 个模型、750 模型轮、166,500 步；A 的历史 150 模型轮与 33,300 步单列。首批仅 BCDEF 的 01 五个模型，共 250 模型轮、55,500 步。

首种子五路全部完成源训练、默认头冻结及实际特征证据后，固定源规则检查：final L CE≤0.5、预算完整、正 KL 调用≥1%、部署特征有变化、导出与去 W 检查精确；C/E/F 还须至少 12/24 cutoff 和 bandwidth 改变、student/EMA 核相对变化及 Sinc 对 joint160 的反事实影响均≥1e−6；E/F 必须完成 11,100 次物理全局裁剪。全部通过才扩展原样 02/03；失败保留首种子并不重调。无论通过或失败，都必须先把源决定提交推送并核对远端 OID，再读取任何本轮新目标汇总。逐行冻结后仍立即自动 clean 预测、独立 CPU truth-last 评分，评分结果不反馈训练。

测试每行固定 168,000 个 clean 物理包＝7 RX×4 day×6 TX；每 RX 24,000、每 day 42,000、每 TX 28,000。完整矩阵 18 行、3,024,000 个逻辑决策与 324 个分层指标，其中 A 的 504,000 决策复用。报告 B−A、E−C、E−D、F−E、F−B 和 F−E−B＋A 的逐 seed 差、均值、样本 SD 与正差数，同时保留 Macro-F1、最差 RX/day、混淆矩阵、修正/引入错误及模型参数和时间成本。保留全部负结果。

发布命令：CVS-RFFI/bin/python -u experiments/cvs_rdrop_sinc_units/runner.py experiments/cvs_rdrop_sinc_units/configs/matrix.json。CWD＝PROJECT/releases/cvs_SIGTIME_RDrop_AF50_20261015_r02；不可变输出＝PROJECT/runs/20261015-phase1-SIGTIME-RDrop-AF-e50-clean-manysig-m18-r02；CPU dispatcher/scorer 的 CUDA_VISIBLE_DEVICES 为空，worker 每个 CPU2、loader0。GPU 由共享容量锁选择，含其他 owner 和启动中进程，每卡总计≤3、剩余显存≥12,000 MiB；保留健康任务。代码与配置身份以本报告所在提交中的实际代码及远端 release_commit.txt、resolved_config.commit 为准。

技术停止规则：错误物理角色/来源、非有限更新、预算或预测不完整、worker/scorer 失败时，保留产物、停止新增 pending，已启动健康行正常完成，不自动重试、不终止其他 owner。输出冲突先核实，不覆盖或重复发布。预期产物包括 scratch initial、final student/ownEMA、source freeze、完整 epoch/step 日志、实际 Sinc 梯度与更新记录、部署特征证据、逐行预测和独立评分、源决定及最终配对汇总；详情见唯一登记。

验证：本机 CPU 与 CUDA 均通过六条件实际网络检查、21 个协议负例、相同增强 IQ 的 RDrop 路径、默认头联合更新、student/EMA/去 W 导出一致性、真实叶版本缓存重算。三个 origin×各 100 步 float64 的统一更新最大物理误差 2.2352e−8 Hz；这只是实现检查。CPU C/E 首步普通参数最大差 0，CUDA 为 3.5763e−7。BCDEF 的完整 feature_check.measure 均通过恢复与导出检查。通过/失败源决定上传顺序共 8 例通过。唯一独立 P0/P1 审查发现两项直接缺陷，已分别修复并通过定向复核。

边界：这是已暴露的 ManySig clean 基准上的预登记组合实验，不是新的独立盲测或新 TX/Phase2/卫星部署证据。本轮 B、E、F 是否有效，必须等待实际固定矩阵证据。

技术恢复：r01 在第一步 A 元数据预检失败，未启动任何新增模型、无 GPU worker、无新训练步或新目标预测。只读检查证明 A02/03 的科学字段与原配方完全相同，差异仅在已保留技术恢复的 run_id/output_dir。修复预期身份与已核实物理根对齐，全部科学字段检查保留；定向独立 P1 复核通过。r01 不覆盖、不重启，r02 使用全新不可变根，方法/数据/预算/seed/门槛完全相同。

实际发布核实：r02 执行提交 09554c3a06c976c53efc4dbc61f5c501a1e7b06e 与远端分支一致，单归档 SHA/远端编译 VERIFIED，三个 A 实际来源及一个真实源 checkpoint 无 query smoke PASS。BCDEF 五个原 worker 的 PID/CWD/argv/GPU、实际 GPU app 与日志增长均独立核实；最新完整 epoch 为13/14/14/14/14，failure 为空。普通增强、LS.01、scratch33L＋17U、clip5、EMA版本化更新实际生效；两个前向使用同一实际增强输入。首轮13批×2个 Sinc 张量×5模型＝130个保存张量的更新已独立复算，物理公式重建最大差0.548Hz，包含明确的 FP32 舍入；不是性能结果。原生梯度范数遥测保持实际叶坐标，E/F 的裁剪本身使用物理Hz，不能直接把原生q范数与RAW范数比较。源决定及新目标汇总均未读取／生成决定，02/03 尚未放行。

半监督切换核实：五个首种子模型的 E33/E34 共2,220条原生 step 记录已重计，均为 E33 U＝0、E34 U＝56,700。B/C/D/E/F 接收的 U 分别为896/879/878/876/896，U 真值可用标记与非有限跳步均为0。旧 batch_neighbor／强图一致门控、ownEMA0.999、lambda_U0.16、lambda_ent0.01及默认头联合训练保持；不根据这组源指标调整候选或权重。实际 cosine 的 E33/E34 alpha 分别为0.00005447744012103691和0.00004891993945310276。熵系数来自实际配置，没有将其当成独立测得的熵损失。新目标汇总仍未读取。

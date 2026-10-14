# 固定 SIG_TIME 的 Phase1 urban 星地增强

本轮按用户最新要求，从头 E30；对比基线为 SIG_TIME＋旧门控＋cos＋普通拼接星地增强。训练环境仅为 practical_low_suburban、practical_mid_urban、practical_low_urban。主验收目标是 low urban 准确率至少 60%；urban +3 个百分点、low urban +5 个百分点、clean 下降不超过 0.5 个百分点是相对 B0 的探索目标，不是结果保证，也不参与训练、扩种子或重跑选择。

## 代码审计与边界

起点 Git：7fceadc5020c2970ae31b7502942f0453abff88f。实际训练器与 practical 实现来自 experiments/adv3b02_xuc/code/。运行时保存 train_ssdg、leo_practical.channel 和 leo_practical.residual 的 __file__。固定单 SIG_TIME：t2 Dropout 后的 6 维局部投影、4 个时间窗、112 维描述子、joint160；特征参数 195041，原默认 ResidualPhysicalHead 6×160 cosine 头、scale=30 联合训练。参考响应和已有增益不增加、不替换。关闭额外拟合分类头。

ManySig 实际 equalized=1；25 MHz，256 点，native center crop 和逐包 RMS 归一化。星地增强再按原 practical 输入归一化，通过原 residual 引擎得到 IQ，最后由原 AGC 处理，不追加输出 RMS 归一化。录音的 equalized=1 不代表新增 practical 信道被均衡。本轮 residual/post_sync、EQ=false，因此 G=1；reference_snr_db=30，质量适配、相位跟踪和注入 RX IQ 补偿均启用，unlocked 状态由原引擎关闭精细跟踪，边界 reflect，数值 FIR 延迟 24 点。新增噪声 SNR 不是录音总 SNR，不能反演未知 TX 硬件效应。

合法源角色保持 L/U/V=6300/56700/27000，source RX=1/3/4/6/8，day=1/2/3，split=392005；native 逐物理角色 ID 与现有 source_contract 精确匹配，U TX 隐藏。源 V 仅健康监测，不梯度更新。RX 硬件按 RX/day/eq session 稳定，session 不含 TX；命名随机流区分训练 epoch/step/view、固定源验证与诊断。所有合法有标签 satellite IQ 都参与 CE，无学生置信度筛选。

已完成的源诊断参考是同一单 SIG_TIME 的固定 final50 checkpoint（源 run 20261014-phase1-moment-area-oldgate-cos-e50-clean-manysig-s01-m4-r01，实际执行提交 512c10a267b5e654cfd2c983ab1debdeebc7f626）。其 scratch、11100 更新、source-only、默认头来源已读回；真实 strict-load smoke 和源域检查由 launcher 第一阶段执行。它只用于冻结诊断，不作为本轮初始化。新模型与自己的 EMA 均从零开始，父训练成本为 0，新增/累计为 E30。SIG_TIME 是用户指定的已暴露 benchmark 起点，本轮不声称新的独立盲测。

## 可执行矩阵与归因

矩阵和每 row 配置唯一维护于 experiment.json。最多登记 18 row，不能当作 18 个完成结果。首 seed=2026092701，先运行以下 5 臂，随后仅按源证据冻结条件原样扩展 02/03；C4 须源标签代理 precision≥.95、coverage≥.1 后独立从零运行。

| 臂 | 唯一差异 |
|---|---|
| C0 | clean 归因控制，原旧门控＋EMA＋cos |
| B0 | 用户指定对比基线，clean/satellite 普通拼接一次前向，三环境等先验 |
| C1 | 同样 IQ 信道 draw，双视图分别前向，clean/satellite CE 各自 mean 后 1:1；无额外一致性 |
| C2 | C1＋有界源域状态课程，eta≤.5；每个已登记场景和状态仍有 nominal 支持 |
| C3 | C2＋单侧默认头间隔，rho=.5、m_max=1、固定 scale=30；clean 正确参考 stop-gradient，q 连续且只调额外约束 |
| C4 | C3＋原 EMA clean 软标签→同一 U 的 mid/low urban 视图；不要求 urban 学生先高置信 |

B0 拼接 CE 和 C1 分开 mean 的 CE 在相同 logits 下数学等价；二者执行与 Dropout 随机实现不同，不能把归一化本身宣称为创新。旧半监督和 EMA 在所有臂中保留为共同训练配方：20 轮 label、10 轮 pseudo，原弱/强门控不变，lambda_u=.16、entropy=.01、label_smoothing=.01；C4 是独立新增 U urban 软监督。所有网络与默认头正常联合更新。

每臂 30×222=6660 更新，L batch128，U batch256（尾批按原 loader）。AdamW lr=.0002，weight_decay=.0001，min_lr=.000001，原 cosine 压缩至 E30，固定 final30。记录实际更新、各类前向样本量、源验证成本、墙钟、完整损失曲线、门控和梯度变化；固定预算不能当作收敛证据。

源诊断固定 720 条 TX×RX×day 平衡 L 样本和同一命名流，比较完整 residual、移除新增噪声、平坦能量匹配 H、移除 carrier 残差、移除注入 RX IQ；分别记录重算接收状态的总效应和固定接收状态/频率/相位/EQ/AGC 的受控效应。另比较白输入代理与实际波形功率、reflect/edge 的人工边界及 FIR 有效区域。仅源 IQ，原默认头，无参数更新或拟合诊断头。

源扩种条件预先固定于 dispatch.py：各 row exact6660、合法角色、真实 labeled satellite 覆盖；候选源 L panel clean 比 B0 下降不超过 .02、hard margin 比早期三轮均值下降不超过 .01、final L CE<2。它们是早期可执行性/学习健康条件，不是目标效果证明。source_qualification.json 冻结在汇总目标成绩之前，目标不影响晋级或选择性重跑。

## 正式评价、命令与产物

复用 20261002-phase1-cvs-residual-sixscene-manysig-m8-r01/received_views 的 VALIDATED_ONCE 固定数组和现有 clean capsule：每场景 168000 条，7 个未见 RX、4 day、6 TX。保持完整 clean＋六 practical，保留全部困难样本，不调测试信道、SNR、状态或归一化。每 row 训练冻结后立即自动预测全部 7 视图；独立 exec scorer 完成后才打开 truth，并独立复算 accuracy/Macro-F1/confusion。报告逐 seed、urban 均值、low/mid、最差 urban RX、clean/suburban，以及全部负结果和缺项。

远端 CWD 为 PROJECT/releases/cvs_sigtime_urban_paired_20261014_r01；环境为 N607 既有 CVS-RFFI Python。精确主命令：`python -u -m experiments.cvs_urban_sigtime.dispatch`，worker 为同模块 `--mode worker --row <row>`，随后同 PID exec 到 predict/score。唯一 owner 为 codex/root/urban-sigtime-20261014，每 GPU 包含其他 owner 最多 3 个实验进程；不修改或重启其他健康任务。

代码/config 的实际 Git commit 由 release_commit.txt 固定，并写入远端 experiment.json 与运行证据；不要求设计、报告或单成员 hash。一次 release 归档、一次本机/远端 archive SHA 对比、一次远端 compile。launcher 第一步真实 checkpoint 无 query smoke，PASS 后直接源诊断和训练；不存在 smoke token 或额外审批。

技术停止仅限协议/角色/预算错误、输出冲突、非有限、无法执行或无法产生合法 prediction。保留全部产物，停止 pending 派发并让其他健康 row 收尾；低性能不技术停止，不自动重启。启动后核对 PID/CWD/argv/GPU/log 增长。

期望产物：source_diagnostic/results.json、各 row 的 final_ssdg.pth、完整 step_metrics、source_learning_progress、training_coverage、gradient_gate_history、七视图 predictions、独立 scores、summary、resources、analysis 和分层图/训练曲线。权重、原始日志、逐样本预测留在既有实验存储，不进入 Git。

## 已执行验证

本地 `python -m experiments.cvs_urban_sigtime.checks`：9 项通过。包括 margin stop-gradient/scale、不筛除真实标签 CE、旧头/实际特征数值更新、隐标签元数据/session、跨 epoch 刷新、原 residual 全数值对齐、EQ 噪声着色、固定/重算状态控制、预算/目标配置负测、真实普通拼接及 U urban 约束。修改模块和发布脚本语法编译通过。

一次独立 P0/P1 审查指出 baseline_reuse 缺字段和 C4 U 混入 suburban，已修复；一次对应问题定点复核确认无遗留。没有额外全量审查、文档 gate、hash 链或目标成绩选模。真实实验结果尚未返回，当前不能宣称任何 urban 提升、clean 代价或跨 seed 重现。

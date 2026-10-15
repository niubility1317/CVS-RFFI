# RDrop 全源 E20 / E50 三种子预算比较

状态 **RUNNING / 全12行已实际启动，发布后独立证据 VERIFIED**。E20与E50各6个从零模型，所有12个own初始checkpoint无query smoke通过、首epoch完成、原全五RX6300/56700/27000角色一致、实际预算/默认头配方一致、CWD/argv/GPU身份及日志持续增长均已独立核实。没有技术失败，训练和测试尚未完成，不预报收益。

执行commit `5530a17b60bc1005eef8de151ae540b92a97da8a` 已推送且独立remote OID一致。一次ordinary N607 preflight、release archive SHA比较和remotecompile通过。前两次本地发布尝试在任何远程preflight/写入前因Git工作区旧stat状态退出；对应共享文件无内容差异，刷新index元数据后恢复，未改变/重训实验或生产代码。

本任务GPU0至7的进程数依次为[1, 1, 1, 1, 2, 2, 2, 2]；该快照计入其他owner后各卡均2个实际GPU进程，总16个，≤3上限。自己12个worker实测每个3348至3348MiB（高于前次源fold计划约3GiB，但仍有充分准入余量）。CPU2/worker、loader0、CPUdispatcher/scorerCUDA为空。首epoch含源V诊断实测40.04至47.84s；它不是纯训练step计时或最终每epoch稳定耗时。所有8卡均使用共享实时准入，其他科学任务正常保留，不独占剩余容量、不停止/迁移/重启任何健康任务。

12行按固定20/50轮预算继续，逐行冻结后立即自动clean168000预测，随后独立CPUtruth-last评分；不等待其他行、不读当前分数调训练。完整12行/216strata完成后比较4组的三seed均值、样本SD及预登记机制/预算配对差。当前goal active，84%尚未证明。

发布前冻结方案：

状态 **PLANNED，已登记，尚未远程发布**。本轮按用户最新要求，同时运行 E20 和 E50；每档 TWO_CE / RDROP × seeds2026092701/02/03，共12个独立从零模型。E50_RDROP 为预登记主结果，E20_RDROP 为预算比较，两个同预算 TWO_CE 提供机制对照。两档都报告是否达到 clean 三种子均值84%，不以 target 最高分挑选主结果。当前目标未完成。

全五源 RX1/3/4/6/8；原 L6300/U56700 hiddenTX/V27000，源日1/2/3、六TX、物理划分与 VALIDATED_ONCE 载体保持。L 为 train pool 的10%及全部90000源数据的7%，不扩大标签。U原门控、Gaussian weak/strong、ownEMA0.999及权重lambda_u0.16/lambda_ent0.01保持，V只作默认头健康诊断。SIG_TIME/CENTER25K、原生dropout、NoExplicitCFO与原补偿、实际joint160、默认cosine6×160 scale30按原流程联合训练；不另训/替换/重拟合任何分类头。无父模型/优化器/EMA权重继承，无星地训练或测试。

| 预算 | 机制 | 种子数 | L/U轮次 | 每模型更新 | 每模型额外L前向 |
|---|---|---|---|---|---|
| E20 | TWO_CE / RDROP | 各3 | 13 / 7 | 4440 | 568320 |
| E50 | TWO_CE / RDROP | 各3 | 33 / 17 | 11100 | 1420800 |

两机制均对同一个 clean L128 做两次原生随机dropout前向，用 mean(CE1,CE2)，LS0。RDROP 固定加 alpha1 对称KL：0.5 mean sum((p−q)(logp−logq))，两侧均可微；TWO_CE alpha0。native 原有 CE1，hook只加 .5(CE2−CE1)+alphaKL，避免双计数，单次原优化器更新。全部步骤启用，不扫描系数。原日志CE1与实际pairedCE/KL分开。所有变体同seed200个初始state完全相同，两个机制均承担第二次L前向成本。RDrop 是[已有方法](https://proceedings.neurips.cc/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html)，本轮验证其CVS训练机制收益，不宣称全球新方法。

E20与E50使用原AdamW lr0.0002/weightdecay0.0001，原cosine日程终点分别为20/50。预算对比同时改变总更新、L/U阶段轮次和学习率日程；不把它解释为同一20轮轨迹继续训练。总成本12模型/420modelepoch/93240实际更新/11934720额外L packet前向，主L前向同量，8164800次U重复观察；2016000次新clean决策/216个分层指标。训练启动initialcheckpoint合成smoke和预测前合成smoke另计，不混入真实训练量。

开展依据是固定单seed真实RX1留出源训练：RDROP75.158730% vs TWO_CE69.285714%，+5.873016个百分点，纠正104/引入30、meanCE也下降；fitV微降、TX1新增5错，收益不均匀。它属于来源合法的科学证据，四RX权重不继承，也不能充当这次全五RX的E20对照。原计划尚未发布的seed02源复核adapter已撤回，未登记/未启动；用户当前完整E50+E20要求覆盖其等待条件。已有NONE小幅正收益、CENTER25K等路线和所有负结果保留；本轮不混合其他机制或跨拟合教师。

各row固定final20/50 student默认头，不按V择轮/挑EMA/权重平均/重训头。每row冻结后立即预测原clean168000物理样本，再由CUDA为空的独立CPU进程读取truth评分；不等整批训练完成。保留overall/RX7/TX6/day4共18strata。四个预登记配对：E20 RD−TWOCE、E50 RD−TWOCE、TWOCE E50−E20、RD E50−E20；逐seed正确/新增错误及三seed均值/样本SD、MacroF1、最差RX/day，完整source训练/健康与成本一并报告。既有benchmark已暴露，不声称新独立盲测。任何分数均不回流其他训练或选择性重跑。

ordinary N607/CVS-RFFI；CWD `PROJECT/releases/cvs_SIGTIME_rdrop_budget20_50_20261015_r01`；命令 `CVS-RFFI/bin/python -u experiments/cvs_rdrop_budget/runner.py experiments/cvs_rdrop_budget/configs/matrix.json`；owner `codex/root/SIGTIME-rdrop-budget20-50-20261015`；唯一不可覆盖root `PROJECT/runs/20261015-phase1-SIGTIME-rdrop-budget-e20e50-clean-manysig-m12-r01`。CPU dispatcher和scorer不使用CUDA，每训练worker CPU2、loaderworkers0。12行均已固定就绪；按空闲优先考虑全部8卡，在既有shared `receiver_residual_capacity4.lock` 内读实际与pre-CUDA全部owner、选卡/Popen/记录，每GPU总进程≤3且准入空余≥12000MiB。每row计划约3GiB（前次源fold实测2.95至2.98GiB），实际全源占用和耗时另测。协调只分配资源，不干预其他对话科学路线；不停止/迁移/重启任何健康任务。

技术停止条件：权限/物理角色/继承来源不匹配、非有限损失/输出、实际预算不足、预测不全、输出碰撞或worker/scorer失败即停止待启动行，保留产物；已启动健康行继续，不自动重跑。源健康报告，不因分数改变固定12行。每launcher首步自己的scratch checkpoint保存/重载、2个合成无query前向smoke PASS即继续。发布前唯一聚焦负例/模型检查和独立P0/P1、actualGitcommit；一次N607 preflight、单archiveSHA比较、remotecompile；之后独立PID/CWD/argv/GPU/log增长证据。预期产物和每行配置/命令/seed见experiment.json。

REJECTED_EXTRA_GATE：单独设计审批/spec链、report-only重复审查、额外config/source/report哈希或新调度平台不加入最小流程。84%及可部署泛化目标仍待完整证据。

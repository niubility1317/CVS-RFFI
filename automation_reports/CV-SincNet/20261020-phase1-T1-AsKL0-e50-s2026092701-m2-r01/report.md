# T1 去除 As 独立 KL：两条 E50 路径

状态：ANALYZED，两组累计E50及全部七环境测试和独立评分均已完成。科学实现 commit `ed652af1f7b427cb7cff293e6b6185ca76a32bec`；当前并行调度 commit `971fd16ac342f3523c4e3112f437e678f32ca459`。

2026-10-20 用户指定原 T1 E30 拉长到 E50、model seed `2026092701`，并明确要求 E30 续训及从零完整 E50 两种都试。逐行矩阵只维护于 experiment.json。

续训使用原 `20261019-phase1-FcrossRX-AsTemperature-e30-s2026092701-m2-r02/T1-s2026092701/source` 的 FINAL30 学生及自身 EMA，追加 20 轮，重置 AdamW、cos20 和 temporal cache，累计 13L＋37U。原前 30 轮成本继承，新增 4440 次更新。另一条从相同模型 seed 随机初始化，50 轮＝33L＋17U、cos50，新增 11100 次更新。总新增 70 模型轮、15540 次更新；两条各累计 50 轮。阶段和学习率日程不同，不能把两者差异归因于单一轮数因素。

保留原 T1 native scale30 的 As CE、As 独立 KL=0、ordinary/raw RDrop KL、原 six-L 前向、LS0.01、ORD128 CROSS_RX、旧 U 门控与自身 EMA、默认头联合训练。保留原 full practical As 训练增强（不是本对话 F clean-only 增强替代配方），不接 R17、KD 或温度 T6。原 As 生成器硬限 E20/E30，现仅延长合法累计边界至 E50／11100，并按累计 epoch/step 使用原 namespace、seed、scene、硬件 session 与信道过程，保持同一原始 L64 观测的两次 CE。

ManySig equalized1／IQ2×256／unitRMS，源 RX1/3/4/6/8、day1/2/3、全部六 TX，固定 split392005；L6300／U56700／V27000，U TX 始终隐藏，复用已验证角色。续训在读权重前复用原完整 scratch20→own10、source-only FINAL、默认头、实际 role_ids 检查，严格加载学生及 EMA；从零路径不加载训练父权重。V 只做健康诊断，固定最终轮，不按 V 或 target 选轮。

初始 clean 测试安排（已由下方用户最新七环境要求覆盖）：每行训练冻结后自动对原 168000 个固定目标 IQ／opaque ID 预测，默认头及全模型无状态更新；该行全部 clean logits 与 predictions 固定后，经独立 exec CPU scorer 才读取 truth，报告 accuracy、Macro-F1 和 RX/day/TX 分层。正式测试已经暴露，不声称新的独立盲测，不用分数回流参数、选模或重跑。

环境/CWD：N607 原普通账户 CVS-RFFI Python3.10／Torch2.1／FP32／CPU2；`PROJECT/releases/cvs_T1_AsKL0_E50_20261020_r01`。命令：`CVS-RFFI/python -u experiments/cvs_t1_e50/runner.py`。物理 GPU4 串行，每次只有一个本 run GPU worker，原有所有 compute PID 清空且 free≥12000MB 才启动；其他 owner 健康任务保持。不可覆盖输出 `PROJECT/runs/20261020-phase1-T1-AsKL0-e50-s2026092701-m2-r01`，唯一 owner `codex/root/T1-e50`。

预检 VERIFIED：普通账户 direct SSH、项目路径、GPU与磁盘。聚焦验证：4 项预算／隐藏标签权限／累计日历边界负测、原 T1 5 项损失梯度测试、12 文件 AST 通过。独立 P0/P1 审查一次，缺 loss 导入及原通道累计边界问题已定点修复并复审，无遗留 P0/P1。launcher 第一步真实原 T1 checkpoint 无 query smoke 与两路径合成 six-L 梯度／预算检查，PASS 直接排队训练；不新增 smoke 许可。

停止规则：仅本 run 的协议／角色／模型来源／默认头／非有限／输出碰撞／无法启动／无法生成合法预测等系统技术失败；保留全部 partial，取消本 run 未启动行，不自动重试。低分属于科学结果，不触发技术停止，不干预其他任务。技术失败先读回所属 PID/CWD/argv/root 和原产物，再决定后续动作。

预期产物：真实 checkpoint smoke、合成 smoke、每行完整训练/epoch/step/As 记录、学生/自身 EMA checkpoint、实际配置与源冻结证据、clean logits/predictions、独立 truth-last scores、results.json 和 complete.json。训练结束不等于实验完成；缺预测或评分时如实保留状态。

REJECTED_EXTRA_GATE：不增加设计审批、报告/配置 SHA、重复数据验证或重复全量审查。Git commit 为代码及配置身份；发布归档只做一次本地/远端 SHA 比较与一次远端编译。

## 发布读回

VERIFIED：release 归档一次本地/远端 SHA 比较一致；12 个 Python 文件远端编译通过。真实原 T1 checkpoint 无 query 2 NN、两路径完整 six-L 合成 1536 NN（含真实 As、默认头与 Sinc 梯度）均 PASS，0 optimizer 更新，直接进入队列。独立读回调度器 PID/CWD/argv 与连续更新的 queue 状态；failure 不存在。GPU4 在部署期间被其他 owner 健康任务占用，当前两行均 QUEUED，尚无本 run GPU 训练或性能结果；资源空闲后调度器自动依次启动。完整私有进程证据保留在 local_artifacts/cvs_t1_e50/latest_readback.json，不提交原始日志或机器状态。


## 用户授权的并行调度更新

2026-10-20 用户要求查看其他 GPU 余量、两组并行，并明确“找进程量最少的，直接启动”。按所有 owner 的进程总数排序，同数时优先显存空余更多的卡；free≥12000MB、总进程≤3/GPU、每卡一个本 run GPU worker、两组不同 GPU，独立 CPU scorer 锁保证最多一个。本条取代上文 GPU4 清空后串行的资源设置，科学配置、数据、预算、seed 与输出 root 保持。

VERIFIED：切换前原 run 仍为两行 pending、无 launch.json、无行输出或训练更新；核对原控制器精确 PID/CWD/argv 后只停止该空队列控制器，保留 dispatcher/queue 历史。新不可变 release `cvs_T1_AsKL0_E50_20261020_parallel_r02` 已完成一次归档比较及编译，采用原 run 已 PASS 的真实 checkpoint/six-L smoke，直接接管未启动队列，没有重复训练或重复 smoke。E30PLUS20 已分配 GPU0，实际预算20、0L20U、cos20、严格原T1 FINAL30 EMA；SCRATCH50 已分配 GPU6，实际预算50、33L17U、cos50、自身scratch EMA。两组独立PID/CWD/argv及resolved config已读回，pending为空，failure不存在。其他 owner 任务保持。


## 用户最新测试范围：clean＋六种星地环境

用户明确要求两条路径训练完成后，自动测试 clean、practical_high、practical_mid、practical_low_suburban、practical_high_urban、practical_mid_urban、practical_low_urban。此项覆盖本 run 先前 clean-only 范围；每行每视图168000个固定目标输入，共1176000条预测，原固定received IQ不重新采样。保留各场景accuracy、Macro-F1及RX/day/TX分层，全部七视图预测完整固定后才由独立CPU scorer连接truth，不以clean评分提前闭合。

训练代码仍为971fd16ac342f3523c4e3112f437e678f32ca459，测试/接续代码为ace90948e4dc62df5ec909176f2948802825a774，新不可变测试release为cvs_T1_AsKL0_E50_20261020_all7_r03。未改动内存中的训练模块、配置、模型、优化器、预算或seed；原GPU0/GPU6训练进程保持同一PID并继续写入日志。核对两行实际训练config与新测试读取config完全一致，七份固定测试输入存在，已有prediction/scoring均尚不存在。

为使原健康worker在训练后实际进入新范围，仅把其后续predict/score执行入口接至新release，原runner备份及训练release_commit保持；在run/evaluation_handoff.json记录训练与测试两个commit、入口及七视图。只替换无训练功能的控制器为新观察器，不停止任何训练worker。新观察器只读取所属行进度并收集七视图结果，不再调度训练；每行源冻结后由原worker自动exec新测试入口，七视图预测后再exec独立CPU评分，CPU scorer锁保证最多一个。原初始clean-only训练配置说明文字不追溯改写，实际测试范围由本节及evaluation_handoff指定。

VERIFIED：发布归档一次比较及远端编译通过；两项七视图完整性/提前truth负测通过，实际N607/Linux predict与score入口转发及原入口fallback检查通过，0模型/目标IQ/truth访问。独立读回新控制器PID/CWD/argv、queue.evaluation_views七项、handoff目标及两个原训练PID，failure不存在。最新可见源进展为续训累计E32、scratch E3，训练日志继续增长；尚无正式测试指标。


## 已完成行的测试结果（2026-10-20 11:48:28 UTC+8）

VERIFIED：E30PLUS20已完成累计E50和七环境独立truth-last评分；clean accuracy85.245833%、Macro-F184.973143%，相对原T1 E30分别+0.173810/+0.188078个百分点；六星地accuracy均值71.738988%（+0.705357个百分点），low urban52.416071%（+1.238095个百分点）。完整场景、RX/day/TX与负向分层见[test_results_partial.md](test_results_partial.md)。scratch目前E40/50仍健康训练、无最终测试分数；整个run继续RUNNING，不能声明两行完成。未改动任何健康训练或按测试调参。


## 两组最终完整比对（2026-10-20 15:34:00 UTC+8）

VERIFIED：两行均完成训练、七环境1176000条预测/行与独立truth-last评分，run已闭合，failure为空。续训clean85.245833%／从零84.838095%（从零−续训−0.407738个百分点）；六星地均值71.738988%／71.663889%（−0.075099个百分点）。从零路径未形成整体优势，分层取舍与单seed限制见[test_results_comparison.md](test_results_comparison.md)。历史partial报告保留，不再作为当前完成状态。

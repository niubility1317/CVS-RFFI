# 用户指定八个历史增强配置补齐三种子

状态：TECHNICAL_PARTIAL（VERIFIED，2026-10-19 22:38 香港时间）。队列因技术错误退出，当前无本 run 活动进程；0 条新增独立评分，0 组凑齐三种子。

用户指定 8 个配置补齐模型 seed2026092701/02/03；seed01 复用历史封存结果，仅训练 02、03。逐行矩阵以 experiment.json 为准。

|配置|本次训练（每 seed）|父模型|
|---|---|---|
|PAOFF：末十轮停普通 PA 波形注入|新增 E10|P49 相同 seed 的原 As scratch E20|
|半份 As，B01|scratch E20＋自身 E10|本轮自身 E20|
|B0：原 As 独立重跑|scratch E20＋自身 E10|本轮自身 E20|
|URAW：额外 raw-U 控制|scratch E20＋自身 E10|本轮自身 E20|
|H01：L 硬标签 CE|scratch E20＋自身 E10|本轮自身 E20|
|平滑 FIR＋As|新增 E10|本轮 B01 相同 seed 的 E20|
|ORD128|scratch E20＋自身 E10|本轮自身 E20|
|DUP192|scratch E20＋自身 E10|本轮自身 E20|

新增预算合计 400 个模型 epoch、88800 次更新。所有模型累计 E30；保留各配置原本的普通增强、As 剂量、原 U/ownEMA/门控、RDrop、SupCon、学习率重置与默认头联合训练方式，不替换成新探索默认配方。源域固定 L6300、U56700（TX 真值隐藏）、V27000，split392005，原 5 个源 RX。物理角色继承原 source_contract，无 Phase2 support/query 或 K-shot。数据协议见 docs/PROJECT_PROTOCOL.md。

资源：只用 N607 物理 GPU 4、6，每卡最多 3 个训练/预测进程，容量计入其他 owner；独立 CPU scorer 最多 1 个。不干预其他任务。空闲至少 12000 MB 才发新进程。

新种子只测 clean：每行固定最后 checkpoint，冻结原默认头，对原 168000 个物理样本生成完整预测，再由独立 CPU 进程连接 truth。记录 overall、RX、day、TX、Macro-F1、最差 RX、三种子均值和样本标准差。历史 low urban 仍仅 seed01，不作三种子星地声明。

入口：CVS-RFFI/python -u experiments/cvs_selected_seed_completion/runner.py；CWD：PROJECT/releases/cvs_F_selected_eight_E30_s0203_20261019_r01；输出：PROJECT/runs/20261019-phase1-F-selected-eight-e30-s0203-m16-r01。环境沿用各历史 immutable runtime；精确 commit 映射见配置。执行 commit 发布前写入登记并固定于 release。

已核实 P49 对应父模型为相同源角色、同 seed 的 scratch E20/4440 更新，空祖先、无 target，含自身 EMA。其余父模型由本轮从零生成；FIR 复用本轮匹配半份 As E20。实际 F 工厂 seed 和 native seed 均为对应 02/03。

首步为真实源 checkpoint 无 query 烟测（PA 两 seed，其余每配置 seed02），通过即继续队列。输入/来源/协议错误、非有限训练、输出冲突或无法产生合法 prediction 等技术失败时停止发新行，保留全部输出，让健康在途行完成。不因分数低停止、不自动重试或挑 seed。逐行训练后立即自动预测和独立 truth-last 评分。

本机 16 条配置负例和实际模型检查通过；独立 P0/P1 复核 PASS；远端烟测在启动首步完成。预期产物含烟测、PID/CWD/argv/GPU/日志增长、逐行完整训练日志/checkpoint、冻结预测、独立评分、全日志核查及最终报告。用户看到 seed01 后指定补跑，保留已暴露基准背景，不宣称新的独立盲测。

启动读回：9 项真实 checkpoint 烟测 PASS（18 个合成前向，0 source IQ，0 target query）；GPU 4、6 各 3 个实际训练进程，PID/CWD/argv/GPU 对应，日志增长已读回。归档传输一次 SHA 比对通过，远端编译通过。执行 commit：9a8f3a164c5c9e53ba718dfa5da9c0cf9c473972。

GitHub 443 连接失败，push 为 FAILED，未能读取远端分支 OID；本地提交保留。上次发布时任务处于训练阶段；最新状态以下方核查为准。

最新核查：PAOFF、B0、URAW、H01、ORD128 的 seed02 已完成训练并冻结，但预测入口将 manifest 的 6 个 TX 名称列表与整数 6 比较，错误拒绝数据句柄，未完成 target 预测/评分。半份 As 的 seed02 已保存最终权重，但原 native 收尾返回 HELDOUT_EVAL_INCOMPLETE/exit7，缺少 source 完成闭环。6 条权重均保留，10 条未启动；当前本 run 无 GPU 进程。

本轮仅只读检查运行状态并更新登记，没有修复、重启、重训或重测。此前 CE_HALF 的三种子 clean 结果仍完整：84.883929% ± 0.808549%（样本标准差，n=3），该结果属于独立的 CE_HALF 批次。

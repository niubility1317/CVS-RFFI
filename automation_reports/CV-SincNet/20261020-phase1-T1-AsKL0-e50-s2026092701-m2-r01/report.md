# T1 去除 As 独立 KL：两条 E50 路径

状态：LOCAL_VERIFIED，等待原普通账户发布；代码 commit `ed652af1f7b427cb7cff293e6b6185ca76a32bec`。

2026-10-20 用户指定原 T1 E30 拉长到 E50、model seed `2026092701`，并明确要求 E30 续训及从零完整 E50 两种都试。逐行矩阵只维护于 experiment.json。

续训使用原 `20261019-phase1-FcrossRX-AsTemperature-e30-s2026092701-m2-r02/T1-s2026092701/source` 的 FINAL30 学生及自身 EMA，追加 20 轮，重置 AdamW、cos20 和 temporal cache，累计 13L＋37U。原前 30 轮成本继承，新增 4440 次更新。另一条从相同模型 seed 随机初始化，50 轮＝33L＋17U、cos50，新增 11100 次更新。总新增 70 模型轮、15540 次更新；两条各累计 50 轮。阶段和学习率日程不同，不能把两者差异归因于单一轮数因素。

保留原 T1 native scale30 的 As CE、As 独立 KL=0、ordinary/raw RDrop KL、原 six-L 前向、LS0.01、ORD128 CROSS_RX、旧 U 门控与自身 EMA、默认头联合训练。保留原 full practical As 训练增强（不是本对话 F clean-only 增强替代配方），不接 R17、KD 或温度 T6。原 As 生成器硬限 E20/E30，现仅延长合法累计边界至 E50／11100，并按累计 epoch/step 使用原 namespace、seed、scene、硬件 session 与信道过程，保持同一原始 L64 观测的两次 CE。

ManySig equalized1／IQ2×256／unitRMS，源 RX1/3/4/6/8、day1/2/3、全部六 TX，固定 split392005；L6300／U56700／V27000，U TX 始终隐藏，复用已验证角色。续训在读权重前复用原完整 scratch20→own10、source-only FINAL、默认头、实际 role_ids 检查，严格加载学生及 EMA；从零路径不加载训练父权重。V 只做健康诊断，固定最终轮，不按 V 或 target 选轮。

仅 clean 测试：每行训练冻结后自动对原 168000 个固定目标 IQ／opaque ID 预测，默认头及全模型无状态更新；该行全部 clean logits 与 predictions 固定后，经独立 exec CPU scorer 才读取 truth，报告 accuracy、Macro-F1 和 RX/day/TX 分层。正式测试已经暴露，不声称新的独立盲测，不用分数回流参数、选模或重跑。

环境/CWD：N607 原普通账户 CVS-RFFI Python3.10／Torch2.1／FP32／CPU2；`PROJECT/releases/cvs_T1_AsKL0_E50_20261020_r01`。命令：`CVS-RFFI/python -u experiments/cvs_t1_e50/runner.py`。物理 GPU4 串行，每次只有一个本 run GPU worker，原有所有 compute PID 清空且 free≥12000MB 才启动；其他 owner 健康任务保持。不可覆盖输出 `PROJECT/runs/20261020-phase1-T1-AsKL0-e50-s2026092701-m2-r01`，唯一 owner `codex/root/T1-e50`。

预检 VERIFIED：普通账户 direct SSH、项目路径、GPU与磁盘。聚焦验证：4 项预算／隐藏标签权限／累计日历边界负测、原 T1 5 项损失梯度测试、12 文件 AST 通过。独立 P0/P1 审查一次，缺 loss 导入及原通道累计边界问题已定点修复并复审，无遗留 P0/P1。launcher 第一步真实原 T1 checkpoint 无 query smoke 与两路径合成 six-L 梯度／预算检查，PASS 直接排队训练；不新增 smoke 许可。

停止规则：仅本 run 的协议／角色／模型来源／默认头／非有限／输出碰撞／无法启动／无法生成合法预测等系统技术失败；保留全部 partial，取消本 run 未启动行，不自动重试。低分属于科学结果，不触发技术停止，不干预其他任务。技术失败先读回所属 PID/CWD/argv/root 和原产物，再决定后续动作。

预期产物：真实 checkpoint smoke、合成 smoke、每行完整训练/epoch/step/As 记录、学生/自身 EMA checkpoint、实际配置与源冻结证据、clean logits/predictions、独立 truth-last scores、results.json 和 complete.json。训练结束不等于实验完成；缺预测或评分时如实保留状态。

REJECTED_EXTRA_GATE：不增加设计审批、报告/配置 SHA、重复数据验证或重复全量审查。Git commit 为代码及配置身份；发布归档只做一次本地/远端 SHA 比较与一次远端编译。

## 发布读回

VERIFIED：release 归档一次本地/远端 SHA 比较一致；12 个 Python 文件远端编译通过。真实原 T1 checkpoint 无 query 2 NN、两路径完整 six-L 合成 1536 NN（含真实 As、默认头与 Sinc 梯度）均 PASS，0 optimizer 更新，直接进入队列。独立读回调度器 PID/CWD/argv 与连续更新的 queue 状态；failure 不存在。GPU4 在部署期间被其他 owner 健康任务占用，当前两行均 QUEUED，尚无本 run GPU 训练或性能结果；资源空闲后调度器自动依次启动。完整私有进程证据保留在 local_artifacts/cvs_t1_e50/latest_readback.json，不提交原始日志或机器状态。

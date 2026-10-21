# T1 续训 E30＋20：三模型种子复现

状态：RUNNING；执行代码 commit `fe13988582fe73f074706743c9e6e1f248052cce`。本报告为发布前固定的最小实验记录，实际 PID、GPU 和完成情况以随后独立读回为准。

用户在 seed01 两条 E50 路径完成七景比较后选择 E30＋20 扩展三种子。模型种子固定为 2026092701、2026092702、2026092703。seed01 复用 `20261020-phase1-T1-AsKL0-e50-s2026092701-m2-r01/E30PLUS20-s2026092701` 的完整训练、七景预测与评分，不再训练、预测或评分。仅新增 seed02、03 两行，各从 `20261019-phase1-FcrossRX-sameAsKD-confirm-e30-m6-r02/T1-s<seed>/source` 对应同 seed 原 T1 FINAL30 学生及自身 EMA 追加 20 轮。每模型累计 50 轮＝13L＋37U；新增总计 40 模型轮、8880 次更新。原 E30 成本保留，增量 AdamW、cos20、temporal cache 重置。

保留原 T1 native scale30 的 As CE、As 独立 KL=0、ordinary/raw RDrop KL、six-L 前向、LS0.01、ORD128 CROSS_RX、旧 U 门控、自身 EMA 和默认头联合训练。使用原 practical As 训练增强及累计 E31 至 E50／step6661 至11100；不接 KD、恢复器或从零 E50。原 P86 backbone/native model seed 分别为 02、03；其 sequence_pool 子模块初始化沿用原历史约定，严格加载各自完整 E30 学生和 EMA，不改写历史种子定义。

复用合法 ManySig equalized1、IQ2×256、unitRMS、Fs25MHz、split392005；源 RX1/3/4/6/8、day1/2/3、全部六 TX。L6300、U56700（TX 隐藏）、V27000。V 仅健康诊断，冻结固定 FINAL50。原 scratch20→own10、完整角色、source-only final、实际 native seed 与自身 EMA 来源逐行核对，不允许来源不明或契约不匹配的替代。原来源元数据只读核验已通过，启动第一步再用实际父 checkpoint 做无 query smoke。

每行训练冻结后立即自动预测 clean、practical_high、practical_mid、practical_low_suburban、practical_high_urban、practical_mid_urban、practical_low_urban。各视图沿用既有 VALIDATED_ONCE 的 168000 个固定 received IQ／opaque physical ID，不重采样。每行全部七视图预测固定后，独立 CPU scorer 才读取 truth；报告 accuracy、Macro-F1 及 RX/day/TX 分层。三行全部完成后汇总三 seed 均值、样本标准差（ddof=1）和相对各自原 E30 的配对差。seed01 比较已经暴露，三 seed 是用户选定路线的已暴露基准复现，不声称新的独立盲测；不按目标分数调参、选模或选择性重跑。

环境为 N607 原普通账户 CVS-RFFI Python3.10／Torch2.1、完整 FP32、CPU2；CWD `PROJECT/releases/cvs_T1_AsKL0_E30plus20_m3_20261021_r01`；命令 `CVS-RFFI/python -u experiments/cvs_t1_e50_confirm/runner.py`。八卡按全部 owner 的进程数最少、空闲显存最多排序，free≥12000MB、每卡总实验进程≤3，本 run 同卡最多一个 worker；新增两行分两卡并行，独立 CPU scorer 最多一个。保留其他 owner 健康进程。唯一 owner `codex/root/T1-e50-three-seed`，不可覆盖输出 `PROJECT/runs/20261021-phase1-T1-AsKL0-e30plus20-m3-r01`。

发布前已完成 3 项种子绑定／预算／隐藏标签权限负测及 13 文件 AST 检查；一次独立 P0/P1 审查未发现阻塞问题。N607 单次普通账户预检通过。launcher 第一步完成真实父 checkpoint 无 query smoke 和两 seed 合成 six-L、As KL0、默认头梯度及预算检查，PASS 后直接启动训练。Git commit 为代码身份，仅一次归档本地／远端 SHA 比较和一次远端编译，不增加审批、重复全量审查或数据重验证。

停止规则仅限本 run 的协议、物理角色、同 seed 父权重／EMA、默认头、非有限、输出碰撞、无法启动或无法产生合法预测等系统技术失败；保留所有 partial，取消未启动行，不自动重试，不停止其他 owner 或本 run 健康子任务。科学低分按计划完成全部七景并保留。

预期产物：真实 checkpoint smoke、两 seed 合成 smoke、seed01 复用证据、seed02/03 完整训练日志／配置／学生和 EMA checkpoint、七景固定预测、独立 truth-last 评分、三 seed results.json、complete.json。训练完成不等于实验完成，缺预测或评分时如实记录缺项。

发布后独立读回（2026-10-21T06:04:29.653844+00:00）：VERIFIED。归档传输和 13 文件远端编译通过；真实父 checkpoint 无 query smoke 与两个 seed 的 six-L／默认头梯度 smoke 全部 PASS。seed02、03 已分别在物理 GPU0、GPU1 执行，进程 CWD、完整参数、实际 native seed、同 seed 父来源和新增 20 轮预算一致；两卡均有对应 GPU 计算进程。seed01 原七景完整产物只读复用，未重复训练、预测或评分。当前两条仍在训练，尚无新增正式测试分数，不能称三种子实验完成。自动七景预测、独立评分及三 seed 汇总由已运行队列继续执行。

后续启动核验 VERIFIED：两行均完成新增第 1 轮（累计 E31、6882 次更新）；各自训练日志由 6556 增长至 11692 字节，训练序列继续增长。原同 seed 自身 EMA 严格加载、优化器和 temporal cache 重置均有实际产物，未访问 U 真值或 target。

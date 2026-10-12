# Scratch20 工程方案：18 行固定探索

目标仍为 clean 三种子均值 ≥84%，完整报告每个种子。当前实现及发布验证不代表性能目标达成。

本批设计依据为已提交 `51820aee7690e09ca9afd239231095b9ecf1e7f2` 中的 `experiments/cvs_scratch_conditional_fishr/ENGINEERING_NEXT.md`，先于 conditional9 目标评分固定。保留当前条件 Fishr 实验，本批增加优化日程、平坦性正则与标签／特征混合工程方向。历史 E200 或其他续训已有 SAM、平均权重及混合探索；本批不宣称这些算法是首次尝试，也不把已存在的频率单位归一化写成创新。

|方案|实际变化|固定参考|
|---|---|---|
|SSL_LR3|peak LR=6e-4，原 epoch cosine20 到 1e-6|原 scratch20 SSL|
|SSL_WARM_LR3|steps1–444：1e-6 线性到 6e-4；445–4440：cosine 到 1e-6|原 scratch20 SSL|
|SSL_SAM|原 peak2e-4；SAM rho=.02，非 adaptive|原 scratch20 SSL|
|SSL_SMOOTH|仅 L CE label_smoothing=.05；U soft KL 不改|原 scratch20 SSL|
|SSL_HIDDEN_MIX|step445 后浅中层交替，每步仅一个层；不同 TX L 供体，Beta(2,2) 的 max(lambda,1-lambda) 同步混合特征与 onehot 软标签；额外 CE 权重1×ramp|原 scratch20 L_REPLAY|
|SSL_WARM_LR3_SAM（主方案）|短预热、高 LR、SAM 组合|原 scratch20 SSL；批内另比较 WARM_LR3 与 SAM|

三种子固定为 2026092701／02／03，所有 18 行从随机原生身份权重训练 20 轮、4440 次正式 AdamW 更新。旧门控、参考响应和分类头保留。L6300、TX 隐藏的 U56700、V27000 使用既有物理 ID 和契约；不扩大标注比例。两份其他 seed 的原 E20 合规源教师概率均值提供 U ALL soft KL。两个 L-only epoch／444 步后启用 U，888 步 ramp；无 EMA、星地增强、U 风格混合、Fishr、辅助头或校准。V E5/10/15/20 仅健康记录，固定 final20，不选模。

SAM 对全部有梯度的身份 optimizer 参数临时加 rho×g/||g||，包含频率 Hz 参数；第二遍重放同一 L/U、私有噪声、Dropout 随机状态和初始缓冲，计算同一个完整目标。恢复原始参数及第一遍缓冲／RNG 后，使用第二遍梯度做一次正式 AdamW 更新。异常也在 finally 中恢复临时参数，不重复更新。记录实际扰动、梯度范数、额外前向和可扰动参数数目。

隐藏表示混合始终保留 clean L CE 与原 U KL。仅在额外 L 前向安装单层 hook；原生 `time_down`／`t2.pool` 的 [B,C,T] 输出被混合，供体允许重复，真标签仅来自 L。step445 使用浅层，随后严格交替。特征与软标签共用逐样本 lambda，额外前向重放 clean Dropout 并恢复主 RNG；推理无 hook。单类别批无合法供体时记录跳过数，保持额外读出预算。每次保留供体、lambda、形状、覆盖和软标签和误差。

每行完成且冻结后自动 clean168000 全量预测，独立进程 truth-last 评分；18 行共 3024000 个预测，七个目标 RX、四个日期。只测 clean，结果不回流训练、超参数、队列或重跑。每 GPU 最多 3 个总 CUDA 进程，含其他运行任务；一个 launch owner，已启动行不重复启动。负结果保留。

固定比较原 boundary27 的 SSL（三种子）及 L_REPLAY（三种子），不重训这些已完成控制。SAM 每行增加 12432 次前向，额外 L568320／U511488 样本模型读出；hidden mix 每行增加 3996 次 L 前向／511488 L 读出。同 epoch 不等于同算力。每学生20＋两教师历史40＝60模型轮，教师共享 U 缓存170100次样本模型读出单列；预测只执行学生。

本地 native CUDA 检查核验 6 个完整目标、SAM 手工 AdamW 对照、异常恢复、单层混合、平滑 CE、实际 source 入口及严格自有 checkpoint 读回。远端只读源数据前置核验沿用原验证契约，不重复解封目标。独立 P0/P1 审查后，本地提交/push/OID 核实，再发布；实际 PID/cwd/argv/resolved config/首轮日志与容量读回构成发布 VERIFIED 的证据。

完整逐步 gzip 日志、CVS 风格文本、20 轮紧凑 JSONL/CSV、梯度审计、损失／学习率／权重移动／耗时均保留。完成后 `collect_results`、`analyze_source`、`reference_report`、`final_report` 读取全部固定行，报告每 seed、RX/TX/day、混淆矩阵、Macro-F1、NLL、纠错／破坏及实际成本。测试结果不能用于下一方案择优或选择性重跑。


新报告状态更新：**DEFERRED_UNLAUNCHED**。本地实现和CUDA验证保留；旧18行从未发布。新报告要求去除显式CFO并增加同路径控制，后续使用新的不可覆盖run。见 `experiments/cvs_source_identity_readout/NEXT_TRAINING.md`，不把本地完成写成实际启动。

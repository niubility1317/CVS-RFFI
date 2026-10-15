# 固定 SIG_TIME 的 E50 urban 增强首轮

状态：LOCAL_VERIFIED，尚未启动。run_id：`20261015-phase1-sigtime-urban-ssl-e50-manysig-s2026092701-m4-r01`。

本轮按用户“拉长到 E50”执行，从头训练四行、一个模型 seed 2026092701，不继承 E30 权重，不自动扩种子。代码与配置以实际发布 Git commit 标识；dispatcher 从 release_commit.txt 记录该值。此前 E30 的全部权重、结果和负结果保留。

| 行 | 与同预算基线的差别 |
|---|---|
| B0 | SIG_TIME＋旧门控＋cos，clean 与 low_suburban/mid_urban/low_urban 普通拼接 CE |
| C3 | 分别归一化的 1:1 配对 CE＋有界状态课程＋单侧默认头间隔保持 |
| SSL_CONF | C3＋同一样本 clean EMA 置信度 ≥0.95 的 urban 软标签监督，系数 0.16 |
| SSL_WEAK | SSL_CONF 另要求同一样本独立弱视图预测一致且置信度 ≥0.95 |

全部固定 50 轮＝33 轮有标签阶段＋17 轮旧门控半监督阶段，222 步/轮，共 11100 更新/模型。每行父训练成本 0，增量与累计均为 E50；总计 44400 更新。默认头、内部融合、参考响应、SIG_TIME 结构和原联合训练方式不变。原旧门控只继续控制原半监督损失；新增 urban SSL 不使用学生 urban 置信度，也不再叠加 batch-neighbor 拒绝。V 仅在新增 SSL 首个半监督步进行一次冻结的自有 EMA 健康校验：precision≥0.95、coverage≥0.10；未通过时记录负结果并跳过额外 SSL，继续原训练。所有合法有标签 satellite 仍参与 CE。

新增监督资格依据上一轮独立源 V 校准：置信度 0.95 的 precision 为 99.82% 至 99.87%，coverage 为 90.97% 至 91.95%。不根据目标测试调整阈值。主候选预先固定 SSL_CONF。科学探索目标：low_urban≥60%；相对 B0，urban 均值 +3 个百分点、low_urban +5 个百分点、clean 下降≤0.5 个百分点。目标不是结果保证，也不触发自动扩种子、重跑或延长预算。

源 L/U/V 为 6300/56700/27000，原物理角色与 split 392005 不变，U TX 标签隐藏。25 MHz、IQ256、中心裁剪，原录音归一化后进入信道；实际 practical 模块为 experiments/adv3b02_xuc/code。residual/post_sync、EQ 关闭（G=1）、reference_snr_db=30、质量适配/相位跟踪/IQ 补偿开启，reflect 边界，无额外输出 RMS。训练三个场景不变，几何、传播、噪声、质量和补偿残差保留依赖；session 不由 TX 定义。正式测试复用原 VALIDATED_ONCE received IQ，绝不重生成或调弱。

首步执行一次真实、已核实来源的冻结 E50 源模型 no-query smoke，该权重仅供 smoke。已有同一参考模型和未改物理引擎的源诊断直接复用，记录旧路径与原成本，不计为新诊断或训练初始化。

环境：普通 N607 账户，Python `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_sigtime_urban_ssl_e50_20261015_r01`。远端命令：`python -u -m experiments.cvs_urban_sigtime50.dispatch`。本地发布：`E:/type10-7/local_envs/ssr-gpu/python.exe -X utf8 -m experiments.cvs_urban_sigtime50.publish --output E:/type10-7/local_artifacts/cvs_urban_sigtime_e50_20261015/delivery_r01`。唯一 owner 为 `codex/root/urban-sigtime-e50-20261015`。

GPU 由实时容量选择，每 GPU 含其他 owner 最多 3 个实验进程，至少 12000 MB 空闲。唯一输出根 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-sigtime-urban-ssl-e50-manysig-s2026092701-m4-r01`。发布只比较一次归档传输 SHA，进行一次远端编译和预检。启动后独立核实 PID/CWD/argv/GPU/日志增长。协议、数据角色、预算、非有限数、输出冲突或无法运行的技术故障停止未启动行，已启动健康行继续；不按性能停止，不自动重试，不修改其他任务。

每行冻结 final50 后立即自动预测 clean＋六种 practical；该行七视图 prediction 全部固定后由独立 exec scorer 连接 truth，再生成同 row 报告。所有测试分数不回流参数、候选、选模或其他健康训练。四行全部完成后汇总逐行七场景、RX/day/TX、Macro-F1、源状态/质量/门控/间隔、覆盖、50 轮曲线、更新数、前向量和墙钟。预期路径见 experiment.json；原始 checkpoint、预测、日志不入 Git。

验证：E50 10 项检查 PASS（含错误预算/头/测试 SNR 拒绝、教师冻结与 U 标签拒绝、真实结构参数逐项相同、默认头/骨干联合梯度、发布入口）；旧 E30 9 项兼容检查 PASS。独立 P0/P1 检查已完成；发布器导入和远端 E50 路径的 P1 已修复并定点核实，未发现剩余 P0/P1。来源为 reviewer `/root/urban_p0p1_review`；不增加重复审查。

本轮只有一个模型 seed，不能回答 E50 收益是否跨 seed 重现。无实际结果前不声称达到 60% 或性能改善。后续状态和结果在本报告续写。

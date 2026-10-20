# R61：R60两压缩版的无F同TX关联

状态：**VERIFIED / WAITING_PARENT_FORMAL**，已发布。36focusedchecks/CLIhelp/ONEP0P1PASS，method `762202a1da51233da77b16d3401d2b9aa819c7d7`。


状态：LOCAL_IMPLEMENTED，尚未发布。两行 R60 仍在训练，本方案固定时未消费其源验证或正式分数。只复用既有 R56 解析坐标和统计方法，不新增恢复器、分类头、训练或校准。

R60 的两个 273765 参数版本均直接从指定 R46 prune_refine350521 分叉。R61 固定比较 context_hint、all_hint、原指定父版、R53 prune_decay 同宽控制及 raw。R53 控制已在 R60 预登记中固定：同普通 GN 图、weight-only 初始化、E20/4940、global256/micro256 和 IQ 教师日程；新增训练期特征损失的作用单独比较。两条 hint 行间另比较相同 context 系数下额外 prior 约束的差值。原父版只作不同宽度和成本的描述参考，不把所有差异归因于参数压缩。

完整正式划分为 168000 个固定物理样本，clean 与六个 practical 场景全部纳入：郊区高、中、低和 urban 高、中、低。复用 R56 的 raw、R46 父版、R53 prune_decay 共 84 个固定特征单元；仅对 R60 两行已保存的恢复 IQ 新算 56 个单元，共 140 个单元。新解析输入为 2352000 条，不是新增神经网络预测；新特征文件约 3.46 GB，缓存参考只读链接。

坐标保持 mirror64、spectral4、reference24、combined92。标准化只复用 R34 既有 8192 条 L_s/U_s 的无标签源均值与尺度，标准差下限 0.05；新增拟合、校准 IQ、F 权重／嵌入／logit 读取、模型 checkpoint 加载、神经前向、训练更新及另训头均为 0。沿用原物理划分和有效缓存，不重新划分或重复验证数据。

启动器先执行真实 2 条 L_s 的参数自由解析 no-query smoke，随后在 CPU 等待 R60 两行全部七景预测及独立评分完成；在这之前不打开目标 IQ 或 truth。两行全完成后读取已保存 IQ，固定全部 140 个特征单元，独立 scorer 才连接原 truth。clean 参考仅用于事后评价，不提供给恢复器。参考池仍按 TX/RX/day 每格最多 2 条最低不透明 physical ID，查询与参考的 RX、day 均不同，完整六 TX 格等权。

输出为 2016 组描述统计、336 条条件阴性对照和 140 条配对差。199 次查询标签置换固定在 RX/day 内，参考标签保持；这些是条件对照，不作总体 p 值。四个分析组、七场景、逐 TX/RX/day、缺失和零范数均完整保留。R34 固定坐标在 clean TX2 已有弱／负关联，须连同原始 clean 对照报告，不能单凭该类负 margin 宣称恢复成错误 TX。

执行环境：N607 既有 Python／Torch 环境，CPU 2 线程、空 CUDA、0 GPU。controller/extract/score 共用既有 R41 score.lock，最多一个本任务 scorer；不改动任何健康训练、预测或其他 owner。代码与配置以一次 Git method commit 固定。独占目录为 `runs/20261021-diagnostic-restoration-feature-hint-formal-analytic-TX-s2026092701-m2-r61`；release 使用单独 `_r61` 后缀，旧 root 不覆盖。精确 command、CWD、环境、数据与输出路径在登记的 `experiment.json`。

固定命令形式：`python -u -m experiments.pamf_hint_formal_identity.runner run --runs-root ../../runs --source-cache ../../data_caches/phase1_manysig_reference_20261019/source_cache --formal-reference ../../runs/20261019-phase1-restoration-formal-frozen-residual-s2026092701-r05/reference --target-truth ../../runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json --output ../../runs/20261021-diagnostic-restoration-feature-hint-formal-analytic-TX-s2026092701-m2-r61 --expected-commit <method_commit>`。

停止规则：父实验技术失败、方法／路径／ID／源尺度不匹配、缺失或非有限输入／特征、输出冲突时保存现场并停止本诊断，不自动重试，不停止或重启健康父任务。预期产物包括 resolved_config、真实源 smoke、extract/score PID 与日志、calibration_reference、physical_ids、features、feature_complete、summary、complete，以及完成后的独立复算和同一行报告。

识别准确率、解析 margin 和重建误差分别报告；这里只提供固定描述符上的同 TX 关联证据，不能证明每个生成细节的 TX 归属或 TX 排他性。它不替代 R60 七景冻结 F50 正式测试，不用于选模、改父版、改系数或选择性重跑。low urban 70%、其他五景距 raw clean ≤5 个百分点和更强的 TX 细节保真目标继续 ACTIVE。


精确配置、命令和路径见[登记](experiment.json)。


VERIFIED / WAITING_PARENT_FORMAL：普通账户、单归档SHA、compile、真实2Ls无参数解析noquerysmoke已核实；实际PID/CWD/argv/空CUDA/CPU2和日志增长已核实。等待R60两行全部七景正式评分后解析；复用R56raw/350521/R53控制84单元，新算56，总140固定后独立truth-last；无F/checkpoint/神经预测/训练/新校准/头拟合。无统计结果，不改变健康父任务或模型选择。

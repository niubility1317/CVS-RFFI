# 新 CVS SIG_TIME：纯 CE E200 八种子 clean 对比

用户 2026-10-14 明确要求新 CVS SIG_TIME 纯 CE、8 个种子，并确认沿用前面 13 网络的 E200、原划分和仅 clean 测试。所有新权重从零开始，种子 2026092701 至 2026092708；已有 13 网络 × 8 种子结果只读复用，不重复训练或评分。

新架构为原 residual_fusion 时间/频率/PA/参考响应及内部补偿，显式 CFO 输入坐标 27 为常数零，原 t2 输出（包括原生 Dropout）接入 6 维学习投影和 tanh、4 个连续窗口、含时间坐标的路径一级增量与二级反对称面积、每窗 28 维、112→160 投影、LayerNorm、初始 0.05/上限 0.2 的有界残差融合。既有默认 6×160 cosine 分类头保持 scale30 并随骨干联合训练；没有另训或替换头，没有 RX/day 诊断头。总参数 196001（0.196001 M），不含类别 W 的纯特征部分 195041，新增 pool 18976。结构来源 eda57144d 的单 SIG_TIME，原投影初始化和前向输出逐元素完全一致。这个新整体架构包括参考响应等差异，不能将与旧 residual CVS 的差异全部解释成单 SIG_TIME 模块的因果效应。

数据保持合法源 RX1/3/4/6/8、源 day1/2/3、全部6TX、split392005；源 L_s6300/U_s56700/V27000、目标7RX×6TX clean168000及物理ID不变，复用已验证数据。只用 L_s clean 做 CE，label smoothing0，U_s 完全不参与训练，不建立 U 迭代/教师/EMA或额外前向，不加信道、随机多视图或其他增强及辅助损失。V 仅健康监测，不反向或选模。AdamW lr2e-4、wd1e-4、cosine至1e-6、batch128/drop_last=False，50个更新/6300个样本每轮，200轮固定末权重，共10000更新/模型；总新增1600 model-epochs/80000更新。默认头正常端到端联合训练。

每 row 自动训练→E200冻结→clean168000预测→独立truth-last评分，所有预测固定后评分程序才读取 truth；逐 seed/RX/TX的准确率、macro-F1、混淆矩阵、均值/样本SD与实际种子数保留。全部8行评分后自动生成 14 网络的 comparison_eight_seed.json、comparison_per_seed.csv、comparison_summary.csv，比较逐seed配对差异，不筛掉负结果或退化 seed。准确率测试不回流改配置、权重选择或其他健康训练。

唯一 owner codex/root/sigtime-purece-eight-20261014。运行根、release、命令、精确配置、六类seed及环境在 experiment.json；新run和所有source/prediction/score路径不可覆盖。共享 receiver_residual_capacity4.lock，GPU 全部owner总进程计数保守限制2（当前上限3之内），空闲至少12000 MiB；不停止/热修改既有健康任务。两个技术row失败时暂停未开始row，已健康运行的row继续冻结/预测/评分；不因低性能停止、不选择性重跑、不删除产物。真实旧E200 checkpoint只用于合法来源核查及 synthetic no-query launch smoke，不用于新模型初始化。

本地4项聚焦检查 PASS：8seed矩阵/输出冲突/路径越界负例，checkpoint/teacher/query/损失/预算拒绝，私有CPU/CUDA RNG保持与路径数学，真实默认头和所有新增参数CE梯度及数值更新/严格checkpoint roundtrip/逐包一致/零输入有限。唯一独立直接P0/P1审查已完成，PASS。下一步一个Git代码配置提交、ordinary N607单archive传输比对/compile/真实CP noquery smoke，PASS直接启动；启动后独立核对 PID/CWD/argv/GPU/配置和日志增长。

当前：LOCAL_VERIFIED，尚未启动、尚无本次性能结果。only clean 的授权覆盖默认七视图测试；不声称新的独立盲测、Phase2或真实接收机/卫星部署。测试完成前不得标完整实验完成。

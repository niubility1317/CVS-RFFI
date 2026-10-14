# 固定 SIG_TIME 方差方向源检验：剩余两种子

状态：PLANNED；不是训练或目标测试。

实现提交：`5cc440c08feba25ce127cbcf73f489bf04b93e5d`。源端扩展决策已在 `cc2409be311c416c7cf4ed3d0ea6409b79f3f973` 冻结；沿用 seed01 的公式、五个 L180 分组及判据，不参考后续目标结果。

固定矩阵：seed2026092702、2026092703 各自对应 BF 原 SIG_TIME ZERO 源对照 final50；父执行 `3c052b2d36903e28694674a39e232a73acd8b6a6`。第一种子不重跑。L6300 数据角色和旧门控＋cos 训练来源不变，当前只测相同 900 个 L 物理样本。每模型 5 个导数批次、10 个实际矩阵、10 层记录和30个向量；额外两模型合计1800次模型样本观测。跨三模型仍为900个唯一物理样本。

方法：实际 t1/t2 点卷积参数的逐样本 CE 梯度，在五个源接收机内计算无偏方差向量；比较原 RAW 目标与单位方差向量的十对平均 1−cos，最后对两层取均值。eps=1e−15、CE smoothing=.01、辅助图内 W 常量。只观察未加权辅助梯度与同批 CE 梯度的范数及余弦。

每模型通过标准：固定900样本、五批；两层 ANGLE 梯度各5/5非零，聚合 CE 与 ANGLE 梯度范数均大于1e−12、余弦均不小于0；模型和缓冲区不变，零头拟合与优化器更新。科学 FAIL 保留并完成另一固定行；任何 FAIL 后不训练该固定公式。即使三模型 PASS，也需另行登记包含梯度幅度、额外计算对照的训练方案，不能宣称泛化提升。

执行环境：普通 N607 既有 CVS-RFFI 环境、FP32、CPU 2线程，CUDA 隐藏。CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_source_variance_angle_seeds_20261015_r01`。命令：`python -u -m experiments.cvs_source_variance_angle_seeds.audit`。唯一 owner：`codex/root/source-variance-angle-seeds-20261015`。独立输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-source-backbone-variance-angle-manysig-s0203-m2-r01`；本地证据：`E:/type10-7/code/worktrees/phase1_feature_challenge_20261013/local_artifacts/source_variance_angle_seeds_delivery_r01`。不使用 GPU 实验槽位，不改健康任务。

技术停止规则：仅当前运行的数据角色/父来源错误、数值非有限或方差方向不可识别、模型状态/累积梯度被改、输出冲突、固定计数或独立复算失败；保留产物，不自动重启。

发布前窄检查与一次独立 P0/P1 审查均 PASS。发布器先执行真实父 checkpoint 的两样本 synthetic IQ 无 query smoke，PASS 后直接进入源诊断；单次归档 SHA 本地/远端比较与远端编译。发布后核实实际 PID/CWD/argv/log增长或终态产物。每行保存原始矩阵/向量到运行根，由第二进程复算；原始矩阵、模型、日志和物理 ID 不入 Git。

预期产物：登记内全部文件，另含逐行 independent_artifact_recount.json、合并三模型 source analysis；汇总报告与必要聚合数据提交并推送 Git。全程不训练、不读 U 真值、不对 V 前向，不接触目标。


完成状态：ANALYZED/VERIFIED，固定科学资格FAIL（仅1/3模型通过）；不进入该公式训练。完整负结果见 [final_analysis.md](final_analysis.md)。

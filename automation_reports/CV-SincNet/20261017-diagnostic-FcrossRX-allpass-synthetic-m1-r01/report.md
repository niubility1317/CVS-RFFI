# F＋跨 RX 同 TX：有界全通相位响应合成 G0

状态：PREREGISTERED；完整 G0 尚未运行。基线为 F＋跨 RX 同 TX SupCon，训练预算每模型从头 E20；本阶段 0 epoch、0 更新、0 网络前向、0 真实 IQ、0 checkpoint。

假设：在去掉已知一采样点数值延迟后，轻度一阶全通响应可以形成非线性的频率相位变化，并使有限窗口边界误差保持有界。这是工程假设，不是身份保持或跨 RX 有效性的结论。正式 practical 测试 IQ、信道及 SNR 不变。

固定构造：H_a(z)=(a+z^-1)/(1+a*z^-1)，a∈{−0.04,0,0.04}；长度 256；两侧反射 32 点；FP64 递推后转 FP32；不做裁剪、后置 RMS、增益补偿、硬件反演或标签条件化。与数值延迟对齐后的稳态响应为 (1+a exp(jw))/(1+a exp(−jw))。反射不等于真实上下文，有限观察窗不保证能量严格守恒。

唯一矩阵见 experiment.json 与 design.py：8 类合成信号，每类 4 个固定样本，seed=202610172101；每样本 3 个 alpha，共 96 个视图，64 个非零扰动视图。包括突发、幅度台阶及边缘脉冲；不剔除困难反例。

判据：4097 点频率网格幅度误差≤1e−12，最大相位≤2 asin(|a|)，去掉最佳线性相位后的相位能量比例≥0.1（非零 alpha）；全部视图相对波形 RMS 变化≤0.1、相对功率变化≤0.1；与独立 96 tap 冲激级数卷积同边界结果 RMS 差≤5e−6；与已知真实合成上下文的边界误差 RMS≤0.05、去掉两端 32 点后的内部误差≤1e−6；非零视图累计人工上下文误差能量/累计变化能量≤0.1。逐例误差比例>0.5 的情况单列，不能被汇总通过掩盖。

这些界限分别约束“小扰动”、数值实现一致及变化是否主要由人工边界制造；不是测得的真实 RX 范围或身份安全阈值。理论保守变化界 2|a|/(1−|a|)=0.083333（稳态）；有限窗仍按上述固定实例检验。

执行：本地 Windows，CWD=E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014；Python=E:/type10-7/local_envs/ssr-gpu/python.exe，只用 NumPy，GPU 不适用。命令：`"E:/type10-7/local_envs/ssr-gpu/python.exe" -X utf8 -m experiments.cvs_allpass_probe21.runner --output "E:/type10-7/local_artifacts/cvs_FcrossRX_allpass_G0_20261017/run" --commit <CODE_COMMIT>`。输出不可覆盖：E:/type10-7/local_artifacts/cvs_FcrossRX_allpass_G0_20261017/run。Git commit 在启动前登记为实际实现提交；不使用其他代码身份哈希。

检查：13 项非法输入拒绝、5 次成功 apply、1 次 response 控制通过；编译通过。首次开发时缺少算子模块的红检查已出现；早先成功调用计数误写为 9，已在完整 G0 运行前更正为 5。JSON 写入需要独立解析读回。

停止：任一固定科学判据失败，关闭本构造/上界/上下文/样本/seed/判据，不改值补救；技术异常保留失败产物，不自动重试。通过仅允许随后另行预登记合法冻结默认头源域身份检查；不会自动启动源域数据访问或训练。G0 不消费模型/数据/远程设施，故不追加真实 checkpoint smoke、N607 发布或 truth-last 评分；这些由消费对应设施的后续阶段执行。

预计产物：resolved_config.json、全部 case_metrics.json、frequency_summary.json、summary.json、completion.json；独立从全部标量记录复算判据后补充同 run 分析。没有分类指标、训练曲线或真实性能提升结论。


## 用户纠偏后撤下（2026-10-17）

状态：STOPPED_BEFORE_LAUNCH。用户指出主线偏离星地增强，要求不再额外留出训练域。完整合成 G0 从未运行，0 真实 IQ / checkpoint / NN / 更新；不赋予科学 PASS 或 FAIL。保留已写代码、13 项输入拒绝检查和 JSON 写入读回结果作为未启动开发记录，不再启动本候选。

后续训练恢复全部原合法源 RX，使用原 L6300 / 隐藏 U56700 / V27000 角色和物理 ID；这些精确计数须在新 run 启动前由实际原契约读回。RX1/RX6 留域实验不能充当完整源域匹配基线，已有权重不用于新契约初始化。主对照为 F＋跨 RX 同 TX SupCon＋普通拼接式 practical 增强，每模型 scratch E20；先完整恢复该对照，再检验单一星地训练改动。正式测试保持 clean＋六 practical 和独立 truth-last。

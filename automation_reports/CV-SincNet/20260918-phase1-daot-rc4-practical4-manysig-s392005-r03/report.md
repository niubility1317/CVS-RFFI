# DAOT＋FastTrust-RC4 practical四组实验r03

状态：RUNNING／VERIFIED。四组均已启动；首轮或最新进度见下表。

四组full_noeq、full_zf、residual_noeq、full_mmse；本run替代r02，不继承任何checkpoint。

ManySig equalized=1保持不变；开/关均衡指新叠加practical信道的接收均衡。

practical源码固定a8746f38；默认post_sync、星载RX、quality_adaptation、IQ/phase补偿和AGC保留；不是实测LEO。

频率fs=25MHz fc=2.462GHz，来源WiSig原文IV-B：均衡后重采样回25Msps；native loader不重采样。https://arxiv.org/html/2112.15363

ZF regularization=1e-6，ZF/MMSE max_gain=20dB，FIR129 delay64；unlocked跳过均衡，execution摘要区分configured/applied。

residual保留默认关闭均衡，≤1ms快照；full波形SNR与residual白输入期望质量估计差异保留，不宣称严格等价。

课程E1-40 p.30 high，E41-90 p.60 mid/low_urban，E91-200 p.80三场景；卫星CE E80 lambda.68，DAOT E21 clean+high。RC4 hard-sat路径仍关闭。

每组测试clean+本组practical三场景；E200附加既有LEO_WEAK参考。目标先预测再独立scorer，不参与选择/重跑。

无继承任何旧checkpoint；四个row各自独立初始化，使用相同model seed，不是4seed统计。

practical为CPU NumPy参考路径；保持信道实现，不替换近似GPU核。

r01未启动训练；r02仅适配Torch2.1/NumPy2.2边界为显式值拷贝。source/target契约、四配置与随机性均不变。

四组在E1训练后source验证阶段缺少当前批物理ID上下文，ValueError退出；全部进程已独立确认结束，无完整epoch。r03仅补齐source验证三入口context，新增7条/batch3末批回归及远端startup验收；四配置保持不变。

## 验证

本地6项聚焦测试通过，包含source验证遗漏的负测复现及修复后7条/batch3末批通过；真实E100预测＋独立评分、四路禁止Torch NumPy ABI桥接、按ID重排复现。独立审查及两种不同技术故障的定点复核均无P0/P1。远端启动前每组增加真实模型source验证冒烟。

独立P0/P1审查：review_practical_four，四配置、适配器、标签隔离、launcher均无阻断项。原生10项相关测试通过，4组真实checkpoint smoke通过。固定代码commit=f1ea4a5fb62cbb00b1d7f38b05e0c051ef68ba61。

## r03正式启动：VERIFIED

执行代码commit=f1ea4a5fb62cbb00b1d7f38b05e0c051ef68ba61。Torch/NumPy显式拷贝兼容修复的定点复核无P0/P1；远端四组GPU scratch checkpoint＋DAOT L/U反传均PASS。

|组|GPU|PID|完整epoch|真实拼接batch|
|---|---:|---:|---|---|
|full_mmse|7|3200302|1|256.0|
|full_noeq|5|3199619|1|256.0|
|full_zf|6|3200124|1|256.0|
|residual_noeq|7|3200214|1|256.0|

独立证据：E:\type10-7\automation_reports\CV-SincNet\20260918-phase1-daot-rc4-practical4-manysig-s392005-r03\evidence\readback_1789722121.json。四组从零初始化，source物理角色EXACT_MATCH，DAOT与RC4配置开启；DAOT实际按E21生效，卫星CE按E80生效。首个真实ManySig增强batch中ZF/MMSE各128/128实际均衡；full/residual无均衡组0/128。失锁时仍按配置跳过均衡，不能推断之后每个batch均全部开启。

旧CONFIG-CONCAT-SAT静态横幅不识别fused开关；以单前向hook及epoch fused telemetry为准。运行仍未到E100，不声明目标测试结果。原始LEO实验及其他进程未改动。

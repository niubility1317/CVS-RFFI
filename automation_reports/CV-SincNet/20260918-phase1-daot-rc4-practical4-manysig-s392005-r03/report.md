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

## 2026-09-20最新保存checkpoint测试：VERIFIED

本次10:14现场枚举所有已保存checkpoint后固定：full无均衡E130、full ZF E110、residual E200、full MMSE E140。四组已有完整预测，故复用预测并进行独立CPU全量复算，没有重新推理、按成绩选模或更改训练。residual final_ssdg.pth与epoch_200_ssdg.pth的model、ema_model及prototype_memory逐张量一致，均为E200；RNG状态不同，不宣称整个checkpoint文件完全一致。

下表为准确率/Macro-F1（百分比）。

|组|epoch|clean|practical_high|practical_mid|practical_low_urban|
|---|---:|---:|---:|---:|---:|
|DAOT_RC4_PRACTICAL_FULL_NOEQ_s392005|130|80.8327/80.5072|80.3708/80.1316|78.1292/77.8944|52.9649/52.8370|
|DAOT_RC4_PRACTICAL_FULL_ZF_s392005|110|79.1696/78.6202|79.4042/78.9933|77.3792/76.8519|52.9012/51.2079|
|DAOT_RC4_PRACTICAL_RESIDUAL_NOEQ_s392005|200|79.2536/78.7569|80.1774/79.8453|77.8696/77.5892|53.2036/52.9718|
|DAOT_RC4_PRACTICAL_FULL_MMSE_s392005|140|79.3298/79.0091|80.5202/80.2929|78.4810/78.1372|53.8554/52.5357|

全部四组各672000条预测，即同一168000条物理样本的四种场景视图；场景及ID覆盖完整、无重复/遗漏，正确数与原score.json逐场景一致。核对scratch来源、source物理角色EXACT_MATCH、无目标训练接触，row/epoch身份、checkpoint路径及对应practical配置一致。ZF正则化1e-6及ZF/MMSE增益限制20dB沿用固定release。full/residual与均衡开关各自对应，不替换为原LEO或LEO_WEAK。不同epoch与不同增强模型的结果不能当作同预算严格对照，均为已曝光目标的探索性测试，不反馈训练。

证据：[全量复算与混淆矩阵](evidence/latest_test_recount_20260920.json)、[最终模型权重一致性](evidence/residual_final_equivalence_20260920.txt)、[权重清单](evidence/checkpoint_inventory_20260920.txt)。每组服务器预测及原评分路径见复算JSON的prediction_path，其同目录保存score.json/scorer.log/evaluation_scope.json。

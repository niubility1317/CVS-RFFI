# 同信道 HQ 桥接：源域零更新预登记

本次只诊断已有固定 A0 final50 与同 checkpoint 内 own EMA，不训练或选择模型。父模型是已登记八位置研究的预定普通拼接基线，未按目标分数选择 S/J。完整源域数据与逐行配置见 [experiment.json](experiment.json)。

同一个源 IQ、传播实现和 RX/day session 构造 full、共享噪声 HQ、独立噪声 HQ、无新增噪声四个视图。HQ 将模拟器新增噪声降低 6 dB，冻结 full 的接收质量代理、锁定状态、残余补偿、传播／残余／均衡核、AGC 增益及限幅决策、FIR 延迟和边界。原录音未知噪声保留，新增噪声 SNR 不冒充总 SNR。实际引擎没有逐样本硬裁剪，不编造 clipping mask。

## 范围与预算

- 仅内部 L_dev 720 个物理样本，源 RX 1/3/4/6/8、day 1/2/3、6 TX；L_train 5580 与 L_dev 分离。复用现有源缓存可能物化 V 容器，但 V 前向为 0；不开 U，不调用目标加载器。
- clean 各前向 student／own EMA 一次，加六种源 practical × 四视图 × 两个模型，共 36,000 个前向样本。0 个新增 epoch、0 次 optimizer／EMA 更新、0 次另训头。父训练 E50／11,100 次更新及实际墙钟单列。
- 默认 cosine 头保持 checkpoint 的 6×160、scale30；所有诊断 eval，记录原头准确率、固定父 clean 正确集保持率、修复／新增错误、除以 30 的间隔、EMA 熵和 RX／状态／质量分层。
- joint160 比较同样本与最近不同 TX 的余弦差；不拟合读出器。实际 SIG112 按每窗 increment7／area21 分成 6／15／6 三组，排除常量 dt。源 own EMA HQ 的固定 RMS 尺度带 1e-3 下限，先组内平均。
- 源随机流 2026101504，与训练／验证／其他源诊断／正式测试分离。HQ 设置不是修改正式测试 IQ 或测试信道。

## 实现、执行与保护

代码为 `experiments/cvs_urban_bridge/`，配置为该目录 `registration.json`。代码身份使用实际发布 Git commit；本报告及源码不增加独立 SHA／签名要求。N607 普通账号、CPU 2 线程、CUDA_VISIBLE_DEVICES 为空；CWD、命令、唯一输出根及 launch owner 见 experiment.json。发布器执行一次 archive SHA 比较和远端 compile，首个真实 checkpoint 无 query smoke PASS 后立即继续；随后独立读回 PID／CWD／argv／日志增长。

遇到契约／checkpoint 来源、物理冻结、非有限数、前向预算、输出冲突或技术错误时仅退出本次 owned 诊断，保留产物。不自动重试，不改变其他健康任务。本次显式 source-only，无正式目标测试或 truth-last 评分，不能把已有父模型重测成新实验。

本地已执行 5 项针对性测试（通过，0.754 s）：full 与原引擎一致、固定状态及 AGC、EQ 噪声着色／context 边界、噪声剂量与非法值、SIG 真实坐标及梯度边界。编译通过。独立 P0/P1 审查发现并修复 Linux validation_ref 路径问题，定向复核通过；未发现其他 P0/P1。真实运行证据待补。

## 后续 D/E 设计边界

D/E 都保留 clean→urban 的直接 joint 身份锚点，另加 HQ student CE 与 clean→HQ 锚点，额外 student／teacher 前向一致。E 单独增加 urban raw SIG112→HQ own EMA raw SIG112 的分组教学。保持原默认头联合训练及旧门控 U 流程，首个比较不加新 U 特征目标。不得用 HQ-only 替换 direct anchor，也不把本次诊断变成额外可行性证书或审批门槛。

本次尚未执行 D/E 训练，尚无训练收益、clean 代价或跨 seed 重现结论。既有八位置族源域扩种子条件失败，继续保留其负结果，不扩种子或选目标最高的分支。

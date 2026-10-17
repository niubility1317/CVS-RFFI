# I/Q 方向变化的 A_g 来源核对

状态：只读代码与来源核对完成，0 新真实 IQ／checkpoint／网络前向／优化更新／target。没有登记或启动共轭增强训练，不构成新的识别收益，也不将证据不足写成物理现象不存在。

本轮在匹配 No-As E30 证实普通 As 的 practical 收益与 clean 代价后，核对了一个不同的候选来源：源 RX 的 I/Q 方向约定是否不同。它与恒相位旋转、线性频偏、时移或一般 FIR 不同；但有不同的公式，并不足以成为有效 A_g。

## 一般物理含义与当前数据证据

GNU Radio 的 Swap IQ 用于校正谱反转；UHD 的前端连接接口也显式区分 IQ 交换与 I/Q 极性。这说明观测约定在实际工具中存在，不证明本项目接收数据具有这种变化。[GNU Radio](https://wiki.gnuradio.org/index.php/Swap_IQ)，[UHD 前端接口](https://files.ettus.com/manual/classuhd_1_1usrp_1_1fe__connection__t.html)。

复共轭 J(I+jQ)=I−jQ；I/Q 交换为 jJ(x)，还包含固定相位旋转。J 是自逆、保能量且无切片接缝的数字操作，但数字变换不等于更换真实 RX。其相对输入能量变化为 4·mean(Q²)/mean(I²+Q²)，范围0至4；I/Q功率相近时约为2，因此它也不是轻度 IQ 扰动。不能由保能量、自逆或可逆，直接推出适合身份泛化。

已按实际 No-As 发布环境 bootstrap，只读导入模块并检查函数，没有构建模型或读取数据。实际入口为 PROJECT/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/code/dataset_wisig.py。WiSigCompactDataset.__getitem__ 统一执行原数组转置、必要长度处理与可选 RMS；不按 RX 交换或取反 I/Q。它保留原记录方向，不能证明采集端方向一致。可见元数据没有前端方向字段。

实际参考响应来自同一固定依赖的 experiments/cvs_reference_identity/model.py。response_contract 明确为 iq_convention_identified=False、rx_arbitrary_channel_invariant=False、hardware_parameter_estimation=False。固定窗80至160，20点周期，使用对称12个 L-STF 音调±4、±8、±12、±16、±20、±24。NoExplicitCFO及既有补偿不变。

## 为什么参考响应不能单独确认方向

对参考支持上的非零 S[k]，当前相对响应表示允许未知复传递量 H[k]：Y[k]=S[k]H[k]。共轭方向产生 Z[k]=Y*[-k]，也可表示为同一参考下的未知响应：

Z[k]=S[k]·H_tilde[k]，其中 H_tilde[k]=S*[-k]H*[-k]/S[k]。

对称音调支持下，若没有经过验证的响应约束或独立采集约定，这两个表示都可由未知复响应解释。该代数说明当前表示的识别边界，并未估计真实 RX 或宣称全部实际信号都是理想 LTI 输出。不能用选择相关峰、参考质量或拟合一个额外 RX 分类器，替代缺失的物理依据。

历史检索中的 mirror 子空间是特征归因或结构实验，不能替代共轭 IQ 增强证据。当前关键词检索未定位到同一共轭增强训练，但“未找到重复实验”也不构成发布理由。

## 决定

当前唯一待核对的来源假设是“RX 方向约定变化提供一个合法 A_g 来源”。本轮未取得本数据中该变化的可靠依据；在现有256点、无更长原始上下文映射的条件下，不发布强共轭训练，也不通过插值共轭来变成未经验证的 IQ 失衡增强。没有新增损失、分支、头、U 规则或数据权限。

这项核对只停止当前候选的发布，不证明方向现象不存在或任何共轭训练都必然无效。当前没有已支持的新 A_g；已失败的频偏、恒相位、FIR、同TX混合、梯度处理等固定路线保持关闭。联合研究目标仍 ACTIVE／未完成。

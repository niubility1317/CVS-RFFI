# WiSig 原版 DURA 估计监督复现

实验 `20261125-phase1-wisig-original-dura-adaptive-m1-r01` 检验一个变化：用源域 clean/received IQ 对的 Adaptive 估计替代原版 R07 的模拟器信道作用和噪声方差。冻结的历史原版 R69 是同包基线。

| 项目 | 固定设置 |
| --- | --- |
| 源数据 | 原版 R07 的 63,000 个物理样本、20 轮原始 received IQ，含 6,300 L 和 56,700 U。R02 与 R07 共用同一份源 IQ 和角色列表。 |
| 估计监督 | 冻结的全包 Adaptive，复杂度 4、收缩 0、融合权重 0.2。把估计的无噪波形拟合进 R07 的 64 抽头反射 FIR 与逐时刻主、镜像系数；把复噪声方差除以 2，按原头的每实分量范围裁剪。源 IQ 按字节相同的 clean 对使用原恒等目标。每轮保存投影误差分位数和方差裁剪数。 |
| 训练 | 保留原 R02 `tf_unet_mixed` E20 权重。重训 R07、R17、R38、R44、R46、R65 和 R69，各 20 轮、4,940 步，共 34,580 个新更新。G 种子 2026092701，批量 256，FP32，固定取 E20。R07 保留两组白噪声探针和 0.1 倍对数方差误差。后继只从本次新权重继承。 |
| 测试 | WiSig Main 的 126,000 cross-RX 与 42,000 cross-both，默认和加重强度、Full 和 Residual、clean 和六种 Practical，冻结三组 PAMF。估计监督 R69 使用完整原推理输出，所有测试 IQ 经 R69 后进入 PAMF。比较冻结的原 R69 和 raw 的同包决策。 |
| 环境 | N607 普通账户，项目既有 CVS-RFFI Python。16 个单线程估计进程，训练在显存余量至少 12 GiB 的低负载卡上占一张。最终预测最多两张卡并行。 |
| 停止条件 | 仅本次运行的数据不匹配、非有限数值、训练或预测系统性失败。保留已有产物。 |

训练进程只获得源 IQ、R02 初始化与估计产物的路径。模拟器内部作用、噪声、真值、目标 IQ 和冻结 PAMF 权重不进入恢复器训练。八个预测分片全部封存后，由单独的 CPU 进程读真值评分。测试成绩不用于训练或选模。

发布入口为 `python -m experiments.pamf_unrolled_restore.wisig_original_dura publish --repo <checkout> --output <local_artifacts> --context <N607_context.json> --ssh-config <N607_config>`。远端不可变工作目录为 `releases/<run>-release`，训练、预测和评分产物位于 `runs/<run>`。首批记录为 `source_smoke.json`、`estimates/01/complete.json` 和 `G/action/first_update.json`，后续保留每轮分项损失、`G/r69/final.pt`、`evaluation_smoke.json`、`predictions/` 和 `scores.json`。

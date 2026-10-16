# F＋跨 RX：星地训练后新增噪声的固定接收状态诊断

状态：PRE_REGISTERED，尚未进行真实 checkpoint 前向或源 IQ 诊断。

增强前底座固定为 F＋跨 RX 同 TX 对比学习：三路 CVS/SIG_TIME_X3、统一 Sinc CENTER_UNITS、NoExplicitCFO、joint160、既有 cosine30 默认头。训练仍按既有流程联合更新，预算每模型从头 E20；本次为冻结诊断，新增训练为零。起点 Git commit：9bcfd611370b110ca5d540fc8b9b200239a3f665。实际发布代码 commit 由 release_commit.txt 与本 run 登记保存。

前轮匹配 E20 噪声配对训练的固定主候选 ANTI 未优于 INDEP：源留出 RX1 的 urban 均值相差 −0.1323 个百分点、low urban 相差 −0.3968 个百分点，固定方案已关闭。ANTI 相对匹配 RAW10 的 clean／low urban 分别增加 6.1111／6.0317 个百分点，但 BLOCKED 与 unlocked 未获改善；这不是正式目标成绩，也不允许据此晋级 INDEP 或扩种子。

唯一假设：训练后默认头在这些困难状态仍有被新增噪声遮蔽的可用身份线索。固定主模型 ANTI 的 BLOCKED 和 unlocked 两层必须各自恢复至少一半 full→clean 正确数差距：gap=clean_correct−full_correct>0 且 2×(control_correct−full_correct)>=gap。缺失分层或 gap<=0 为不支持；两层允许重叠。任一失败即关闭当前默认头噪声遮蔽假设，不能改变阈值或补抽样挽救。通过也只构成源域研究依据，不自动启动训练或访问目标。

最小矩阵为两个冻结模型 RAW10（描述对照）与 ANTI（主检验），均为 seed 2026092701；各自使用 P8 同 fold 从零训练 E20／4,440 更新的 FINAL20 student 和原联合训练默认头。没有上游模型继承、目标训练或目标选模、另训头。逐项核验原 config、实际 args、初始化、角色及冻结报告后严格加载。P8 路径和 backend commit ee305facac7874bfc1340291ee5360a3665621d9 见 experiment.json。

只消费合法 seen-fit L_s 5,040 条（RX 3/4/6/8，day 1/2/3，TX 0..5；72 个 TX×RX×day 单元各 70 条）。原 VALIDATED_ONCE split392005 角色复用，不重验数据。加载既有 source_cache.pt 时实际物化 L6300、未使用的 V 和旧父特征；随后仅抽取精确 fitL5040 的 IQ／可见标签／角色字段，旧特征、U 和 V 数值不进入任何前向或诊断。留出 RX1 不在本矩阵内。诊断样本曾用于原训练，不能宣称跨 RX 或目标泛化。

三视图：原始已归一化 clean、原 full practical_low_urban，以及去除同一次 full 中新增噪声的控制。25 MHz、IQ 256、fc 2462 MHz、residual／post_sync、EQ 关闭 G=1、reference_snr_db=30、reflect 均固定。原 full 只运行一次，使用原 signal/noise trace 验证 IQ 完全重建，控制固定同一接收状态、接收质量、相位／IQ 补偿、H／残余核和裁限后的原 AGC 标量；不重新计算 AGC 或状态，不反演录音中未知 TX／RX。此为人工受控干预，不能部署，也不是信息论上限。信噪比字段为 signal_to_added_noise_db，不是总 SNR。正式测试 IQ 和配置不变。

命名流固定：order=20261017601、channel=20261017602、noise=20261017603、receiver=20261016102；两模型每个物理样本使用相同原信道／噪声观测。硬件 session 为 RX/day/eq，不由 TX 定义。

沿原 forward_identity_only 与默认头记录完整六类 logits、间隔／30、CE、现有七处分支输出范数、与自身 clean 的余弦、时频及 PA 原门控、CE 对中间输出的导数。仅临时观察钩子，不修改前向，不存 IQ／embedding，不拟合任何参数。eval 状态，参数及缓冲逐位不变，无 .grad 累积、优化器或 EMA 更新。NoExplicitCFO 第 27 坐标实际必须为零。

预算：每模型首先真实 checkpoint 无 query 的两条零输入 smoke，通过后立即继续；之后 5,040×3=15,120 源域决策，共 15,122 NN、40 批、120 次 autograd.grad、15,120 包等效中间导数。两模型合计 30,244 NN／240 次 autograd、0 新训练 epoch／0 更新。每个父模型 E20／4,440 更新和原前向成本单列，不冒充本次训练。

本地验证 PASS：47 个协议负例、96 个合成 NN、0 真实 IQ／checkpoint、0 更新；原推理 logits 与挂点推理逐位一致，完整 IQ 重建精确、控制 AGC 未重算、参数缓冲不变、钩子移除。JSON resolved config 往返纯静态核对通过。唯一独立 P0/P1 静态审查 PASS，未发现问题；本地编译 PASS。

环境／命令：普通 N607 CVS-RFFI Python，原 fullFP32，CWD 为 releases/cvs_FcrossRX_postAs_noise_control_fitL_20261017_r01；完整 launcher／逐行命令见 experiment.json。唯一 owner=codex/root/FcrossRX-postAs-noise-control-20261017。共享 GPU 全 owner 总进程不超过 3，空闲至少 12,000 MiB；2 个诊断 worker，各 CPU2。独占 runs/20261017-diagnostic-FcrossRX-postAs-noise-control-fitL-s2026092701-m2-r01，不覆盖旧输出，不停止或热改其他健康进程。一次预检、一次归档传输校验、远端编译后发布，并以 PID／CWD／argv／GPU／日志增长独立核实。

停止规则：仅本 run 的非法角色／来源、非有限值、冻结状态或预算改变、冲突和不可启动等系统技术故障；保留部分产物，无自动重试。低准确率属于科学结果。旧额外审批、seal、重复审查等均为 REJECTED_EXTRA_GATE。

预期保存每 row config／smoke／原模块路径／角色／完整 prediction 和 physics／summary，完成后独立离线重算准确率、宏 F1、正确数恢复、固定 RAW10 clean 正确集合保留率、各状态／质量／RX／day／TX／核／边界／AGC 分层、现有分支和门控图。非训练诊断没有新的训练曲线；父训练曲线保留 P8 原报告。全部正式目标与跨种子联合增强验证仍未完成。

# F＋跨 RX：白输入质量代理的状态切换与边界反事实

run_id：20261017-diagnostic-FcrossRX-qualityproxy-fitL-manysig-m1-r01。当前为 PLANNED；本报告固定首次诊断，尚无真实数据结果。

增强前基线保持 F＋跨 RX 同 TX 对比学习（SIG_TIME_X3、统一 Sinc 修正、NoExplicitCFO、joint160、默认 cosine 头），训练实验仍为从头 E20。本次只诊断物理接收过程，不实例化模型，不读 checkpoint，不训练或新增分类头。

唯一假设：残差实现按白输入功率近似计算接收质量，对实际短时结构化波形的估计差异，足以在 mid/low urban 造成稳定接收状态切换，并且差异主要不来自人工 FIR 前缀。

## 数据和实际运行配置

复用既有 VALIDATED_ONCE source carrier。原 L6300/U56700/V27000 契约不变；只使用当前 C0 训练 RX3/4/6/8 的 L5040（TX0至5、day1至3、72 个 TX×RX×day 单元，各70包）。源 RX1 的 L1260、U、V 和目标均不用于物理统计。已有 source_cache 整包反序列化包含未用源角色，读取器不检查这些角色的 IQ 或 U 真值。C0 仅提供实际 L_s 物理 ID 元数据，不继承任何模型权重。

采样率25 MHz，中心频率2.462 GHz，IQ256。cache 已按原协议 RMS 归一化，进入原信道 normalize_input；完整输出始终保留256点。真正导入既有 release 的 experiments/adv3b02_xuc/code/leo_practical，在 actual_engine.json 保存 channel/residual 的 __file__ 及六种场景全部实际配置。固定 residual、post_sync、EQ关闭（G=1）、reference_snr_db=30、reflect；保留原接收质量适配、相位跟踪、IQ补偿、AGC及几何/遮挡/传播依赖。RX 硬件 session 为 rx/day/eq1，绝不由 TX、scene、draw 定义。

质量分母为原新增噪声期望方差0.001，不能称为原录音与新增噪声合并后的总 SNR。真实 Hx 这里只是相同冻结 H 的 preRX 功率反事实，不声称等同完整时变 full-route 接收过程。FIR 数值延迟24点；内部功率统计使用索引 len(H)-1 至255，排除人工前缀影响，绝不裁剪送出波形或删除困难包。

## 同物理样本的五个条件

1. 原生白输入代理：mean|x|² × sum|H|² / 新增噪声期望方差。
2. 显式同质量重放，验证与原生 IQ 输出逐点完全相同。
3. 同一 Hx 全256点功率估计质量，重新计算接收状态和补偿：总效应。
4. 同一 Hx 的无人工前缀内部功率估计质量，重新计算接收状态和补偿：边界对照。
5. 条件3的质量，但固定原生 receiver_quality 返回的 status、residual_scale；其余原处理和随机抽样不变：受控零效应校验。

五个条件共享 H、RX session 和底层噪声随机流。状态改变可能改变残余补偿和最终信号，不把总效应冒充固定接收状态效应。

## 预算、判据与停止规则

两个独立物理命名随机流 × 六种原 practical × L5040，共60,480条记录、302,400次原生物理 process 调用；它们不是60,480个独立源样本。augmentation seed20261017891，receiver seed20261016102，split392005；无 model/evaluation/support seed。记录12个 scene/draw、864个 TX×RX×day 单元、12个固定硬件 session。新模型、checkpoint读取、神经网络前向、优化器更新、教师前向和头拟合均为0。

每一个 draw 的 mid_urban、low_urban 都须同时满足：质量绝对偏差中位数≥1 dB；全长估计接收 lock 切换比例≥5%；全长与内部估计差异中位数≤0.5 dB；全长切换中至少50%在内部估计上仍切换到相同状态；全部内部区间至少128点。

PASS 只支持另行登记冻结默认头可恢复性检验，不代表识别收益、有效 Ag/As 或自动开展 E20。FAIL 关闭本固定代理/状态切换假设；不改阈值、draw、seed、场景、窗口、RX 或训练预算补救。任何权限、输入、非有限值、输出冲突或启动失败均保留产物，不自动重启，不影响其他健康任务。科学结果不反馈目标调参。

## 执行和验证

实际代码/配置身份以本次 Git commit 为准，release_commit.txt 保存发布时 HEAD；不增加文件或配置哈希门槛。命令、数据句柄、独占路径、owner 和预期产物见 experiment.json。N607 普通账号、原 CVS-RFFI 环境、CPU2、CUDA隐藏；CWD为本次唯一 release 目录。启动入口：/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_qualityproxy_fitL_20261017_r01/experiments/cvs_quality_proxy_source/runner.py。

已完成本地 compileall，以及22个输入/协议反例、5个科学判据反例、六场景6个合成包/30次原生物理调用、1,048,576个解析白输入观测和独立充分统计复算。真实 IQ、checkpoint、NN、更新均为0。一次独立静态 P0/P1 review 为 PASS；没有新增审批。

本阶段不消费模型/checkpoint，无实际 checkpoint smoke 所需机制；launcher 首步执行原生信道 no-query smoke（2合成包、10物理调用、0NN），PASS即继续。发布执行一次 N607 preflight、一个归档本地/远端 SHA 比较和远端 compile；启动后独立读回 PID/CWD/argv/CUDA隐藏及日志增长。

## 预期产物与尚未完成

实际模块/config/数据角色、完整60,480条充分统计、全部场景/状态/质量/接收切换分层、独立完整复算、边界/CFO/AGC图与实际墙钟成本。原始 IQ、权重、原始日志和生成图不提交 Git。源域诊断不能代替冻结默认头准确率或正式 clean＋六 practical 测试；跨模型 seed 重现与联合目标目前均未证明。

## 发布读回

运行代码 commit：4f7968ec32ea52ff56f63c42c1654fb4507573d6，已推送且 GitHub 独立 OID 比对 identical。N607 preflight、单归档传输和远端 compile 均 PASS。普通 CPU 进程156603的 PID/CWD/argv/CUDA隐藏及增长日志已独立核实；完整源统计仍在运行，当前记录不是科学结果。离线复算及绘图的后续提交不热修改运行 release。

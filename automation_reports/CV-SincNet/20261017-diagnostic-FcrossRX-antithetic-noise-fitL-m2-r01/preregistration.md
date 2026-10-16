# F＋跨 RX：正负噪声配对的 CE 方差源诊断

本轮只检验一个训练采样理由，不训练模型，不把梯度变化当作准确率改善。固定基线为 F＋跨 RX 同 TX SupCon、SIG_TIME_X3、统一 Sinc CENTER_UNITS、NoExplicitCFO、joint160、默认 cosine30。既有 raw8L 计算匹配对照的两个 scratch E20（2026092701/02）仅作冻结观察对象，不用关闭候选的权重初始化新训练。

前一固定相干 IQ 插值已因跨 seed clean 确认失败关闭：seed02 相对 raw +2.301587 pp，seed01 −0.714286 pp；low urban 两 seed 都下降。固定 additive As 曾提高 urban，但 clean 保持未跨 seed 通过。此前 ordinary／As 梯度支配理由也未通过。这里不重复这些机制或调其幅度、预算、RX。

唯一假设：相同物理信道、接收机与补偿状态下，对原始独立高斯新增噪声使用 z 与 −z，可降低 satellite CE 梯度的采样方差。两种符号各自具有相同边际噪声分布；不平均 IQ、不减少噪声方差、不改测试 SNR。每个视图都从原 residual 的噪声入口经过 RX IQ、补偿、G 和原 AGC；AGC 按视图重新计算。状态仍用原白输入期望功率代理，新增噪声 SNR 不冒充总 SNR。实际 EQ 关闭即 G=1；合成检查另覆盖 EQ 开启的噪声着色。

两 row：frozen-D_raw-s2026092701、frozen-D_raw-s2026092702。严格核对各自已有 scratch E20／4440 更新、无父来源／目标选模／额外头、相同物理 L/U/V 角色，完整加载默认头。fit RX3/4/6/8 的全部 L5040、72 个 TX×RX×day 单元各70包，标签无关固定排列为40批，末批48保留。旧缓存反序列化完整 L6300、源 V 和父特征，只有 fitL5040 数值进入观察器；held RX1、U/V 不前向，不用 U 真值。每个 row 在读取该缓存前先做真实 CP 的2包 zero IQ no-query smoke。

六种 practical 的 nominal 均匀先验保持，25MHz／IQ256／fc2462MHz／residual／post_sync／reference30／reflect；不拒绝状态或困难样本。RX/day/eq 硬件 session 不由 TX 定义。每物理包的所有非噪声 latent 固定，8 条独立命名噪声各生成正负视图；两个模型复用相同实际观察记录。诊断随机流与训练、源验证、正式测试分离。

模型 eval，无 dropout；观察默认头 CE（label smoothing .01）的全部可训练参数导数，不执行 backward 累积、optimizer、EMA、头拟合、KL 或 SupCon。Sinc q 导数除25000映射回物理 Hz，其余坐标不变；五组互斥参数之和为 whole。对每批8个噪声的 g+、g−计算无偏方差：V_ind=(Vplus+Vminus)/4，V_ant=Var((g+ +g−)/2)，并校验 V_ant=V_ind+C/2。独立对照由相同边际梯度样本估计，不冒称额外运行了独立双视图前向。保存完整16×16梯度 Gram，离线独立重算全40批的组／whole 统计和全部预测、accuracy/F1、margin/30、scene/state/lock/RX/day。

固定证据条件：两个 seed 都要求 whole 的 sum(V_ant)/sum(V_ind)≤0.80，且至少60%的40批 C<0。任一失败即关闭这个方差理由，不换层、状态、RX、seed、抽样数或阈值救援。通过只允许另行登记同预算训练检验，不能证明 Ag、clean／urban 收益、真实 AdamW 轨迹或联合目标，不自动训练或访问 target。

新成本：0 model-E／0 更新／0 EMA／0 头拟合；每 row 80642 NN（含真实 CP zero smoke2包）和80640次描述性fitL判决，总161284 NN。既有父训练40 model-E／8880更新／11646636 NN单列。本地已执行32个合成真实网络前向包、0更新；34个配置／权限／统计负例，完整配对批次及原视图一致性通过；新增 reader oracle曾因合成4抽样与固定8抽样不匹配失败，修正fixture后通过，未读取真实数据。一次独立 P0/P1 发现 tuple/list 配置比较会误报，已本地复现并统一两侧规范化，真实SNR变化仍拒绝；唯一该项定向复审 PASS。

一个不可覆盖 run／release、唯一 owner codex/root/FcrossRX-antithetic-noise-20261017；共享 all-owner≤3/GPU、空闲显存≥12000MiB、每 worker CPU2。仅本run协议／来源／非有限／状态变动／碰撞／无法产生完整预测等技术错误保留 partial 后停止，无自动重试，不操作其他健康任务。一次普通账户 N607 预检已通过；发布仍按一次归档本远端比较、一次 remote compile、启动后 PID/CWD/argv/GPU/log增长读回。

实际代码／配置 commit：1bc071c9d3148de703f2ffcd6322feaa3aa33db6（已推送，远端 OID 独立相同）；逐 row 配置、命令、环境/CWD、路径、GPU政策、stop 与预期产物见 experiment.json。入口 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_antithetic_noise_fitL_20261017_r01/experiments/cvs_f_noise_pair_probe/runner.py。本地输出 E:/type10-7/local_artifacts/cvs_FcrossRX_noise_pair_probe_20261017；远端输出 /home/szu2070436088/2510044040/CV-SincNet/runs/20261017-diagnostic-FcrossRX-antithetic-noise-fitL-m2-r01。

REJECTED_EXTRA_GATE：不增加独立设计审批、配置 SHA、receipt 链或重复审查。当前 ACTIVE 目标仍未完成；正式 clean＋六 practical、low urban>60%、suburban 接近 clean、合格 Ag 与四格交互均未获证明。

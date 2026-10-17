# E30冻结默认头：数值延迟与全局传播相位的源域四格诊断

状态：REGISTERED，尚未运行。实际代码commit 71f0c8f9fbdd530c51f54764f092b61b9e5f6f18，起点4be849bc060068c1bf72f55676ac8ec3eabf6968。既有P32四因子候选已关闭，P33源归因已完成；本次不是MMSE24/PDPshape/allpass或固定周期Ag的救跑，不改变训练预算或正式测试。

## 唯一问题与最小对照

P33发现：As总体收益在单独移除新增噪声、频率选择性、残余载波/相位或RX-IQ残差后仍存在；这些控制均未显示预定的稳定敏感性下降。flat_H/fixed仍保留24点数值延迟及原全局复相位。因此检验它们在这个平坦核family中是否构成模型差距的重要来源。

四格以同一物理V样本/同latent/同白输入核能量构造：D00冲激位置24+原传播相位；D10位置0+原相位；D01位置24+零传播相位；D11位置0+零相位。所有格固定原full的lock、scale、频率估计、相位/IQ补偿、G1、AGC和同一新增噪声。保留RX/day硬件session，不按TX定义。D00逐包对原P24 flat_H/fixed重放，误差超2e−6即技术失败，缓存原预测复用。

只在源端构造已知平坦核counterfactual，不反演原录音TX，不重构未知未来样本；D10不是将真实色散观测无损前移的证明。改变脉冲位置同时改变反射边界的受影响前缀范围，解释限制必须保留。原传播几何/遮挡/径功率用于核能量，其他残余接收状态固定。正式IQ/SNR/信道/困难样本均不变。

## 模型与数据来源

固定P32 matched B00/B01各自scratch E30（20L10U、6660更新）、FINAL30的model state，非EMA，默认头与训练骨干原联合方式保持；本次全部eval/frozen无optimizer/autograd。F=SIG_TIME_X3、Sinc CENTER_25K/CENTER_UNITS、joint160/defaultcos30、NoExplicitCFO/原内部补偿、900722参数/206state；无新部署结构或诊断头。

严格own_model先核实scratch来源/无ancestors/原source_contract物理角色/FINAL30选择，第一数据步骤是2真实CP+4合成包no-query smoke，PASS即继续，源IQ此前0。之后复用P30合法panel函数读取原V缓存：1800包=6TX×5原RX索引1/3/4/6/8×3day1/2/3×20，完整L6300/U56700hidden/V27000、split392005及7/63/30保持。不留训练RX，不切L_dev、不解封U TX、不更新V状态。

复用P24 draw0/1的mid_urban/low_urban命名随机流，seed202610172402、panel_seed202610172401、hardware_seed2026092701；种子大整数由Python直接读取，无JS数值中转。P33原same-CP flat_H/fixed的logits/IDs/TX/RX/day逐值对齐，clean logits同样复用。actual_engine模块__file__与原记录匹配。采样率25MHz/256点/原RMS输入，residual/post_sync/EQoff(G1)/refSNR30/quality+phase+IQ补偿/reflect原引擎不改。新增噪声比不是总SNR。

## 先固定的源判据和放弃条件

逐scene/draw、逐模型记录四格accuracy/macroF1/统一logits30尺度的margin、固定B00 clean正确集合保持率、同包修复与破坏、RX分层及delay×phase交互。

分别对delay和phase单项，要求全部4个urban×draw单元：B00从D00到对应单项至少+5pp，且As优势减少至少max(5pp,原As优势的50%)。这只支持平坦核family中的重要贡献。任一单元失败即不支持该轴解释，不扫描延迟/相位/比例/seed/阈值救跑。D11仅报告交互，不能用联合格涨分掩盖两个独立条件失败。两个channel draw不是两个modelseed，无跨seed证明、不自动启动训练或扩seed。

## 预算、权限与验证

0新epoch/训练更新/拟合头/目标访问/正式成绩读取。2模型各21602NN（2smoke+2draw×2urban×3新格×1800），共43204NN；D00共14400历史NN复用。7200个物理draw×4个受控renderer=28800调用，0新native ResidualChannel.process；原source缓存复用，没有新物理ID。父模型历史60epochs/13320更新和P33的216004NN分别记账，不冒充新训练或免费前向。CPU2/CUDA空、OMP/MKL/OPENBLAS各2，无GPU准入需求，不动其他owner任务。

5项针对性合成测试PASS（能量/相位/脉冲位置、源权限反例、四格符号、四单元门槛、完整collect路径）；6次合成native+6次原flat+24次新renderer，D00最大误差0、四格噪声差0，0真实IQ/CP/NN。local compile PASS，一次限定P0/P1静态审查PASS。一次N607普通账户/新root无冲突/原源输入句柄preflight PASS。真实CP smoke/完整源重放尚待launcher执行，不预填通过或科学结果。

唯一owner codex/root/delay-phase34。不可变run=/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-diagnostic-FcrossRX-delay-phase-e30-s2026092701-m2-r01；release=/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_delay_phase_E30_20261017_r01；launch日志=/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-diagnostic-FcrossRX-delay-phase-e30-s2026092701-m2-r01-launch；本地E:/type10-7/local_artifacts/cvs_FcrossRX_delay_phase_E30_20261017。CWD=/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_delay_phase_E30_20261017_r01，命令：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_delay_phase_E30_20261017_r01/experiments/cvs_delay_phase_attribution34/runner.py
```

源role/物理ID/顺序/CP/默认头/种子/hardware/D00重放/noise/energy/冻结状态/NN/finite或输出冲突使本任务停止并保留partial，无自动重试、旧权重/目标回退，不修改健康旧任务。一个archive做一次local/remoteSHA比较和remotecompile，启动后独立PID/start/CWD/argv/CUDA/log增长核实。

预期产物详见experiment.json：source roles/runtime/actual_engine/smoke、controlled_iq/control_records、observation_completion，逐model logits/complete，metrics/paired_effects/contrasts CSV、summary、source_delay_phase.png/.pdf和completion。此处仅为预期，不称产物已生成。

Git提交已生成，网络钩子超时后核实本地HEAD和无Git残留进程；未重复commit。直接GitHub TCP/TLS故障下保留本地commit，正常push和独立OID读回继续记录。REJECTED_EXTRA_GATE：brainstorming设计审批及通用全库测试前置不在项目最小白名单内，本次已授权源诊断按有界设计、聚焦测试、一次P0/P1执行，无新增审批/报告hash/memberhash链。

科学目标仍ACTIVE；此诊断不能替代clean跨RX正收益、formal low>60%、suburban保持及跨seed重现。

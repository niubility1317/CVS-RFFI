# 全源 E30 冻结默认头：practical 损伤归因

状态：REGISTERED，尚未运行。代码commit c3700b6636bb0ed32a8c7bdb00f8f94ef1a8f632，工作树E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014。一次限定P0/P1独立静态审查PASS，5项合成协议负测/配对数值/完整统计回归PASS，本地compile PASS；均未读真实IQ/CP/target。N607只读preflight确认普通账户、两个own FINAL30/6660来源、旧受控矩阵完整、新run/release/launch根不存在。Git提交保留，直接push暂受GitHub443连接故障影响，后续正常补推并独立核对，不重写历史。

问题：原生As适应到底降低了哪类practical残余损伤的默认头敏感性？不新增训练或泛化扰动，不救回周期重排路线。本次固定对照是P32匹配B00/B01；B10/B11均不参与，选取依据是预先定义的As因子，不是正式成绩排名。

## 输入、来源与范围

复用原P24合法源V1800（6TX×5原RX索引1/3/4/6/8×3day索引1/2/3，每格20）×2命名channel draw×3场景low_suburban/mid_urban/low_urban。完整原契约L6300/U56700hidden/V27000、split392005不变；不留出训练RX，不切L_dev，不解封U真值。读取P24既有observations/iq.npy/physics.npz及data_roles，所有role/ID/条件顺序匹配；不重建波形、不重做VALIDATED_ONCE验证链、不读目标IQ/truth/正式scores。

模型为20261017-phase1-FcrossRX-cycleorder-factorial-e30-s2026092701-m4-r01中B00/B01各自scratch E30/20L10U/6660更新、FINAL30固定终态的model state（非EMA）。两者原始全部L/U物理角色一致，无继承父/目标选模；P32 own_model先核实完整来源与source_contract角色再加载。固定F=SIG_TIME_X3+CENTER_25K/CENTER_UNITS、joint160、默认cos尺度30、900722参数/206state；default head与骨干训练方式不改，此次全部eval/frozen，无optimizer/autograd/拟合诊断头。

第一数据步骤是两真实CP严格加载及4合成包no-query smoke，0源IQ/0query；PASS立即继续。clean1800与full residual10800 logits/模型从同row已有source_qualification复用，核对配置/IDs/TX/RX/day与完整原源角色；新前向只10控制条件。

实际缓存As使用既有adv3b02_xuc/code/leo_practical residual/post_sync、EQoff(G1)、quality/phase/IQ补偿enabled、referenceSNR30、fs25MHz/fc2.45GHz、reflect、数值延迟24等原引擎，不改变测试或再施加信道。RX硬件由RX/day而非TX定义。新增噪声比不等于总SNR。

## 已固定矩阵与效应定义

11条件严格保留原顺序：full、no_noise/fixed、no_noise/total、flat_H/fixed、flat_H/total、no_carrier/fixed、no_carrier/total、no_RX_IQ/fixed、no_RX_IQ/total、actual_power_quality、edge_boundary。

fixed沿用full的lock、residual_scale、频率估计、相位跟踪、EQ参数、AGC；total按干预后重新计算状态。no_noise仅删除新增噪声，原录音噪声未知；flat_H保持原白输入核能量及数值延迟，删除频率选择性，不等于移除所有遮挡/衰落；no_carrier同时移除残余载波/相位相关项；no_RX_IQ仅移除注入RX IQ残差，不反演真实TX或原录音RX；actual_power_quality替换白输入功率代理；edge_boundary固定接收状态改人工padding，并非真实长上下文。

每模型每条件报告accuracy/macroF1/默认头margin(logits/30)/固定B00 clean正确集合保持率、同包修复与新增错误。主效应：D_m=accuracy_control−accuracy_full；As敏感性下降=D_B00−D_B01。正值表示As模型更少依赖该损伤的移除而获益；不能由正值断言所有损伤被消除，也不能把condition间非线性效应相加。

各scene/condition分别采用描述性重复标准：两个channel draw的敏感性下降均≥1pp，才称该源条件方向在两draw稳定。该标准不是训练候选晋级、不自动扩seed、不按结果补调参数。完整报告全部条件及负值。两个draw不是modelseed重现；sourceV的state/lock/质量区间不是正式target状态。

分层统一用full原始state/lock/新增噪声比/RX，不按干预后状态重新分组而混淆总效应。质量bin固定<0、0–3、3–5、5–10、10–15、≥15dB。汇总draw0/1及双draw合并，保存CM/原始logits/计数，不拟合头或强制特征一致。

## 预算、运行与停止规则

0epoch/0训练更新/0新native信道；每模型新增108002NN=2smoke+2draw×3scene×10condition×1800，两模型216004NN；复用clean/full25200历史NN。父模型已付60epochs/13320updates，本次不冒充重新训练或新增训练预算。CPU2/CUDA_VISIBLE_DEVICES空，OMP/MKL/OPENBLAS各2，不占GPU，不触碰其他owner进程。

唯一owner=codex/root/practical-attribution33；新独占run=/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-diagnostic-FcrossRX-practical-attribution-e30-s2026092701-m2-r01，release=/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_practical_attribution_E30_20261017_r01，launch日志=/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-diagnostic-FcrossRX-practical-attribution-e30-s2026092701-m2-r01-launch；本地E:/type10-7/local_artifacts/cvs_FcrossRX_practical_attribution_E30_20261017。CWD=/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_practical_attribution_E30_20261017_r01。命令：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_practical_attribution_E30_20261017_r01/experiments/cvs_practical_attribution33/runner.py
```

任何源role/IDs/条件顺序/CP来源/默认头/shape/finite/前向数/冻结状态不一致直接终止own任务、保留partial/failure，无自动重试、无目标或旧权重回退、不修改健康任务。发布一次archive local/remote SHA比较和remotecompile；启动后独立PID/start/CWD/argv/CPU/log增长核验。过程记录不作为新审批门槛。

预期实际产物：resolved_config/runtime/checkpoint_noquery_smoke/data_roles，逐模型predictions/complete，condition_metrics.csv、paired_effects.csv、sensitivity_reduction.csv、summary.json、source_impairment_effects.png/.pdf、completion.json。登记字段与完整逐row路径见experiment.json。此次只做已授权源诊断、没有新训练和正式测试重跑；原P32已经完成clean+6practical的全部truth-last。

科学结果尚未产生。即使某条件稳定也不等于双增强突破；全局目标仍ACTIVE。REJECTED_EXTRA_GATE：不添加全库测试、成员hash、报告审查或额外许可；5项针对性检查与一次P0/P1按最小流程执行。

## 实际启动

发布archive一次local/remoteSHA比较和remotecompile VERIFIED。CPU2任务PID633698/start148945274，CWD/argv/CUDA空独立核实；stdout由0增长117bytes，stderr0，真实2CP/4synthetic no-query smoke PASS且此前源IQ0；B00已完成首scene控制读出18002NN，failure为空。该进度不代表归因结果。

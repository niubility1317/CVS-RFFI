# 全源 E30 practical 损伤归因：冻结默认头结果

状态：**ANALYZED，source-only**。两份 matched E30 模型自然完成，0训练更新、0拟合头、0目标数据/正式成绩读取、0新增信道生成。保留原完整L6300/U56700hidden/V27000及全部5个源RX，不留训练域。此次源诊断不等于新的正式测试，原P32两模型均已完成clean+6practical的truth-last。

结论：As对完整practical观测的总体收益成立，但本次没有一项噪声/频率选择性/残余载波相位/注入RX-IQ控制，在两个urban、两个channel draw中证明至少1pp的敏感性下降。最明确的剩余损失来自新增噪声：固定接收状态移除后，As模型mid/low源准确率可再升5.694444/10.833333pp。不能声称As已解决噪声，亦不能把总体收益唯一归于多径/CFO/门控。

## 固定来源和已执行设计

代码commit c3700b6636bb0ed32a8c7bdb00f8f94ef1a8f632，起点618b9b5912f28de5650049c493a8e8a38484bba4；启动登记commit b909fc04045198b07bc8c5413421cdae6b161077。两源CP来自20261017-phase1-FcrossRX-cycleorder-factorial-e30-s2026092701-m4-r01的B00/B01，各自scratch E30=20L10U/6660更新、FINAL30固定model state，非EMA；无上游权重、无目标训练/选模，完整原source角色严格匹配。F=SIG_TIME_X3+CENTER_25K/CENTER_UNITS、joint160/defaultcos30、900722参数/206state。骨干/默认头训练配方不改变，此次全部eval/frozen且参数state逐tensor不变。

第一步两真实CP严格加载和4合成包no-query smoke PASS，此前源IQ0/target0，PASS即继续。P32同row source_qualification的clean1800与full10800 logits直接复用；源ID/TX/RX/day/配置一致，未重新做完整view前向。只对既有P24 V1800×2draw×3scene×10控制条件做新前向，完整11条件原样保留。V面板6TX×5源RX索引1/3/4/6/8×3day1/2/3，每格20包；原role与ID句柄复用VALIDATED_ONCE，无U真值或target访问。

既有缓存来自实际adv3b02_xuc/code/leo_practical residual/post_sync、EQoff/G1、refSNR30/fs25MHz/fc2.45GHz/reflect/数值延迟24，quality/phase/IQ补偿enabled；接收状态/补偿/噪声/AGC依赖与原工程参数保持，未弱化测试或生成测试新IQ。RX硬件按RX/day session稳定，非TX定义。

fixed控制沿用原lock/residual_scale/频率估计/相位跟踪/EQ/AGC；total重新计算接收状态。no_noise只移除新增噪声；flat_H保持白输入核能量与数值延迟而去除频率选择性，仍含衰落与全局复增益；no_carrier移除残余载波/相位相关项；no_RX_IQ仅移除注入RX残差；actual_power_quality替换白输入功率代理；edge_boundary是人工padding变化，**不是移除损伤**或真实长上下文。

预注册原文保留preregistration.md。无数据驱动改参或自动训练。源效应D_m=accuracy_control−accuracy_full，交互I=D_B00−D_B01；按原full的state/lock/RX/新增噪声区间分层，不按干预后的状态重分组。margin统一logits/30，保持率基准为固定matchedB00 clean正确集合（1771包），不重训分类头。所有CM、准确率/F1/间隔/保持率/同包修复与破坏计数保留。

## 实际控制结果

完整源view（两个draw合并）B00/B01：low_suburban54.750000%/90.777778%，mid_urban45.861111%/75.861111%，low_urban38.555556%/62.138889%。这些是源V，不是正式目标low>60%。As在no_noise/fixed后mid/low为81.555556%/72.972222%；no_noise/total后为81.555556%/74.250000%。low total相对fixed仅再增加1.277778pp，接收状态重算效应不能与直接去噪混为一谈。

|源scene|control|B00的D pp|B01的D pp|I pp|I draw0 / draw1 pp|
|---|---|---:|---:|---:|---|
|practical_low_suburban|no_noise/fixed|1.527778|2.833333|-1.305556|-1.555556 / -1.055556|
|practical_low_suburban|no_noise/total|2.111111|2.888889|-0.777778|-1.055556 / -0.500000|
|practical_low_suburban|flat_H/fixed|1.166667|0.527778|0.638889|-0.333333 / 1.611111|
|practical_low_suburban|flat_H/total|1.388889|0.666667|0.722222|0.000000 / 1.444444|
|practical_low_suburban|no_carrier/fixed|0.083333|-0.222222|0.305556|0.277778 / 0.333333|
|practical_low_suburban|no_carrier/total|0.083333|-0.222222|0.305556|0.277778 / 0.333333|
|practical_low_suburban|no_RX_IQ/fixed|-0.055556|0.000000|-0.055556|-0.166667 / 0.055556|
|practical_low_suburban|no_RX_IQ/total|-0.055556|0.000000|-0.055556|-0.166667 / 0.055556|
|practical_low_suburban|actual_power_quality|0.000000|-0.055556|0.055556|0.055556 / 0.055556|
|practical_low_suburban|edge_boundary|-11.472222|-11.750000|0.277778|-0.777778 / 1.333333|
|practical_mid_urban|no_noise/fixed|3.722222|5.694444|-1.972222|-1.500000 / -2.444444|
|practical_mid_urban|no_noise/total|3.888889|5.694444|-1.805556|-0.944444 / -2.666667|
|practical_mid_urban|flat_H/fixed|3.277778|3.055556|0.222222|0.555556 / -0.111111|
|practical_mid_urban|flat_H/total|3.333333|3.305556|0.027778|0.111111 / -0.055556|
|practical_mid_urban|no_carrier/fixed|0.166667|0.250000|-0.083333|-0.055556 / -0.111111|
|practical_mid_urban|no_carrier/total|0.166667|0.250000|-0.083333|-0.055556 / -0.111111|
|practical_mid_urban|no_RX_IQ/fixed|0.055556|0.055556|0.000000|-0.166667 / 0.166667|
|practical_mid_urban|no_RX_IQ/total|0.055556|0.055556|0.000000|-0.166667 / 0.166667|
|practical_mid_urban|actual_power_quality|-0.111111|0.055556|-0.166667|-0.277778 / -0.055556|
|practical_mid_urban|edge_boundary|-7.972222|-9.944444|1.972222|1.722222 / 2.222222|
|practical_low_urban|no_noise/fixed|4.888889|10.833333|-5.944444|-6.055556 / -5.833333|
|practical_low_urban|no_noise/total|6.250000|12.111111|-5.861111|-5.833333 / -5.888889|
|practical_low_urban|flat_H/fixed|1.916667|2.194444|-0.277778|0.611111 / -1.166667|
|practical_low_urban|flat_H/total|1.722222|1.888889|-0.166667|0.722222 / -1.055556|
|practical_low_urban|no_carrier/fixed|-0.250000|0.166667|-0.416667|-0.500000 / -0.333333|
|practical_low_urban|no_carrier/total|-0.250000|0.166667|-0.416667|-0.555556 / -0.277778|
|practical_low_urban|no_RX_IQ/fixed|-0.027778|0.055556|-0.083333|-0.055556 / -0.111111|
|practical_low_urban|no_RX_IQ/total|-0.027778|0.083333|-0.111111|-0.111111 / -0.111111|
|practical_low_urban|actual_power_quality|-0.138889|0.083333|-0.222222|-0.111111 / -0.333333|
|practical_low_urban|edge_boundary|-6.083333|-8.027778|1.944444|2.111111 / 1.777778|

正I仅在“移除同一损伤且D为改善”的语境可解读为敏感性降低。**edge_boundary列的正I不是鲁棒性提升**：mid的B00/B01各下降7.972222/9.944444pp，low各下降6.083333/8.027778pp，即As对这个额外边界改变损失更多。其正I在两draw超过1pp的纯数学事实保留，但不得称科学通过或据此晋级；原图纵轴和CSV列沿用预注册的代数名称，解释以该符号定义为准。

noise固定控制B00的mid/low改善3.722222/4.888889pp，As模型改善5.694444/10.833333pp。As模型有更多可恢复识别能力，去噪能进一步释放它；I为负不直接证明训练降低了绝对噪声鲁棒性，因模型基准水平、非线性交互与饱和均改变。至少它否定“主要收益等于消除了新增噪声敏感性”的强结论。

flat_H/fixed的I mid+.222222pp、low−.277778pp且跨draw符号不稳定，没有频率选择性损伤已被稳定消除的证据。载波/相位和注入RX-IQ移除对As模型mid/low均≤.25pp；提高这类损伤强度不是本次数据支持的主方向。实际功率替代白输入代理对As mid/low仅+.055556/+.083333pp，不能把大幅urban失败归为质量阈值代理误差。signal_to_added_noise_db仍不是总SNR。

主四类移除控制没有达到预定双draw各≥1pp的描述性下降条件；两个draw不是modelseed重现。仅edge边界交互出现该数值条件，方向表示额外伤害增加，不构成成功候选。完整源分层/小样本计数均保留CSV，不以单个小格替代总体。

## 验证、成本和真实产物

5项合成测试PASS：源role/heldRX/Utruth/target/条件顺序负测、6类metric/尺度/非有限、同包修复符号、双draw条件、完整collect路径与聚合。local/remotecompile PASS；一次限定P0/P1独立静态审查PASS；一次archive local/remoteSHA比较VERIFIED；真实2CP smoke PASS。PID633698/start148945274、CWD/argv/CUDA空独立核实，stdout0→117→774bytes、stderr0，最终自然退出、failure为空。无健康旧任务停止/重启/热改。

新增216004NN=各108002（各2smoke+108000控制）；复用历史clean/full25200NN，不新增native channel calls/物理包/epoch/更新。父模型历史60epochs/13320updates分别披露，非本次训练。CPU2/CUDA空，实际总墙钟143.663103s，逐模型控制读出68.931636/70.690842s。保存3520条分层metric、3200条同包效应、1600条交互。冻结state无变化/无.grad，无技术失败或结果重跑。

实际命令/CWD/owner/停止规则见experiment.json。命令为：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_practical_attribution_E30_20261017_r01/experiments/cvs_practical_attribution33/runner.py
```

remote run：/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-diagnostic-FcrossRX-practical-attribution-e30-s2026092701-m2-r01，release：/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_practical_attribution_E30_20261017_r01，launch日志：/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-diagnostic-FcrossRX-practical-attribution-e30-s2026092701-m2-r01-launch。本地：[完整统计](E:/type10-7/local_artifacts/cvs_FcrossRX_practical_attribution_E30_20261017/results_r01/summary.json)、[分层指标](E:/type10-7/local_artifacts/cvs_FcrossRX_practical_attribution_E30_20261017/results_r01/condition_metrics.csv)、[同包效应](E:/type10-7/local_artifacts/cvs_FcrossRX_practical_attribution_E30_20261017/results_r01/paired_effects.csv)、[交互逐draw](E:/type10-7/local_artifacts/cvs_FcrossRX_practical_attribution_E30_20261017/results_r01/sensitivity_reduction.csv)、[完成证据](E:/type10-7/local_artifacts/cvs_FcrossRX_practical_attribution_E30_20261017/results_r01/completion.json)、[实际源图PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_practical_attribution_E30_20261017/results_r01/source_impairment_effects.png) / [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_practical_attribution_E30_20261017/results_r01/source_impairment_effects.pdf)。PNG已实际查看，全部逐模型logits/元信息与原日志保留；IQ/CP未复制进Git。

实现文件experiments/cvs_practical_attribution33/{design,run,runner,stats,checks,publish}.py。启动登记时Git code已本地提交；直连GitHub443连接/重置曾导致push失败，超时后本地Git helper已退出，独立API确认当时远端仍为618b9b59。保留本地commit，以后续正常push与独立remote OID读回记录最终交付状态，不force/rewrite。REJECTED_EXTRA_GATE：不新增报告审查、全库测试、memberhash或许可链。

## 下一项可证伪问题与限制

As巨大总体收益在单独删除噪声/选择性/载波/RX-IQ后仍存在。固定数值FIR延迟24与flat_H保留的全局复增益/相位尚未独立拆开；人工edge控制仍明显有害。下一项源假设应定位**固定滤波延迟/有效窗口和全局复相位**是否解释适应收益，而非立即增加噪声、Doppler或新一致性损失。若严格同latent/固定状态控制不能显示这两类因素的独立贡献，就放弃该解释。

该后续控制尚未实现/启动；不得把源端人工counterfactual改进当成正式测试提升，不改原测试IQ、SNR或边界。当前正式joint仍clean83.182738%/low53.339881%，只有1seed，全局目标仍ACTIVE；未证明clean双赢或跨seed重现。

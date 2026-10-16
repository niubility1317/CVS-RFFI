# F/E20：有界观察原点抖动的源 RX1 结果

源规则 **FAIL**，0 目标访问；一个新 scratch E20，A0 结果复用，未加载其权重。

|模型|held RX1 准确率|宏平均 F1|seen V 均值|seen V 最差 RX|
|---|---:|---:|---:|---:|
|A0|81.148148%|80.603394%|98.041667%|95.537037%|
|OJ24|79.240741%|79.420087%|97.856481%|95.185185%|

OJ24−A0 准确率 -1.907407 个百分点，宏平均 F1 -1.183307 个百分点；修复 247／新增错误 350；seen V 损失 +0.185185 个百分点。

新 20 epoch／4440 step／native CSV／完整 stdout 与复用 A0 日志完整解析，全部 5994 个 L/U 批次逐条相同。25 种偏移全部覆盖，568320 个实际新视图。原 ordinary／As 调用均为 0。

新成本 20 模型轮／4440 更新／3546618 真实网络前向，墙钟 1102.234 s；复用控制 20 模型轮／4440 更新／3546618 前向另列。本地合成检查另计。

一个已使用过的源 RX 折、一个模型 seed，不能替代完整 DG、正式 clean/practical 或跨 seed 证据。失败关闭 OJ24，不扫描偏移范围、概率或更换 RX/seed 救援；通过只支持另登记固定其他 4 个源 RX，不直接进入 target 或 As。

OJ24 是 causal-reflect 观察原点变化；前 d 点是人工反射过去，末 d 点被截去，不称为真实长记录、未来补齐或循环平移。无后归一化，窗口功率可以改变。所有合法有标签视图均参与原默认头监督，F 与头联合训练。完整曲线和逐 TX/day 正负结果保存，预算耗尽不等于收敛。

实际运行与科学状态：**VERIFIED / FAIL_SOURCE_RX1_ONLY**。实际训练代码 `f601fc65f6eaafbfd98080882108f4d50e07db69`；worker 和 dispatcher 自然退出，无 failure。单 archive transfer／remotecompile、真实 own scratch checkpoint 无查询 smoke 及独立 PID/CWD/argv/GPU/log 增长证据保留。没有热改、重启原 release 或触碰其他健康任务。

模型保持 F 的三路 SIG_TIME/CENTER_25K/CENTER_UNITS、25 MHz／IQ256／equalized1／原 native RMS／NoExplicitCFO／joint160／默认 cosine30。默认头、三路和 Sinc 均按原流程实际联合更新，无额外头拟合。0.5 RD(raw)＋0.5 RD(OJ24)，每个 RD 的两次前向复用同一 IQ；无 raw/G 一致性，无新增部署参数。原 U、own EMA、旧门控及 strong_noise_std=0.015 保持，U TX 始终隐藏。

本候选仅作用于 IQ 的观察原点。每包 d∼Uniform{0,…,24}，G[n]=x[abs(n−d)]；前 d 点为反射过去、末 d 点被截去，至少保留 232 个原观察位置，不能称为完整长记录或未来补齐。无循环 roll、插值、PA、DAC、CFO、公共相位、AWGN、多径或 As。native aug 时间移位字段只用作工厂开关，实际调用自有算子，原 ordinary 工厂／schedule 未执行。窗口能量可以变化，无后归一化。

实际 25 个偏移全部覆盖：最少 22350、最多 23080 次，总计 568320 个生成视图；输出／输入窗口能量比范围 0.859006794 至 1.287734495。波形 MSE 为 0.965749485，它是执行统计，不作为有效性标准。

有效训练角色为源 RX3/4/6/8 的 L5040、隐藏标签 U45360、seen V21600；held RX1 的 L/U 不参与训练，held V5400 仅 FINAL。全部 6TX×4RX×3day 训练覆盖，以及 5994 条实际 L/U 批次物理顺序与 F A0 完全相同。仅复用匹配 A0 的结果／配置／trace，没有加载其权重、EMA、optimizer 或统计。遗留描述字段含旧 updates／阶段轮数／E50 文案；实际 nativeargs20/13/7 和 4440 成功更新为执行预算，原发布配置保持原状。

完整解析两组各 20 epoch／4440 step／native CSV／完整 stdout。全部关键 loss／gradient 有限，无非有限跳步或 U 真值访问；F 的 4440 次物理单位 clip／lr 映射逐步核对。准确率、F1、混淆、逐 TX/day 及纠错由保存的 5400×2 默认头决策独立复算；CE 沿用执行器记录。

匹配 A0 原 clean 正确集合 4382 包中，OJ24 保留 4032 包（92.012780%）。默认头归一化间隔 mean A0=0.117288660、OJ24=0.103003497，定义为（真类 logit−最强错类 logit）/30。该保持率针对源 clean，不冒充 satellite 保持率，间隔也不替代识别收益。

|源分层|A0 准确率|OJ24 准确率|OJ24−A0／百分点|
|---|---:|---:|---:|
|TX0|96.2222%|71.6667%|-24.5556|
|TX1|86.0000%|87.0000%|+1.0000|
|TX2|56.7778%|73.0000%|+16.2222|
|TX3|65.5556%|62.7778%|-2.7778|
|TX4|100.0000%|100.0000%|+0.0000|
|TX5|82.3333%|81.0000%|-1.3333|
|day1|80.0556%|79.2222%|-0.8333|
|day2|88.2222%|89.7778%|+1.5556|
|day3|75.1667%|68.7222%|-6.4444|

|成本|复用 A0|新 OJ24|
|---|---:|---:|
|L_forward_packets|2273280|2273280|
|student_U_forward_packets|396368|396368|
|student_source_eval_forward_packets|475202|475202|
|EMA_eval_forward_packets|396368|396368|
|held_V_forward_packets|5400|5400|

新增成本 E20／4440 更新／3546618 真实网络包，矩阵墙钟 1102.233619 s；复用控制 E20／4440 更新／3546618 网络包另计，0 父权重训练成本。源 eval 中含 own scratch checkpoint smoke2。本地累计 512 个合成网络包、896 个合法 jitter 输入、50 个显式公式输入、1 次合成 optimizer 更新、13 个负例另计；独立检查额外网络前向为 0。CPU 分析／绘图未统一计时，E20 用尽不等于收敛。

一个已使用过的源 RX 折、一个模型 seed；F 由用户在已暴露历史 A–F 基准后指定，不声称新独立盲测。FAIL 关闭 OJ24，不扫偏移范围／概率或换 RX/seed 救援；PASS 仅支持另登记固定其余 4 个源 RX 确认，不自动访问 target 或训练联合 As。单折训练收益仍不能区分原点变化、人工边界与窗口能量变化的贡献。

当前没有该模型的正式 clean／六 practical 成绩，不能回答 urban 哪种退化已经缓解、正式默认头提高多少或 clean 付出多大代价；跨 seed 也没有证明。low urban >60%、clean 净收益和 suburban 接近 clean 的联合目标继续 active。训练后的源域最终评估已按本项明确 source-only 范围自动完成；不冒充完整正式测试。

复现入口、命令、环境、CWD、路径和停止规则见 experiment.json；已完成 run 根不可重复启动。离线复算使用 `F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_origin_jitter20.harvest --output E:/type10-7/local_artifacts/cvs_origin_jitter20_20261016/final --state E:/type10-7/local_artifacts/cvs_origin_jitter20_20261016/terminal_readback.json --control-root E:/type10-7/local_artifacts/cvs_F_raw_anchor20_20261016/final/A0-RX1-s2026092701/source --cached`，不读 IQ 或 checkpoint。

实际输出：

- [完整训练曲线](E:/type10-7/local_artifacts/cvs_origin_jitter20_20261016/final/complete_training_curves.png)
- [默认头逐 TX/day 分层](E:/type10-7/local_artifacts/cvs_origin_jitter20_20261016/final/held_source_strata.png)
- [偏移覆盖／能量／前向成本](E:/type10-7/local_artifacts/cvs_origin_jitter20_20261016/final/origin_and_cost.png)
- [全部分层数值](E:/type10-7/local_artifacts/cvs_origin_jitter20_20261016/final/source_strata.csv)
- [完整曲线数值](E:/type10-7/local_artifacts/cvs_origin_jitter20_20261016/final/training_curves.csv)

## 负结果解释与后续基线覆盖

三张实际图已逐图查看。OJ24 的 CE/KL 与物理梯度仍有限但高于匹配 A0；E14 按计划启用 U，学习率轨迹一致，seen V 小幅下降。held CE 从 0.824229 降至 0.785186，准确率却下降，不能用损失下降代替身份识别收益。TX2 提高 16.2222 个百分点，但 TX0 下降 24.5556 个百分点；day2 +1.5556，day1 −0.8333，day3 −6.4444 个百分点，错误重分配没有形成总体收益。保持全部分层，不以有利类别救援，也不由曲线平台宣称已充分收敛。

冻结延迟诊断的 32.54 个百分点损失支持输入脆弱性，却没有转化成该固定半 raw／OJ24 训练的跨 RX 正收益。它不证明所有时序增强无效，也不证明损失唯一来自反射前缀、窗口截取、能量或某分支；这些因素在本算子中共同变化。按原规则关闭 OJ24，不扫范围／概率或换 RX/seed，不据更换底座重跑救援。

用户最新将增强前基线改为 F＋跨 RX 同 TX 对比学习，E20 保持。该指定覆盖后续研究，不改变本行旧 F 的实际运行、对照、负判定或成本；本结果不能作为新基线增益或新对照。新契约为 configs/phase1/F_crossrx_e20_baseline.json，当前尚未发布新的 E20 基线。正式 clean／六 practical／跨 seed 联合目标继续 active。

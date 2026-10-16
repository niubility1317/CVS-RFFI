# F 后续 IQ 路线：包内缓变增益的源域证伪

当前 PLANNED，尚无真实源数据结果。本轮只检验固定参考窗内的缓变功率包络能否提供跨 TX/day 的 RX 观测变化依据；不把统计通过写成增强有效。

- Run：`20261016-diagnostic-F-gainenvelope-source-crossfit-manysig-m1-r01`。
- 配置与唯一矩阵：[experiment.json](experiment.json)；状态：[events.jsonl](events.jsonl)。
- 实际执行 commit 由发布包 `release_commit.txt` 与远端 `resolved_config.json` 记录。
- CPU 2 线程；0 个模型、0 个 checkpoint、0 次网络前向、0 次更新、0 个 target。

## 依据与边界

固定网络仍为 F（三路 SIG_TIME＋统一 Sinc 修正）、真实 joint160、默认 cosine 头联合训练、NoExplicitCFO。当前不修改网络或输入。本候选区别于另一对话“开发 CVS Phase1 星地增强训练”的公共相位、观测起点及 OJ24，也区别于已关闭的分数延迟和 DC 路线。

接收 AGC 的增益可能在前导期间调整，[ADI 的 AD936x 说明](https://wiki.analog.com/resources/eval/user-guides/ad9361_agc_tuning)讨论了前导确定增益后锁定的处理方式。这只提供物理候选依据，不能证明 ManySig 使用相同硬件或存在同样过程。观测斜率也可能包含 TX 瞬态、传播及波形结构。

已核查普通增强中的增益为整包常数；另一 receiverDG 的 AGC 操作还包含软剪裁。本候选只测周期功率斜率，不把上述实现的名称当作启用证据，不调用其非线性处理。整包常数增益会被 RMS 消除，而中心化对数功率斜率在数学上保留；仍用实际数据读回验证数值性质。

## 固定数据与检验

复用已有合法缓存和原物理角色，原 L6300 中仅 RX3/4/6/8 的 L5040 参与统计；6TX×4RX×3day，每格 70 包。RX1 的 1260 包在转换或统计 IQ 前排除。旧缓存会反序列化未使用的其他角色及旧特征，明确披露；不读取 U 真值，不使用 U/V 统计，不继承参数。

固定 `[80,160)` 中四个 20 样本周期，计算各周期 I/Q 总功率的 dB 值，去除四周期均值，在 `q=(-1,-1/3,1/3,1)` 上求最小二乘斜率。该斜率描述周期中心的半摆幅，不等于整个 256 点窗口的最大增益。主统计是每格绝对斜率的中位数，保留符号斜率作完整诊断，不在结果后换主指标。

18 折同时留出一个 TX 和一个 day。RX 均值及共同均值都只由其他 TX、其他 day 估计，比较四 RX 留出格的等权平方预测风险。保留 3 种循环错配 RX 对照，不拟合身份分类器。

以下条件必须全部通过：

1. 共同均值风险非零，RX 风险相对降低至少 5%。
2. 至少 14/18 折改善，4 个 RX 风险增益均不为负。
3. 比最好的错配 RX 增益至少高 2 个百分点。
4. 两组 TX 的中心化 RX 统计余弦至少 0.8。
5. 5040 包线性拟合 R² 中位数至少 0.8，绝对斜率中位数至少 0.01 dB。
6. 对实际包重新 RMS 归一化，最大斜率差不超过 `1e-10` dB。

零周期功率、缺格、非有限值或角色不匹配均保留失败，不删除困难包。科学失败即关闭此固定包络依据，不改窗口、统计量、seed、RX 或强度补救。通过只支持另行固定有界正增益视图与匹配 F/E20 默认头身份实验，没有自动训练或 target 访问。

## 发布、停止与产物

唯一 owner：`codex/root/F-gain-envelope-20261016`。普通 N607 账号、现有 CVS-RFFI Python，CUDA 隐藏，CPU 2 线程。精确 CWD、命令、路径见登记配置。发布入口 `python -m experiments.cvs_gain_envelope16.publish`；远端 `python -u -m experiments.cvs_gain_envelope16.run`。

不可覆盖输出根：`PROJECT/runs/20261016-diagnostic-F-gainenvelope-source-crossfit-manysig-m1-r01`；发布根 `PROJECT/releases/cvs_F_gain_envelope_source_20261016_r01`。技术错误保存 failure 后停止，不自动重试，不修改任何健康旧任务。复用已有匹配源数据，不新增数据验证链；无模型或 checkpoint，真实 checkpoint smoke 不适用。

已有合成检查：RX 稳定斜率通过，纯 TX 与 day 反转失败；解析斜率及 RMS 不变性通过；7 个非法输入拒绝检查通过。一次独立 P0/P1 审查 PASS。发布使用一次归档传输比对、一次远端编译和独立产物读回。

预期产物包括实际配置、完整 72 格及 18 折、3 个错配对照、5040 包的周期统计、私有物理 ID、完整 stdout、completion、独立重算和同 row 报告。IQ、ID、运行日志及机器状态不进入 Git。

## 联合目标未完成

本轮没有 A_g 识别收益、A_s 训练或正式目标评分。即使统计通过，仍须分别证明 A_g 的跨 RX clean 收益、A_s 的 practical 收益，再做匹配 B00/B10/B01/B11、交互项及独立组合视图检验。低 urban >60% 与 suburban 接近 clean 均不是当前结果。

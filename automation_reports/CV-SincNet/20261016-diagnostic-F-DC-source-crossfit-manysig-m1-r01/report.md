# F 后续 IQ 路线：DC 观测变化的源域证伪

本轮先回答一个问题：固定参考窗的复 DC 是否包含能够跨 TX、跨 day 重复的源 RX 差异。尚未启动真实数据分析；不把源统计通过写成识别收益。

- Run：`20261016-diagnostic-F-DC-source-crossfit-manysig-m1-r01`。
- 代码身份：发布时实际 Git commit，保存在 `release_commit.txt` 和 `resolved_config.json`；不增加代码哈希层。
- 配置：[experiment.json](experiment.json)；状态：[events.jsonl](events.jsonl)。
- 当前预算：CPU 2 线程，5040 个原始合法 L 包，0 个网络、0 次模型前向、0 次更新、0 个 checkpoint、0 个 target。后续训练仍使用固定 F / E20。

## 为什么检验 DC

直接变频 RX 的本振自混频可以形成 DC，见 [ADI 的接收机说明](https://www.analog.com/en/resources/design-notes/optimizing-performance-wideband-direct-conversion-receivers.html)。实际观测的 DC 也可能包含 TX 本振泄漏与 RX 本身的 DC，见 [ADI 的泄漏测量讨论](https://www.analog.com/en/resources/analog-dialogue/articles/transmit-lo-leakage-lol-an-issue-of-zero-if-that-isn-t-making-people-laugh-out-loud.html)。有限窗口、残余 CFO 和既有预处理也可能影响均值。因此硬件原理只提供待检验的依据，不能证明 ManySig 的 DC 是非身份因素，更不能证明加 DC 保持标签。

本路线区别于另一对话“开发 CVS Phase1 星地增强训练”正在研究的公共相位、观测起点和 OJ24。此前本对话的 ±0.125 样本非循环分数延迟已在源 RX1 上失败：默认头准确率变化 −0.555556 个百分点，macro-F1 变化 −0.635343 个百分点。结果保留，不改幅度或挑 RX 补救。

## 固定数据与统计

复用已有 `VALIDATED_ONCE` 源缓存及原始物理角色。原 L6300 中仅 RX3/4/6/8 的 5040 包参与统计；6 个 TX × 4 个 RX × 3 个 day，每格 70 包。RX1 的 1260 包在转换或统计 IQ 前排除。旧缓存会反序列化未使用的其他角色及旧特征，明确披露；不读取 U 真值、不使用 U/V 特征、不继承参数。

对未修改、原有 RMS 处理后的 IQ256，只计算预先固定的 `[80,160)` 复均值。四个 20 样本周期另给出奇偶周期 DC 差，用于检查短窗均值是否稳定。不新估 CFO，不修改内部补偿，不更换参考窗。

18 个留出折同时排除一个 TX 和一个 day。训练格分别估计 RX 均值和共同均值，在 4 个 RX 的留出格上比较 DC 二维平方预测风险。另保留 3 种循环错配 RX 对照。没有拟合身份分类器。

唯一预登记通过条件必须全部成立：

1. 共同均值风险非零；RX 均值相对共同均值总风险降低至少 5%。
2. 至少 14/18 折风险改善，且 4 个 RX 的风险增益均不为负。
3. 比最好的错配 RX 对照多改善至少 2 个百分点。
4. 两组 TX 得到的 RX 中心化复偏移余弦至少 0.8。
5. 包均值 DC 功率 / 奇偶周期差功率至少为 2。

基准风险为零时，增益记为未定义且判失败。失败即关闭本固定 DC 假设，不改强度、窗口、seed 或 RX 补救。通过只支持下一步单独固定有界 DC 输入检查及 F/E20 训练；不能据此声称 A_g 有效。

## 执行与停止

唯一 owner：`codex/root/F-DC-source-20261016`。N607 普通账号，现有 CVS-RFFI Python，CUDA 隐藏，CPU 2 线程。环境、CWD、远程路径及命令见 `experiment.json`；发布入口为 `python -m experiments.cvs_dc_source16.publish`，远端执行 `python -u -m experiments.cvs_dc_source16.run`。

输出根不可覆盖：`PROJECT/runs/20261016-diagnostic-F-DC-source-crossfit-manysig-m1-r01`。发布根：`PROJECT/releases/cvs_F_DC_source_crossfit_20261016_r01`。协议或矩阵错误、非有限值、缺缓存及运行异常保留产物后停止，不自动重试，不修改健康旧任务。本阶段不构造模型、不消费 checkpoint，真实 checkpoint smoke 不适用。只执行一次范围内独立 P0/P1 审查、一次发布包传输比对及远端编译。

预期产物：实际配置、72 格与 18 折完整统计、全部对照、私有物理 ID、stdout、completion、独立重算、结果报告和登记。原始 IQ、物理 ID、日志及运行状态不进入 Git。

## 当前验证与联合目标边界

合成的 RX 稳定偏移通过，纯 TX 偏移和 day 反转均失败；零风险返回未定义而非虚假高分；7 个非法输入拒绝检查通过。一次独立 P0/P1 审查为 PASS。尚无本轮 DC 真实结果，没有新增 A_g 训练、A_s 训练及正式目标评分。

本轮只检验 A_g 的候选依据。联合目标仍要求固定 F、真实 joint160、默认 cosine 头联合训练、NoExplicitCFO、原 U 规则，按匹配 B00/B10/B01/B11 判断 clean、low urban、suburban 和交互项。不得用本次统计替代这些证据。

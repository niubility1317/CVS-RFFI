# 原完整 practical E50：三种子匹配验证已完成

状态：**ANALYZED／VERIFIED**。四个新增 row 已完成训练、源域读出、clean＋六种 practical 预测和独立 truth-last 评分；与既有 seed01 E50 和六个 E30 历史条件合并后，三个固定模型种子、12 个逻辑条件全部完整。无技术缺项。

**E50 对 low／mid urban 的增益可在三个固定种子复现，但 clean 未达到阶段要求，尚无可重复双提升证据。** 相对各自 As30，As50 的 low／mid urban 平均增加 1.156746 ± 0.092682／0.954167 ± 0.156568 个百分点，clean 平均变化 -0.017659 ± 0.296162 个百分点。相对匹配 NoAs50，clean 为 -0.150198 ± 0.955701 个百分点，均值和全种子口径都未满足下降不超过 0.1 个绝对百分点。

01／02／03 的匹配 clean 变化分别为 +0.365476／+0.436905／−1.252976 pp，前两个通过、第三个失败。全部固定种子保留，没有剔除第三个种子、补选种子或选择性重跑。E50 只指总预算 50 轮，轨迹为 20＋10＋20、追加段重置 AdamW／cosine20，不是连续 cosine50。

本轮证实“更多预算可进一步改善完整 As 的 urban 识别”，否定“单靠此固定预算扩展已解决 clean 代价”的判断。NoAs 自身延长训练使 urban 均值下降 1.943320 pp，因此不能把 As50−NoAs50 的全部差距归因于多训 20 轮。

|视图|NoAs50 准确率 %|As50 准确率 %|As50−NoAs50 pp（配对均值±样本 SD）|As50−As30 pp|As50 宏 F1 %|As50 最差 RX %|
|---|---:|---:|---:|---:|---:|---:|
|clean|84.714087|84.563889|-0.150198 ± 0.955701|-0.017659 ± 0.296162|84.214124|77.687500|
|practical_high|47.103770|80.764286|33.660516 ± 8.238954|0.437500 ± 0.463044|80.492918|72.925000|
|practical_mid|45.710119|78.223016|32.512897 ± 7.553454|0.587698 ± 0.372703|77.979399|70.998611|
|practical_low_suburban|44.131746|75.784524|31.652778 ± 6.656782|0.699603 ± 0.207028|75.554855|68.918056|
|practical_high_urban|45.565675|77.272421|31.706746 ± 7.664944|0.575992 ± 0.447469|77.032625|70.173611|
|practical_mid_urban|38.417460|62.989484|24.572024 ± 5.066220|0.954167 ± 0.156568|62.754430|57.552778|
|practical_low_urban|32.764683|51.926587|19.161905 ± 3.110887|1.156746 ± 0.092682|51.537900|47.713889|

urban 三场均值：NoAs50 38.915939%，As50 64.062831%，匹配增益 25.146892 ± 5.278997 pp；As50−As30 0.895635 ± 0.219271 pp。表中最差 RX 为每个种子自身最差 RX 的均值，不是所有种子最小值；完整逐种子值见下文。标准差 ddof=1，非置信区间。

完整逐种子准确率、宏 F1、最差 RX、RX／day／TX 分层、修复与新增错误、源域口径和成本见 [final_analysis.md](final_analysis.md)，机器可读摘要见 [results.json](results.json)。三种子 clean 为 +0.365476／+0.436905／−1.252976 pp，保留全部结果。

## 预固定假设、选择依据与范围

唯一检验是原配方预算收益在原定 02／03 的复制；不是新增增强、损失、门控、分支或头。选择 E50 的依据是 01 合法源 full V clean 预算 +0.118519 pp、源 mid／low +1.972222／+1.638889 pp，未用正式成绩选候选或种子。原 01 不重训不重测。新四行 02／03 各 NoAs／完整 As，从各自合法 FINAL30 student／own EMA 追加 20 轮。原 parent release `cvs_FcrossRX_As_replication_E30_s0203_20261018_r02`，提交 `63b7c8cd7bb4139a2d800e6f4c4a7f1f2163fd47`；固定 P43 祖先 `19aedc3604fed9c193237dd92366164178af0983`。直接父本地 epoch10／EMA2220、own scratch20 祖先 epoch20／EMA4440；累计 E30 不冒充 checkpoint 本地计数。

四行矩阵、准确命令、环境、GPU、输出与停止规则由 experiment.json 唯一维护。固定 FINAL50，不按源或正式成绩选 epoch、seed 或跳过评分；保留每 seed 和均值的 −0.1 绝对百分点阶段判据，以及三 seed clean／low／mid 全正且 clean 配对均值≥0.1 pp 才称固定三种子一致双提升的预声明。两个判据均未通过，未启动补种或救跑。

保留 F／CROSS_RX／SIG_TIME 三路／统一 Sinc／joint160／默认 cosine 头联合训练、NoExplicitCFO 及内部补偿、IQ256；原 L6300／U56700 隐藏／V27000、全部五源 RX 和三源 day、split392005。无 RX 留出或 Ldev。原六次 L 前向、1.25 L 尺度、同 IQ RDrop、64 个完整 As、原 KL／dropout、own scratch20 args 的 ordinary epoch13 冻结、既有 U 规则均保留。正式 IQ／SNR／信道／全部包不变。

## 执行与验证

实现 `4ea924316d6a5fe84278c81e3db78706ff674f21`，实际训练 release `7bbb5536cb72c9b95c7c2045ba660e1e72e234e7`；完整零模型分析 `5f3a0fc9c128944994f40d4f4858a5f8e7730b40`。唯一 owner 为 codex/root/practical-budget50-replication。96 个协议负例、4 个真实 seed 正例、真实父 checkpoint no-query smoke 2 NN、四行真实 F 合成 smoke 3072 NN 全部 PASS；唯一 P0/P1 审查 PASS，未重复审查。普通 N607 账户的一次 preflight、一次 release archive 比较、远端 compile 与启动后 PID／CWD／argv／GPU／日志增长均 VERIFIED。历史启动快照保留在 execution_verification.json，最终原进程退出且无 failure。

完整读取新增 17760 step／80 epoch；匹配物理与普通增强序列 mismatch 均为 0。每 row 七视图预测固定后由独立 scorer 连接该 row truth；汇总分析先完成源与全日志读取，再等待全部 12×7 固定预测和独立评分后读取 truth 重算，核对 1512 个 CM／252 次分区。7 项定向分析检查 PASS，分析 0 NN／IQ／checkpoint／更新。最终 11 个公开汇总经一次 archive SHA 比较 VERIFIED，无私有原始数据入 Git。

实际新增四行 80 model-epochs／17760 更新／29,845,170 次样本前向，含共享 smoke3074；旧 seed01 E50 与六 E30 父分别列账。完整学生、教师、读出、每行耗时在 final_analysis.md。详见 execution_verification.json／analysis_verification.json。

## 结论与后续边界

原完整 As 与 E50 的 urban 预算收益获得三个固定种子的支持；当前固定配方未解决 clean 代价。保留它作为星地训练参照，不因单种子正结果自动延长至 E50 以上。不重复另一对话正在开展的源状态报表、既有 urban SSL 或拟研究的单侧间隔。

当前预算假设已检验完毕。现有源摘要不足以把剩余损伤唯一归因于增强比例、加入时机或监督方式，因此本轮没有强行命名并启动下一机制。后续候选须给出一个源依据、一个主要变化与同预算匹配对照；这不构成新的审批或泛化前置门槛。可重复 clean＋星地双提升仍未证明，研究目标保持未完成。

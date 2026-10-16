# 孤立轻度 AWGN(raw)：两默认头源身份检验

**ANALYZED／VERIFIED，PASS_SOURCE_IDENTITY_ONLY。** 两模型的pooled和每draw均通过原固定身份保护与实际噪声变化条件，支持进入独立训练检验；没有证明跨RX泛化、practical收益或联合目标。

[原始预登记](preregistration.md)和[实际唯一矩阵／模型来源](experiment.json)保留。基线F＋跨RX同TX，三SIG_TIME／统一Sinc／joint160／NoExplicitCFO／原cosine30默认头。两个source-only scratchE20 final20参考冻结；只使用其原训练fitL5040（RX3/4/6/8×day1/2/3×6TX，72格各70包），raw准确率是source-fit成绩，不是heldRX或正式clean。四draw重复同一5040物理包，不充当20160独立物理样本。

## 完整识别结果

| seed | raw准确率 % | noise准确率 % | accuracy变化 pp | macro-F1变化 pp | raw正确保留 % |
|---|---:|---:|---:|---:|---:|
| 2026092701 | 99.841270 | 99.697421 | -0.143849 | -0.144068 | 99.850954 |
| 2026092702 | 99.841270 | 99.702381 | -0.138889 | -0.138991 | 99.845986 |

| seed | draw1 pp | draw2 pp | draw3 pp | draw4 pp |
|---|---:|---:|---:|---:|
| 2026092701 | -0.138889 | -0.158730 | -0.138889 | -0.138889 |
| 2026092702 | -0.138889 | -0.158730 | -0.138889 | -0.119048 |

两个模型pooled及每draw均满足：accuracy和macro-F1相对raw≥−0.5pp、raw正确保留≥99%、每RX/TX/day准确率相对raw≥−1pp。没有抽掉困难包、RX、TX、day或噪声尾部。每draw及pooled全部72格完整复算；72格用于透明报告，未事后新增逐格筛选。

两个模型采用同一命名噪声／shuffle流，其完整320batch统计逐项一致。20160噪声观察的相对IQ RMS变化median0.039368733、max0.106857479，满足原median≥0.01、observedmax≤0.15条件。这个max是固定源观察中的实际最大值，不是对高斯未来样本的有限幅度保证。

## 实际算子和边界

唯一算子：原native _rand_uniform逐包生成20至36dB名义SNR，再调用原_apply_awgn处理raw复数IQ，输出FP32两路IQ。方差按原包功率，I/Q独立proper complex Gaussian；没有MP、PA、DAC、class-signature、CFO、时移、相位、量化、剪裁、RMS恢复、过滤或重抽。幅度、TX/RX/day和置信度不进入算子决策；数字变换已接收IQ不等于更换真实RX。

范围来自仓库既有ordinary，而非目标成绩或新幅度搜索。新增噪声SNR不等于原录音总SNR。名义SNR有界，高斯样本幅度不严格有界；全部实际能量、交叉项和尾部保留。当前CPU源实现验证的是同一native分布，未来GPU训练须使用正确设备上的原生调用与独立随机流，不能直接误用CPU输入约束。

公共旧ordinary的真实args、base配置和导入文件保存用于来源核对，但本轮没有执行旧ordinary。未来训练仍保留原公共ordinary，不复活已失败的联合信道移除、后RMS或PA条件救跑。既有As本来读raw，没有本轮同视图ordinary→practical重复完整信道。

## 独立复算、成本和执行证据

独立reader没有调用候选score/qualify/model：从全部50400硬判决和完整logits重算argmax、标量logsumexp CE、accuracy、macro-F1、混淆矩阵、RX/TX/day与72格、四draw、raw正确保留、修复／新错和所有原固定条件。320noisebatch共40320观察全部保留，逐包检查名义SNR、raw/noise/output RMS、实际added-noise SNR、相对IQ/功率变化与归一化交叉项，验证功率恒等式；没有重新读取源IQ、加载模型或执行前向。

实际50402NN＝50400源默认头＋首模型真实checkpoint合成2包no-query smoke。两次checkpoint加载，40320source native noise生成，另128canonical合成native noise（原生和包装各32/model）；本地1152合成native noise、14输入配置负例和9科学反例，0源IQ/CP/NN/更新。旧ordinary／PA／As生成、优化器／教师／另训头／新epoch／target均0。两个参照历史40model-E／8880更新／9373356NN单列，不作为本次新增训练。

原缓存L6300及未用V／历史特征反序列化，RX1原L1260在推断前排除；U只含角色ID，无UIQ或TX标签。本轮不推断heldRX1/U/V/target，不混同完整L6300或L_train5580/L_dev720。原checkpoint训练来源与角色合法、无未知父权重/target/另训头；只用作本冻结源检验，不作为未来训练初始化。

一次独立P0/P1审查PASS；一个普通账户预检、一个archive传输SHA比较和remote compile均VERIFIED。dispatcher和首worker的PID/CWD/argv/CUDA空值已实际观察，首worker两次live观察间日志增长。第二worker已取得active/completion/全预测及自然退出证据，未取得其存活时进程快照，不虚构第二worker的live观察。最终全部ownPID自然退出，无技术failure、重启或其他任务干预。worker主体耗时和总和见execution_summary，未另测dispatcher总墙钟。图仅使用聚合结果、0源IQ/NN/更新，PNG已实际查看。

执行commit 0b5f6ffe7146914f5878cf0fb21d2739f13e462a。聚合见[source_analysis.json](source_analysis.json)，执行与成本见[execution_summary.json](execution_summary.json)。图：[完整源结果与噪声尾部](E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_isolated_AWGN_source_20261017/final/figures/isolated_awgn_source.png)；同目录PDF保留。逐包ID／完整预测／统计／日志仅保留于local_artifacts/F_isolated_AWGN_source_20261017/final，不提交Git。

## 下一唯一假设与放弃条件

假设：在公共F＋跨RX同TX配方不变的前提下，额外孤立轻噪声raw视图可以改善未见源RX身份识别，并超过额外raw／同一ordinary视图带来的前向与监督量收益。下一最小实验是单seed、三行scratchE20，公共raw.75＋ordinary.25＋原SupCon/U不变，额外.25分别raw、同一已生成ordinary、noise(raw)，均8个L前向／损失pair权重1.25／同IQ RDrop。此训练应另登记与启动，本源launcher没有自动训练。

候选相对两个控制均须heldRX1 clean≥+1pp、macro-F1不降、每practical和seenV≥−0.5pp；FAIL则关闭固定20至36dB／全覆盖训练方案，不提高SNR、改范围／系数、clip／删尾／重抽、按RX/TX/day条件或换seed/RX/预算救跑。PASS后只支持独立源seed确认，不能直称正式跨域或联合成功。

[OpenRFI（AAAI2025）](https://ojs.aaai.org/index.php/AAAI/article/download/32003/34158)包含轻度AWGN与片段重排的组合预训练；其总效果不能代替本协议的孤立AWGN跨RX证据。本次不采用重排、原型或新增损失。最终Ag/As完整契约四格及交互、稳定跨seed、正式clean＋六practical、lowurban>60%与suburban接近自身clean仍未完成，联合目标ACTIVE。

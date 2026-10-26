# 单视图 PAMF：L/U 恢复视图与噪声位置

用户最新提出同一clean IQ经不同practical residual信道/DURA、加高斯噪声、利用U的RX标签，并分析L/U在半监督阶段的优化差异。先做冻结、同物理包机制诊断，已运行3个REPL和6个严格基线保持原计划。

固定模型、数据、矩阵、命令、预算、独占root及停止规则见[experiment.json](experiment.json)。单SIG_TIME ordinaryPL E20 seed01、同固定G，L全6300、U每RX固定1260共6300；U不读TX，0V/target。raw/rawNoise40和三景Ga/Gb/preNoiseG/postNoiseG共14视图。0optimizer，F176402/G113402含sourceL2smoke。普通账号N607既有Phase1 Python/Torch2.1、FP32/TF32off/CPU2，无安装。先实际L2smoke，PASS即继续。

比较波形差异、L真类与伪标签错误、同包双G共同错误、跨RX正对质量、原默认头feature-leaf梯度。U只给覆盖和配对数，真实精度未知；L为已见训练集，不能据其分数声称泛化。噪声40dB按注入点信号功率定义，无IQ重新归一化。全F/G参数与buffers不变，未另训头。该诊断不选择候选、扩种子、读取target或改变当前REPL源门槛。

下一步拟定L-only/U-only/L+U恢复视图比较及同包无标签关系的独立消融；仅设计，尚不启动新学生。完整方法依据和保持原目标的比较规则见[研究计划](../../../docs/goals/PAMF_REPL_RESTORED_LU_20261026.md)。

每卡全局CUDA+pending<2/free>=10000MiB，计入其他owner，不改变其进程。一份不可覆盖release/run、一次archive比较、remotecompile、独立PID/CWD/cmd/GPU/日志增长；任何角色/隐藏TX/finite/shape/buffer/smoke/输出错误结束本run，保留产物不自动重复提交。

当前PLANNED，6项聚焦测试PASS，唯一独立P0/P1审查PASS，远端实际smoke为发布第一步。


## 实际启动VERIFIED

执行commit `0a840408f2b35cf77556a2e8e39c4b4ac2dc9445`（包含已核实的新实现提交994a876d）；一次archive比较VERIFIED、remotecompile PASS。实际L2单视图E20/默认头/同G smoke PASS。controller/唯一worker的PID/CWD/完整命令/物理GPU3与日志增长、L进度增长独立核实，读取时L4352/6300；后续自动U6300。0optimizer/V/target，仍是诊断中。此前一次本地版本检查在打包前拒绝了共享检出的提交切换，未创建archive或远端run；待提交/远端OID核实后只成功发布这一个run，没有重复worker。[启动证据](launch_evidence.json)。


## 完整诊断与独立核算

ANALYZED/VERIFIED。冻结已完成单SIG_TIME ordinaryPL E20 seed01与同G，在149.17秒完成L6300+固定RX均衡U6300的14视图。模型/G全部参数和注册buffers不变，0optimizer、0V、0U真值、0target；不是训练新方法。独立读取全部176400次源视图判决、28224000个特征值，逐项重算224个单元、物理ID/角色/RX子集、单位特征、同包判决、跨RX正对及噪声位置指标。原waveform和feature-leaf梯度汇总由已检查的冻结worker产生，未冒充独立重算。

### L与U不能使用相同标签门控

L原始IQ准确率98.793651%，.95门控接受5812/6300（92.253968%），这5812条伪标签在本已见L上全部正确。若L也套用U门控，会丢掉488条真实标签，其中412条原判对但低置信、76条原判错。这些样本的真实TX已知，不能因模型暂时不会而丢弃监督。L恢复视图应使用真实TX/全样本CE，是否附加与U相同的配对一致性、跨RX关系再作独立比较。

U子集接受5756/6300（91.365079%），真实伪标签精度未知，不从L的100%推断U也100%。已知RX可构造约445万有序同伪类跨RX对，但RX不同本身不能证明TX相同；这些组合也不是新增独立物理数据。同包每景的Ga/Gb则有共同底层TX，建立这种正对不需猜TX；和跨RX同类关系分别使用。

### 恢复视图多样性真实存在，也存在共同错误

| 场景 | L双G判决不同 | U双G判决不同 | L双G一致但判错 |
| --- | ---: | ---: | ---: |
| low suburban | 6.253968% | 5.841270% | 1.206349% |
| mid urban | 13.587302% | 13.730159% | 1.873016% |
| low urban | 22.730159% | 22.682540% | 2.793651% |

同包换信道的G相对L2位移中位数约9.3%/14.0%/19.7%（L），U相近。它支持研究多视图；双G会共同出错，故纯G-G一致性不能代替真实身份锚点。旧对比损失因为要求不同RX，原本没有将同包两G当作正对；新方向需要显式区分这种物理正对与伪类别正对。

### 噪声位置不能混同

40dB后置噪声的相对波形位移固定1%；相同局部SNR的前置噪声经过G后，输出位移中位数约0.65%/0.56%/0.42%（L）。它们不是相同恢复域扰动。

| 场景 | L前置判决变化 | L后置判决变化 | U前置判决变化 | U后置判决变化 |
| --- | ---: | ---: | ---: | ---: |
| low suburban | 0.206349% | 0.460317% | 0.206349% | 0.428571% |
| mid urban | 0.269841% | 0.555556% | 0.317460% | 0.698413% |
| low urban | 0.269841% | 0.984127% | 0.222222% | 0.873016% |

后置噪声更直接改变F判决，可形成额外训练压力，也可能增加错误；单次加入噪声的准确率小幅变化不是训练收益。先比较多信道恢复视图的L-only/U-only/L+U，再将噪声位置单列，避免把共同效果混在一起。初轮不依据某个偶然噪声结果挑seed或堆强度。

L的原默认头feature-leaf梯度诊断保留原真标签CE/伪标签CE/原始soft一致性相对方向和范数，见CSV；它不是骨干参数或优化器更新方向。L是已见训练数据；本诊断不证明目标泛化、U精度、三seed收益，原目标仍未完成。

证据：[独立核算及完整覆盖](complete_evidence.json)、[224个视图单元](source_view_metrics.csv)、[噪声位置比较](source_noise_placement.csv)、[原头特征梯度](source_feature_gradient.csv)。

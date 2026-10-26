# 单视图 PAMF：L/U 恢复视图与噪声位置

用户最新提出同一clean IQ经不同practical residual信道/DURA、加高斯噪声、利用U的RX标签，并分析L/U在半监督阶段的优化差异。先做冻结、同物理包机制诊断，已运行3个REPL和6个严格基线保持原计划。

固定模型、数据、矩阵、命令、预算、独占root及停止规则见[experiment.json](experiment.json)。单SIG_TIME ordinaryPL E20 seed01、同固定G，L全6300、U每RX固定1260共6300；U不读TX，0V/target。raw/rawNoise40和三景Ga/Gb/preNoiseG/postNoiseG共14视图。0optimizer，F176402/G113402含sourceL2smoke。普通账号N607既有Phase1 Python/Torch2.1、FP32/TF32off/CPU2，无安装。先实际L2smoke，PASS即继续。

比较波形差异、L真类与伪标签错误、同包双G共同错误、跨RX正对质量、原默认头feature-leaf梯度。U只给覆盖和配对数，真实精度未知；L为已见训练集，不能据其分数声称泛化。噪声40dB按注入点信号功率定义，无IQ重新归一化。全F/G参数与buffers不变，未另训头。该诊断不选择候选、扩种子、读取target或改变当前REPL源门槛。

下一步拟定L-only/U-only/L+U恢复视图比较及同包无标签关系的独立消融；仅设计，尚不启动新学生。完整方法依据和保持原目标的比较规则见[研究计划](../../../docs/goals/PAMF_REPL_RESTORED_LU_20261026.md)。

每卡全局CUDA+pending<2/free>=10000MiB，计入其他owner，不改变其进程。一份不可覆盖release/run、一次archive比较、remotecompile、独立PID/CWD/cmd/GPU/日志增长；任何角色/隐藏TX/finite/shape/buffer/smoke/输出错误结束本run，保留产物不自动重复提交。

当前PLANNED，6项聚焦测试PASS，唯一独立P0/P1审查PASS，远端实际smoke为发布第一步。


## 实际启动VERIFIED

执行commit `0a840408f2b35cf77556a2e8e39c4b4ac2dc9445`（包含已核实的新实现提交994a876d）；一次archive比较VERIFIED、remotecompile PASS。实际L2单视图E20/默认头/同G smoke PASS。controller/唯一worker的PID/CWD/完整命令/物理GPU3与日志增长、L进度增长独立核实，读取时L4352/6300；后续自动U6300。0optimizer/V/target，仍是诊断中。此前一次本地版本检查在打包前拒绝了共享检出的提交切换，未创建archive或远端run；待提交/远端OID核实后只成功发布这一个run，没有重复worker。[启动证据](launch_evidence.json)。

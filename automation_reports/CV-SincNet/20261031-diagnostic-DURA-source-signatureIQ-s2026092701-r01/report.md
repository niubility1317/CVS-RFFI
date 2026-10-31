# 时序签名身份保持：源域有限 IQ 修正可达性

- run_id：`20261031-diagnostic-DURA-source-signatureIQ-s2026092701-r01`
- group_id：`phase1-dura-source-signature-iq-feasibility`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

独立检验原F加权TripleSIG损伤能否在1% RMS IQ改动内改善源clean-F选择一致。默认DURA源512×01/20，固定4步，对比同球最近clean波形oracle。0G/模型训练/TX-V-target。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

完整SOURCE_ONLY结果见analysis.md/results.json；本次没有G训练或正式预测。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。


## 独立源 IQ 可达性假设

旧 branchFeatures 的 PA 主假设 FAIL 保持。时序签名的描述结果提示另一条源域线索，本次独立注册为真实 IQ 可达性检验。只用默认 DURA 原 source512（128L/384U）×缓存01/20，0新G/模型训练/TX/V/target。源clean参考只用于诊断，不声称部署可用或物理极限。

原 F50 冻结，仍为原默认头。读取实际 native t2 的输出提前退出原 F 后续计算，再经过同一原 TripleSignaturePool；没有复制近似骨干。3项聚焦检查和真实模块入口通过，唯一 P0/P1 通过。首步真实2Ls对比完整F实际加权SIG与提前退出前缀的输出和输入VJP，同时复现原source缓存/原头；PASS立即继续。

每条固定4步，每步为0.25%原IQ RMS的归一化负梯度，投影到原IQ周围1% RMS球，固定最终步，不选中间结果。损失仅与同条source clean的实际加权SIG相对MSE，不使用TX或类别损失。IQ_oracle是同一个球内距source clean最近的IQ，作为同失真预算的波形参照；其参照同样不能在部署取得。

两缓存均须：最终SIG一致数相对native至少+4/512、相对IQ_oracle至少+2；五个源RX均不下降；每个非clean景最多少1、clean不少；平均SIG相对MSE至少下降10%，平均IQNMSE最多增加1%。通过仅支持另登记从零单阶段E30候选；不自动训练。不通过不调整步数/步长/阈值或更换主对象。这里一致数是clean F选择代理，不是真实TX准确率。

CPU2、空CUDA、单进程。主完整F2048、前缀4096、输入VJP4096、显式原头3072；smoke另完整F2/前缀2/VJP4/原头2。0G权重读取/前向/模型优化。源512重复用于两份缓存，不能当作1024个独立物理样本。一次N607 preflight、单归档SHA比较、远端编译及实际PID/CWD/完整命令/空CUDA/日志增长核实。完整数组随后用独立NumPy复核，不新增神经网络前向。默认DURA及六景3pp/clean/同质量效率目标保持。


## 完整源 IQ 诊断结果

VERIFIED / ANALYZED，SOURCE_ONLY。defaultDURA原源512（128L/384U）×缓存01/20，CPU2主循环5.614998秒自然结束。1% nativeIQ RMS球内固定四步SIG修正使cleanF选择一致数486→501、498→504，净+15/+6；同球最近clean波形仅+3/+1。SIG平均相对误差降63.674%/66.793%，IQNMSE降0.620%/0.603%，五源RX均不负。缓存01clean84→83，故总判定FAIL，不放宽或自动训练。clean组SIG误差反而增约10.88/10.54倍，四步损失震荡支持固定归一化步长在小误差样本过冲的解释。不是目标TX效果/部署算法/恢复极限。完整F2048/前缀4096/VJP4096/原头3072，0G/模型更新/TX-V-target，F状态不变；1024源观测/3072原头输入/52组独立复算。默认与完整目标保持。

详见[完整分析](analysis.md)。执行d8e47399ffdf5ae0507dfc6efbfaf0bd7132a6ee，PID1677577已自然结束。

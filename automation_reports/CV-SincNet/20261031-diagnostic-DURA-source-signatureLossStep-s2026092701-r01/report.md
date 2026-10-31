# 时序签名 IQ 修正：损失缩步与 clean 保护

- run_id：`20261031-diagnostic-DURA-source-signatureLossStep-s2026092701-r01`
- group_id：`phase1-dura-source-signature-loss-scaled-step`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

针对固定归一化步幅clean过冲，独立检验四步L/梯度能量缩步；source512×01/20、同1%RMS球和全部原验收不变。0G/模型训练/TX-V-target。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。


## 损失缩步的独立源假设

原PA和固定步幅IQ源诊断FAIL保持。原固定步幅clean组SIG误差约增加10倍并出现四步震荡，本次独立改变IQ更新规则，其余对象、源数据、原F50/defaulthead和全部验收不变。不是旧run技术重跑或阈值放宽。

固定defaultDURA原source512（128L/384U）×缓存01/20，四步最终输出。每步delta=−(L/||gradL||²)gradL，L是同条source clean实际加权TripleSIG相对MSE，梯度能量下限1e−20；每步最多0.25%原IQ RMS，总改动投影到1% RMS球。无回溯或超参数搜索，不用clean标签路由。零loss不改变IQ；缩步动机不构成非线性模型收敛保证。对照仍是同球内距source clean最近的IQ。

两缓存均须SIG一致数相对native至少+4/512、相对IQ_oracle至少+2，五源RX不负，每非clean景最多少1、clean不少，平均SIG误差至少降10%、平均IQNMSE最多增1%。通过仅支持另登记scratch单阶段E30，不自动训练；失败不放宽或搜索。参照是源clean F选择代理，不是真实TX或部署可用的clean参考。

5项聚焦检查和实际入口PASS，唯一P0/P1 PASS。首步真实2Ls完整F/原前缀的SIG输出和输入VJP、源缓存及原头smoke通过立即继续。CPU2/空CUDA；主完整F2048/前缀4096/VJP4096/显式原头3072，smoke另完整F2/前缀2/VJP4/原头2；0G读取/前向/模型优化/TX/V/target。一份归档SHA比较/远端编译、实际PID/CWD/argv/空CUDA/日志增长；独立全量NumPy复算。默认DURA与六景3pp/clean/同质量效率目标保持。


## 完整损失缩步结果

VERIFIED / ANALYZED，SOURCE_ONLY。CPU2/5.812552秒自然结束，原source512×01/20/defaultDURA。损失缩步IQ修正仍净增15/6条cleanF选择一致，clean84/81均保持，clean组SIG误差分别降低82.143%/77.978%，修复旧固定步幅的clean过冲。缓存01全部条件通过；缓存20 RX8净−1，其他条件通过，总判定FAIL，不更改旧FAIL/本次阈值，不直接部署或自动训练。总体SIG误差降64.598%/67.777%，IQNMSE降0.625%/0.605%，同球最近波形对照+3/+1；不是目标TX收益/物理极限。完整F2048/前缀4096/VJP4096/原头3072，smoke另F2/前缀2/VJP4/原头2；0G/模型更新/TX-V-target、F不变；1024观测/3072头输入/52组独立复算。

详见[完整分析](analysis.md)，执行3cd8781b0046d6dec0c5bfd92e88f7da2ba30f7f。

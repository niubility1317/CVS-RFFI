# 原 R17 合规源域 IQ 教师逐层保真诊断

- run_id：`20261031-diagnostic-DURA-R17-source-IQpaths-s2026092701-r01`
- group_id：`phase1-dura-source-IQ-teacher-paths`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定source512×cached01/20，对原R17 coarse及三个后求解IQ路径评估保真与cleanF选择保持；0训练/TX/V/target，不默认中间输出可作教师。

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


## 最小研究说明

原R17已实际核实metadata，source7/63/30，coarse/action/prior各历史20轮、无TX/F/target训练，原源V标签选择先于目标访问。旧scratch_guidance是三F指导的1.305m直接网络E40，旧R17compactdistill则复制过教师权重，与本次从零单阶段IQ教师设想不同；不混淆权限或成本。现先固定source512×缓存01/20，检查R17 coarse及三个求解后IQ。默认DURA中间IQ源误差可能更大而最终更好，所以本步不预设中间IQ为clean、不据目标排名教师或自动启动训练。所有四路径都输出IQ/F特征和logits供独立复核，仅cleanF选择一致性，不读TX/V/target。真实2Ls source检查通过即继续CPU2，数据与原路径保持。当前待发布，科学效果尚未知。


## 实际完成

VERIFIED / ANALYZED_SOURCE_ONLY。实际CPU2/空CUDA、source2Ls严格原R17/F检查通过；512×2×4路径计算13.514224秒，G/F不变，更新/TX/V/target0。208组2288标量独立复核，FP32/FP64差异最大6.30e-7。R17 coarse Fclean保持467/487，相同原noVar coarse257/272；stage1/2并不统一优于coarse。下一早期coarse-IQ指导候选尚待实现/登记，不把本source比较当目标收益。详见 [完整分析](analysis.md) 和 results.json。

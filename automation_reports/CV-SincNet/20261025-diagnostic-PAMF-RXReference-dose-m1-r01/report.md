# 跨接收机身份参照的无更新源梯度剂量诊断

- run_id：`20261025-diagnostic-PAMF-RXReference-dose-m1-r01`
- group_id：`phase1-pamf-receiver-reference-20261025`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原三E30只读sourceL建立固定身份参照，seed01以12源L/U批测raw/同G/双G关系约束梯度；无optimizer/V/target。

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

## 源诊断发布前核实

范围见 [最小预登记](preregistration.md)。六项相关检查PASS，模块compile通过。一次独立审查发现的NumPy桥兼容P1已修复且定向复审PASS，无新增审查门槛。未创建optimizer，真实E30/G及默认头特征smoke由launcher首步执行。本诊断只测L参照和12批源梯度，不访问V或目标；尚不能说明训练或性能目标达成。

## 完整源诊断结果

ANALYZED，独立读回VERIFIED。实际科学代码 `95e0ae90eca7f764906a372230663a9004b483cd`；6300条可见L建立三套own E30参照，12批L128/U256完整、60次autograd.grad。真实权重L2/defaulthead/160维特征一致性smoke通过，源NumPy元数据兼容修复在实际环境完成运行。

原主项梯度范数中位数为0.978037；预定最大中位范数规则给出共同系数 **0.071458449155536924**。

| 新增分支 | 梯度范数中位数 | 与主项cosine中位数 | 加权中位范数/主项 |
| --- | ---: | ---: | ---: |
| RawReference | 1.368680 | -0.037431 | 10.000% |
| RestoredReference | 1.348240 | -0.057523 | 9.851% |
| PairedReference | 1.305353 | -0.053732 | 9.537% |

U曝光3072，原raw教师共识接受2858；新增分支排除排序首个源RX=1后接受2261。这只是12批接受计数，不是U伪标签正确率。原L每批768次包前向、As KL0和分类头非零梯度保持；新增关系分支不单独训练默认头。

全部F/G及临时student参数和registered buffers不变，optimizer/V/目标/U真值/新头拟合/参照更新均为0。实际耗时20.31秒。源梯度cosine比历史G直接CE的诊断值更接近0，不构成训练、迁移或目标收益证明，也不能理解为AdamW更新无冲突。

参见[完整12批](source_gradient_batches.csv)和[源剂量与执行证据](source_decision.json)。下一步按既定source新增分支留出矩阵验证；原clean+1pp、raw星地>1.5pp、同G星地正收益目标仍未达到。

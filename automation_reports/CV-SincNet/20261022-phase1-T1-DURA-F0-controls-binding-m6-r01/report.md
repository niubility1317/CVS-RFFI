# F0输入绑定修复：监督与普通PL×3种子

- run_id：`20261022-phase1-T1-DURA-F0-controls-binding-m6-r01`
- group_id：`phase1-cric-dura-complete-system`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Confirmed source IQ/ID ordering defect repair; original scientific matrix and budgets retained.

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

# F 消融输入顺序修复与重发

根因已确认，属于源输入绑定错误。F0 的 IQ 数组按旧缓存顺序存储，而新元数据按另一套物理 ID 顺序排列；旧 SourceDataset 只检查形状和声明 ID，没有把 ID 映射到数组的实际行。因此监督标签、RX/day 与 IQ 错配；不是移除某个机制导致的有效负结果。

## 已核实证据

完整解析78条相关F行的现有结构化源日志，共3138个epoch记录，并扫描其完整stdout异常标记。17个源契约的既有IQ索引逐一比对，只发现F0顺序不一致；不访问target、不读取U的TX标签、不新建数据验证或哈希链。

| 源角色 | 样本数 | 顺序不一致位置 | 旧可见标签错配 | 按ID对齐后错配 |
| --- | ---: | ---: | ---: | ---: |
| L_s | 6300 | 6298 | 5291 | 0 |
| U_s | 56700 | 56699 | 不读取标签 | 不读取标签 |
| V | 27000 | 26999 | 22564 | 0 |

同一训练实现下，m3-natural seed01源V约98.14%至98.26%，F0 F01/F15/F16及匹配控制约16.6%至17.0%，且训练正确率仅约30%。该差异与缓存顺序错误一致；这里只使用源侧证据诊断，不把目标低分作为调参信号。旧目标评分继续保存并标记INVALID_INPUT_BINDING，不能作为组件贡献或机制失效证据。

## 修复

`source_iq.aligned_source_iq`只从既有数组对应的`*_index.npz`（或实际拥有数组的metadata）读取物理ID；校验集合/唯一性后，将IQ重排到原声明顺序。物理集合、原标签值、数据比例和字节均保持，U只读取ID。相同顺序路径保持原数组。缺失、重复或不同集合直接拒绝。F加载器与未来G新观测准备共用该读取器，防止其他入口再次按错误位置读取；不重建当前合法G缓存。

新回归首先在旧实现复现L/V/U错序和未拒绝坏ID，再验证修复。实际checkpoint无query smoke在新owner第一步执行：固定已有最终T1 seed01，仅合法L_s256条，要求对齐后准确率≥80%、比错位读取高至少50个百分点。此固定checkpoint只用于源侧输入核对，绝不初始化新训练；PASS后立即接管并继续，不另设审批。

## 不变矩阵与技术替代

- `20261022-phase1-T1-DURA-F-ablation-binding-m63-r01`：原F01至F21，各三seed2026092701/02/03，共63条，从零20(13L+7LU)→自身10LU→自身20LU，累计E50/11100更新。F11至F21的已登记机制例外保持。
- `20261022-phase1-T1-DURA-F0-controls-binding-m6-r01`：原F0监督/普通PL两种设置×三seed，共6条；默认sat拼接.68和PL.95保持同次联合训练。
- `20261022-phase1-T1-DURA-F-binding-mechanism-timing-r01`：重新绑定修复后F06至F10的机制独占时延，原G机制权重复用。统一时延r02和原最终T1三Fseed/单G结果不重测。
- `20261022-phase1-T1-DURA-F-binding-control-r01`：先源checkpoint smoke，再精确接管当前资源owner。所有健康PID保留；原未启动错误F及下游预测/评分撤下，完成和正在运行的原产物只留历史，不覆盖。

新69条训练共3450模型epoch/765900更新；各自raw与原同契约冻结DURA的全部七景固定预测后独立truth-last。保留三seed完整结果、逐RX/TX/day、Macro-F1、净纠错、成本和负结果。F16才与最终T1组成当前CRIC×DURA匹配2×2。原错误weights/optimizer/EMA不得作为初始化。

只重跑技术受损的69条，没有新增五源扫点，不重训其余16契约、原最终T1或恢复器。原G加速行的per-job release保留，资源仍r03策略：基础3/card，仅0/6可第四条F，全机≤26、同卡G≤1，内存/CPU门槛不变。机制计时独占且排除GPU2/3。两次同技术指纹停止新派发，保留全部产物。

原769job历史保留，加208个技术替代job（69训练、69预测、69评分、1时延）。excluded旧行不再计入有效必需闭合；原已知timing-r01失败仍由独立已完成r02替代，不重试。发布后的实际PID/日志增长、有效行计数和源首轮曲线另在登记中更新，未回来的结果不作完成声明。

## 发布结果与当前边界

**VERIFIED**：执行commit `a462b7cf4c12a655e6c7a348ce2170a089f2d1cc` 已发布；实际固定T1 checkpoint源侧核对PASS，同一L256错位16.015625%→按ID99.609375%，不读取target/U真值，也不用于初始化新训练。29项聚焦测试分批通过，独立P0/P1及直接缺陷修复复核通过。

当前25条原健康worker已按原PID接管或自然完成，延时回读24条同PID日志增长。新69条修复训练已登记且处于QUEUED，等待现有任务自然释放合法槽位；尚不能宣称已开跑/已训练完成。旧208关联job从有效闭合中排除，原769历史保留，总977；有效计数为{"COMPLETE": 351, "RUNNING": 12, "QUEUED": 405, "FAILED": 1}。原已知timing导入失败由完成的r02替代，无新增失败。

小时监控新owner及替代映射已更新并独立读回ACTIVE。预计修复新增3450模型epoch/765900更新，约105至140共享worker小时（不是独占GPU时）；结合其余队列、预测评分、机制独占时延及论文导出，暂自2026-10-22 20:33 香港时间估18至30小时，首批实际吞吐回来再校准。新结果未返回前，旧错误分数只保留历史，既有正常T1/G和其他源契约结果继续复用。

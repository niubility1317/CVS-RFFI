# 四种Full方法正确practical residual与普通PL四格及PAMF匹配对照

- run_id：`20261023-phase1-practical-residual-full2x2-m600-r01`
- group_id：`phase1-full-practical-residual-factorial`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

11物理合同、四种Full原生目标×信道开关×普通PL；匹配PAMF监督与新RX原生T1，三种泛化测试。

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


## 发布实现和本地验证

本次600个新F=528外部Full+66匹配监督PAMF+6新RX原生T1，另4条完整匹配G；27个既有T1只补三类泛化读出。全部逐行冻结raw/匹配G、三类泛化和七景后独立CPU评分。取消外部CE、固定L/V、chrono训练、m5和细控制扩张。

加速已实现为逐轮异步CPU producer、按L物理ID读取的共享FP32单场景缓存及下一轮预取；不缓存模型目标。多方法/种子/PL开关复用同一个确定性输入函数，重复ID/乱序消费逐元素一致。PAMF监督同时修正As和附加槽，原生T1配方不变。新G复用已验证源观测并启用既有冻结教师epoch/ID缓存与零权重FFT跳过。

10项聚焦本地检查PASS，包括route、矩阵/预算/RX、禁U/禁truth、无增强冻结读取与缓存同函数乱序消费。管理Python未安装Torch的初次环境检查不算模型失败；实际检查使用现有phase1-cu128环境。唯一独立P0/P1审查3个直接配置问题已修复并定点确认，无未解决P0/P1。真实checkpoint/source smoke是远端launch第一步之一，PASS后自动继续，当前不提前宣称通过。

用户最新要求保留三张卡：仅GPU0–4，每卡最多3、最多15个本任务GPU进程；GPU5/6/7保留。正式资源准入仍计入其它owner、显存/CPU/RAM/IO guard。没有停止、热改或重启任何旧健康任务。无已有正式新性能。

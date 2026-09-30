# 解析自由截距残差LocalRidge与support监督adapter联合适应注册

- run_id：`20261001-phase2-d92-affine-joint-support-m2-r01`
- group_id：`d92-affine-joint-support`；类别：`diagnostic`；阶段：`Phase2-support-only-diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：LOCAL_VERIFIED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

保留旧类核尺度、实际B函数先验和函数坐标adapter，以带解析自由类截距的残差LocalRidge经完整隐式梯度联合训练；保留完整interaction，不做参数网格。

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

LOCAL_VERIFIED/VERIFIED：单一解析自由截距结构已实现，79个不同相关检查通过（core/entry/ops58、summary10、reporter11）；唯一独立P0/P1审查无未解决问题。完整Schur和非零g_b伴随、当次B→C状态、全部RHS与状态字节、真实事件顺序经合成检查。实际四缓存/160parent输入绑定及新输出无冲突已核实。方案在旧AJLR完整评分前按数学推导确定，无部分结果选模/参数扫描。未发布或启动，真实性能尚未知；A及B−A=N/A，query/source样本不读，目标ACTIVE。

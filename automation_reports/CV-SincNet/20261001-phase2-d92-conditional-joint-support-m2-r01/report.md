# 旧点函数约束的条件核LocalRidge与support监督adapter联合适应注册

- run_id：`20261001-phase2-d92-conditional-joint-support-m2-r01`
- group_id：`d92-conditional-joint-support`；类别：`diagnostic`；阶段：`Phase2-support-only-diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定Phase1及既有合法support缓存，B用解析Affine LocalRidge；C在actual B函数上用全列旧点零残差条件核注册，完整隐式梯度与固定坐标球优化类RMSCE；结构改动而非参数扫描。

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

VERIFIED：独立完整KKT证书63项及核心与5ENTRY联合37项合成检查通过；C q-gauge审计、JSON标量口径和Training.audit_dict ABI缺口已修，原失败保留。固定Phase1、合法support缓存的4row/160parent声明已预登记，只读元数据preflight核实缓存/来源/输出未占用，不读取query或源样本。新方法独立summary/analyzer/report/collector仍在实现；没有publish/launch或真实性能结果。A与B−A仍N/A，R0不得替代A；理想目标为soft，目标ACTIVE。

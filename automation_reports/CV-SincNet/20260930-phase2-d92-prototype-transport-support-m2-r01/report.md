# LocalRidge与目标原型条件化监督adapter联合适应及新类注册

- run_id：`20260930-phase2-d92-prototype-transport-support-m2-r01`
- group_id：`d92-prototype-transport-support`；类别：`diagnostic`；阶段：`Phase2-support-only-diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

LocalRidge为最终分类器；合法support原型条件化切向旋转训练10参数/9自由度，保留半份原interaction，B后C顺序继承，与原方法和C重置配对。

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

VERIFIED：PrototypeTransport-LocalRidge唯一160parent support pilot已预登记/availability通过、尚未启动。固定Phase1/practical residual，source-free、不读query；10存储参数/9约束自由度、0.5原几何+切向适配、LocalRidge最终头、真实B/C继承。核心27、入口13、编排14相关测试已通过，独立核心实施审查无P0/未关闭P1。真实失败和修复证据保留，不宣称性能或资源改善。root唯一launch owner。

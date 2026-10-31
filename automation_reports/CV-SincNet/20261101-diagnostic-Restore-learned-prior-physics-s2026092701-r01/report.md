# 恢复性能源端诊断：强先验与真实信道校正、F特征误差几何

- run_id：`20261101-diagnostic-Restore-learned-prior-physics-s2026092701-r01`
- group_id：`restoration-source-learned-prior-vs-known-action-and-Fgeometry`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原L_s/U_s冻结DURA与MSAT，分别加入真实模拟action的MAP校正，并分析源F0特征逐维标准化误差；0训练/0目标/0TX标签，不称可部署oracle或理论上限。

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

## 源端反事实与特征几何

只用原 L_s/U_s，0 TX 标签、V/OE/novel/目标输入和优化。冻结实际原 Tri-SIG-TIME SSL、默认 DURA 与 MSAT final30，固定512条源包×20份已有R07模拟。DURA/MSAT各自与真实sourceAction的16/64CG校正比较，real prior variance固定.01；这是源端不可部署oracle，不是新G或理论上限。另用原63000 clean计算F0逐维总体方差，分析标准化特征误差及方差四分位组，防止用单一全局cos描述所有方向。全部网络保持冻结，完整主性能目标和默认版本保持。

## 已完成

VERIFIED / ANALYZED：真实source smoke、10240实例/20draw/6固定路径、63000原源F方差全部完成，0更新/TX/V/OE/target，PID已退出。独立聚合差0、逐维聚合差≤7.62e-9，旧DURA对照逐实例一致。MSAT真实信道校正改善IQ NMSE但低档城区F一致率净−8条；16到64CG判决变化0。源方差标准化误差与判决偏离的关联略优于cos，但不证明新训练/目标收益。主性能目标ACTIVE，默认DURA保持，详见analysis.md/results.json/independent_check.json。

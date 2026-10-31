# 恢复性能受控实验：同MSAT结构与预算，增加逐维源F特征保真

- run_id：`20261101-phase1-Restore-MSAT-VarF-E30-s2026092701-r01`
- group_id：`restoration-source-Fvariance-controlled-single30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

根据原源特征方差与判决偏离关联，只改变F0特征目标：.1(cos+逐维源方差标准化MSE)，保留.5IQ梯度cap；scratch单阶段30、正确原Main七景自动测试。

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

## 只检验F特征约束的改变

同原MSAT结构、随机初值、原LsUs、R07二十份加十份复用、源shuffle、优化器/LR及30轮/7410预算。只将.1cos改为.1(cos+逐维源方差归一化MSE)，统一保留.5逐记录IQ梯度上限。源F0方差来自原63000 clean（DATAONLY、F0-only、非G输出），其producer的DURA/MSAT诊断输出不进入本G训练或初始化。无TX标签、logit/CE/原型、另训头或OE/V/目标学习。

源方差误差与Fclean判决偏离的关联略优于cos，仅是做受控实验的依据，不能保证Main提升。新G仍从零联合单阶段30，固定final30后无条件自动原地面Tri-SIG-TIME SSL/Main split两总体七景，并在全部预测固定后独立truth-last。旧MSAT作事后精确对照、不重训、不继承。性能主线与默认DURA保持。

## 启动已核实

VERIFIED：方法3729a7135728500ab6013894ac060d85e3db8a67，6聚焦检查/唯一P0P1 PASS；一次archive SHA/remote compile/preflight完成。GPU7原compute0、剩余24124MiB，新增单个本任务NNworker。实际PID/CWD/命令/CUDA与日志增长已读回，真实source smoke/严格E0重载PASS。当前训练中，无新版正式性能结果。所有旧模型/输出及其他owner保持。

## 技术失败：无正式性能结果

FAILED / NO_PERFORMANCE_RESULT。第30轮CUDA异步launch failure；进程退出后独立设备查询确认GPU7句柄不可用（exit6），GPU4/6仍可查询。故障层在设备/驱动，具体硬件原因UNKNOWN。29轮完成日志保留，已完成更新至少7163、至多7409；只保存E0，无训练态或final30可续训/测试。未读取目标或产生正式预测，不能把该尝试当作性能负结果或达标模型。

新独立r02仅作技术恢复，科学配方/数据/种子/预算不变，从零30；避开GPU7，按容量只选4/6并用UUID绑定CUDA。新增每完整epoch的模型/优化器/RNG恢复快照（不选模、PRIVATE），防止同类故障再次丢失全部进度。保留本失败root，记录已耗训练成本；不重置设备或干预其他owner。

# 完整分支四相位轨道与交叉熵的support四臂诊断

- run_id：`20260929-phase2-d92-branch-orbit-ce-support-m4-r01`
- group_id：`d92-branch-orbit-ce-support-development`；类别：`diagnostic`；阶段：`Phase2-support-only-diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定Phase1与同一received观测，四相位完整分支轨道表示和交叉熵判别头的2×2对照；完整物理OOF及support内部全部单样本anchors。

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

完整四臂和物理OOF/全部proxy anchors已预登记，received-only四相位不增加K；实际配置固定，source-only原模型来源未变。原capsule及GPU0/CPU/空输出路径preflight VERIFIED，尚未发布或启动。

发布前独立审查完成，两项P1已修复并定点关闭：轨道近抵消的thinQR等价稳定计算、CE异常上下文和已完成阶段保留。冻结公式与配置未变，核心32、exporter18、entry10、summary70、编排16项测试通过；真实support实验尚未启动。

VERIFIED启动：supervisor185899，首exporter185909；readback_1790674303.json核实实际argv/CWD和support特征进度。实际release d57307814ef6a51f542a0455f25c3fe22cc0a6d7。目标仍ACTIVE；当前仅真实support诊断，无query成绩。

readback_1790674538.json确认首组导出完成8273物理support/33092视图前向，checkpoint smoke PASS且参数buffer未变；已进入CPU probe并完成22/900任务，下一组GPU export同步推进。新增源样本与地面统计payload均0B；88923468B特征文件是接收端缓存而非地面传输。尚无完整性能结果。

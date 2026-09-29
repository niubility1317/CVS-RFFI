# 完整分支四相位轨道与交叉熵的support四臂诊断

- run_id：`20260929-phase2-d92-branch-orbit-ce-support-m4-r01`
- group_id：`d92-branch-orbit-ce-support-development`；类别：`diagnostic`；阶段：`Phase2-support-only-diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

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

八组export完成已VERIFIED，证据readback_1790675306.json；所有smoke和冻结检查通过。资源口径见docs/D92_BRANCH_ORBIT_COST_20260929.md，source/ground新增0B不等于全部通信0。最终summary将澄清CE NLL沿用ridge文字标签，数值及运行中方法不变。

VERIFIED：readback_1790686403.json确认8组与全部4800任务完成，1200个K1数值任务、3600个OOF任务、42000个proxy anchors、105600次ridge分解，实际CE优化5563959步。supervisor及所有所属子进程均退出，尚待完整汇总科学判定。本地ssr-gpu旧解释器路径不可用，本次只读监控/编排使用已核实的F:/App/miniconda3/python.exe 3.13.9标准库，未替换实验的N607 CVS-RFFI环境，未重跑测试或训练。

VERIFIED完整support诊断已ANALYZED：4800parent/3600OOF/42000proxy/105600CE fits及5563959实际更新全部核验；标准与proxy筛选均失败，不进入query、不晋级、不按成绩重跑。完整解释见support_interpretation.md和results/support_summary。runtime d57307814ef6a51f542a0455f25c3fe22cc0a6d7，analysis 1140fdfbc79287effe88fd1e3afbf4498c86f1b7。

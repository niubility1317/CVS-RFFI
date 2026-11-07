# 固定识别器源域一致性诊断

- run_id：`20261108-phase1-frozen-source-proxy-agreement-s2026092701-r02`
- group_id：`phase1-frozen-source-proxy-agreement`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

在源域非真值 clean 伪类别参照下比较三个已冻结独立恢复器，不能视为身份准确率。

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

## 运行核实

- 修复提交 `099ceaaed55c16cf83d766943b2e4ab0588aa9a7`；旧 r01 在首批比较时因 NumPy 数组类型不兼容失败，未保存预测。本次使用新的输出目录。
- N607 后台 CPU、2 线程、隐藏 GPU，PID 928117；发布包本地与远端 SHA 一致，远端编译通过，首批 8 条真实源域记录与既有固定参照一致并继续处理。
- 三个候选均在诊断前冻结；F 和 DURA 只用于冻结后的源域比较，不读取 TX 真值或目标测试。clean 伪类别一致率不是识别准确率，不用于更改已登记的正式测试。

- 已完成 672/672 条源域冻结代理比较；完整汇总见 [analysis.md](analysis.md)。DURA 在六受损场景合计 546/576，三个直接模型分别为 479、464、479/576；这不是 TX 识别准确率。

# Phase2 第二组：C 主干交叉对比的 MoPC 加速接管

- run_id：`20261103-phase2-csource-crosscil-fast-m4-r02`
- group_id：`phase2-group2-csource-bc`；类别：`comparison`；阶段：`Phase2-B-C`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Safe single-owner continuation with identical matrix and batched MoPC elementwise regularization.

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

## 2026-11-03 接管启动读回

新 controller PID 2180731 已运行；旧 controller PID 2128146 已精准暂停派发，等待 16 条旧 worker 自然完成，并核对所有已启动索引闭合；此状态不代表接管成功。实际转移、旧预测复用、源 checkpoint 无 query smoke 和服务器实测加速以随后产物为准。

## 2026-11-03 加速接管完成

旧 controller 仅在 16 条自有 worker 自然结束后退出，新 controller 2180731 接管。真实 source-only 无 query smoke 21/21 PASS；在真实 MoPC 源模型的 151 组参数上，新旧正则损失和梯度逐位一致。旧 run 370 个闭合任务、2770 行预测已逐行核实并硬链接复用；先导预测及评分引用保留。目标 truth 仍待全部预测冻结后独立使用。服务器端的实际速度需在新任务完成后按相同 K/N、seed 与 GPU 共享情况比较，不能引用本机合成倍数替代。

## 2026-11-03 评分加速接管完成

父 controller 2180731 等待 16 个自有 worker 自然完成后退出，新 controller 2199356 已接管。源 checkpoint 无 query smoke 21/21 PASS；MoPC 源模型 151 组参数的优化前后正则损失及梯度逐位相等。父 run 515 个闭合任务、3515 行预测均已独立核实并硬链接复用，所有预测固定后才由最多 4 个常驻 CPU 进程执行原独立评分。

N607 共享 CPU 的一次 8 任务合成评分测试中，逐任务独立进程为 7.13 秒，常驻进程为 4.21 秒；两组 4 个并行评分进程，所有合成指标一致。该测试不涉及实验目标数据，不能代替正式全量评分耗时。另在相同 B/C/seed/scene/K5N20、不同数据池且共享 GPU 的首批样本中，MoPC 注册旧版70项中位17.92秒、新版35项18.08秒，尚无服务器端注册加速证据。

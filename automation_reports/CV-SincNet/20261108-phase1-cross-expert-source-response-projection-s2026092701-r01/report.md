# 时频恢复器源域响应梯度投影单步诊断

- run_id：`20261108-phase1-cross-expert-source-response-projection-s2026092701-r01`
- group_id：`phase1-cross-expert-source-response-projection`；类别：`diagnostic`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

同一源波形批次上比较波形目标、固定响应联合目标与梯度冲突投影，检查实际 AdamW 更新及独立源样本损失。

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

## 后台源域诊断

代码 commit `d452377d30b04fd938ef7a38645a492d0a01e237`；两项针对性测试及独立 P0/P1 只读审查通过。发布包两端 SHA256 一致、远端编译通过。N607 CPU 进程 1214284 已核实实际命令和工作目录，真实 DURA checkpoint 的 8 条源样本 smoke 已通过，六个单步对照正在运行。

这是在已经探索过的 32 条源拟合样本和 32 条不重叠源留出样本上，测试波形损失与 1% 幅度扰动响应损失的梯度冲突。它只更新新初始化网络的输出头，每条件一步，未保存模型；与实际在训全网络、GPU 训练、TX 身份准确率和七景地面结果不同。投影只保证未预条件梯度的局部方向关系，记录实际 AdamW 更新以检验该限制。

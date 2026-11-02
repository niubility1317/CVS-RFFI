# R42 直接注册：共享核与 CUDA 加速无中断接管

> **科学处置：搁置。** 按用户要求，此次零目标旧类 support 的直接注册变体不再作为当前候选或自动继续；结果保留为已完成的负结果。此前使用目标旧类 support 的原 R42 不受此处置影响。


- run_id：`20261102-phase2-r42-directc-fast-m4-r02`
- group_id：`phase2-r42-direct-c-zero-target-old`；类别：`cvs`；阶段：`Phase2-C`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ARTIFACTS_COMPLETE（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Fixed direct-C MIX01 16 KN; identical source and task definitions; inherit finished/in-flight predictions; share CPU kernels and use FP64 CUDA for large support.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

结果与覆盖见下方独立评分记录及完整结果报告。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 已核实的接管

N607 controller 1526811、独立 readout 1528741；旧 controller 已退出，原 16 个完整单元复用，8 个旧 worker 自然运行并计入限额，首批 11 个新 worker 已启动。4 个实际源模型无 target/query 数值检查 PASS，最大分数差 2.22e-16。全部预测闭合后独立评分；原生 CIL 只复用 raw/raw。

本地合成数据单环节测试显示共享核与大支持 CUDA 可减少部分判决耗时，GPU 共享及负载变化使其不能作为本实验整体速度；实际吞吐待新任务闭合后核算。

## 独立评分与结果

84/84 单元、10752/10752 行预测完成并独立评分，24 单元复用 r01、60 单元由 r02 新计算。128 个汇总组与三个原生 CIL 的配对结果已核对；[完整结果报告](results_direct_c/report.md)。N=2 四个 K、两变体及四路径旧类宏准确率和宏 H 均为 0。实验为已暴露开发评估，不基于这些结果重选或调参。

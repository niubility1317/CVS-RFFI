# PAMF 七方法论文及补充对比矩阵

- run_id：`20261126-phase1-pamf-seven-method-matrix-m111-r01`
- group_id：`phase1-pamf-seven-method-matrix`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（已核验全部独立评分及七方法总表）

## 目的与对照

发布 Receiver-Agnostic RFFI r04 和 STAR-RFFI 的匹配训练与读出，汇总全部七种识别方法。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

Main split 的既有 raw 评分可以先形成七方法对照表。新启动的 111 行按相同数据契约独立预测及评分，结论随评分结果更新。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。


## N607 启动核验

- 训练及预测代码提交 `8178a77cac807c29b1e5e7eaec6052f1fe6116ec`。远端归档校验和编译通过，控制进程 PID 782363 的工作目录和命令已核对。
- 两个真实模型通过源域无 query 测试，零目标或真值读取。首批 GPU 2、6、7 补 Main split 读出，GPU 5 已写入训练日志。STAR-RFFI practical 缓存同步生成。
- Word 报告后续控制实验包括总标签预算 630、3150、6300，16 个等样本量接收机组合，以及 5 个日期留出或前推设置。对应数据、种子和覆盖在 experiment.json 列出。
- 历史 RIEI、DRIFT、IB-RFF、SDG 和 PAMF 各 57 行七景独立评分已核对。逐种子结果及均值由统一汇总生成。
- 初始汇总代码在纯路径上调用文件检查，已为正在运行的不可变发布安排独立修正的汇总进程。训练、预测和评分继续使用原发布提交。汇总目录以 complete.json 确认可用。


## 最终核验

2026 年 11 月 27 日 07:46（香港时间）核验。108 个从零训练行和 3 个 Shen r04 Main split 补测行均已完成预测和独立评分，共新增 111 行。STAR-RFFI 的 34 项 practical 增强缓存全部完成。

七方法矩阵含 399 个方法、数据设置和种子组合，每种方法 57 行。覆盖 11 个主设置和 24 个补充设置，其中包括 630、3150、6300 三档标签预算、16 个等样本量接收机组合和 5 个日期控制设置。总表另外列出五源接收机场景的公共六接收机汇总。

N607 的 `tables/complete.json` 为 `ANALYZED`，缺失 0 行。 `seven_method_columns.csv` 有 904 行，各行七个方法均有值。 `per_seed.csv` 有 14168 条逐视图评分记录。等样本量接收机组合汇总在 `equal_size_receiver_groups.csv`。

原调度器完成所有训练、预测和评分后，在自身的最终汇总步骤触发了已知路径错误，留下 `FAILED` 状态。独立汇总进程（提交 `4cf2ac7fe`）已完成并通过产物核验。以逐行评分和总表完成标记为准。

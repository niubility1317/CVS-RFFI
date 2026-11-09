# 源模拟信道与估计信道的微扰响应对照

- run_id：`20261109-phase1-true-estimated-perturbation-source-s2026092701-r02`
- group_id：`phase1-true-estimated-perturbation`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED；672 条源观测的诊断已完成。

## 目的与对照

在合法源 IQ 上比较已固定 DURA 估计信道与匹配模拟器信道对相同合成扰动的响应，无 F、TX 标签或目标读取。

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

## 固定方案与验证

独立重试 r01 的源域诊断，源样本、四种扰动、冻结 DURA 和计算公式保持不变；仅使用唯一的新输出目录、只含诊断模块的发布归档，以及逐项列在 experiment.json 的既有固定发布路径。首次 r01 因依赖导入失败、未进入源 smoke，已保留原日志。

代码 commit `895e0fd944ec70de48ce54778cf2e73c66e1b7b1`；发布归档本地和远端 SHA-256 均为 `cf83c13fb4eedba19fdd8d83149fbd9583d449c8226a5957c6e0760c43aca6ad`。远端编译、原 DURA 历史依赖导入及同向／反向／缩放断言通过；形状、类型及非有限数在前次相同代码的实际模块检查中通过。独立 P0/P1 静态审查无阻断。真实权重首批 smoke 为正式执行首步。

N607 GPU4 预检不少于 8192 MiB 空闲、少于 3 个总计算进程且本任务同卡最多一 worker；不读 TX、F、V 或目标，不更新模型。全部 672 条源模拟观测的十四个分组及四种扰动只用于源物理响应诊断，不能证明设备识别收益。路径、命令和失败停止规则见 experiment.json。

## 结果

真实冻结 DURA 首批检查及 672 条源观测完成。幅度三次方向的作用差异高于另外三个人工方向；数据和边界见 [源域分析](analysis.md)。这是源物理代理诊断，不是识别准确率或目标收益。

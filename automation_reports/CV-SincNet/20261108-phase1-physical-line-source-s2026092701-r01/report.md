# 源域物理残差步长诊断

- run_id：`20261108-phase1-physical-line-source-s2026092701-r01`
- group_id：`phase1-physical-line-source`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

冻结 DURA 与源域修正器，比较仅由接收信号和估计信道计算的逐样本修正步长；源 clean 仅用于波形评价和不可部署 oracle。

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

## 诊断方案与运行

冻结第 20 轮源域修正模型与 DURA v1。沿候选修正方向，在 [0,1] 内最小化估计信道下的接收残差，步长只使用 received IQ、固定 DURA 输出和估计信道；同时计算需要源 clean 的理想步长用于比较，后者不能部署。672 条源 pilot 与候选训练 ID 不相交，DURA 是否见过未知；本结果不是身份识别准确率，不影响既定正式测试。

N607 后台 PID 948531，CPU 2 线程、隐藏 GPU；发布包一次 SHA 比较及远端编译通过。两项针对性测试及独立 P0/P1 审查通过，真实源数据首批 8 条 smoke 已通过。无 F、TX 标签、U_s 真值、V 或目标读取，也无模型更新。

- 源域 672/672 条完成；[分组分析](analysis.md)。物理步长在 12 个受损组的平均波形 NMSE 均低于 DURA，逐记录 290 条更低、31 条更高、255 条相同；这不是身份准确率。

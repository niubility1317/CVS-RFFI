# 源域多时滞四阶统计记录几何诊断

- run_id：`20261109-phase1-cumulant-source-geometry-s2026092701-r01`
- group_id：`phase1-cumulant-source-geometry`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

比较固定源物理记录的功率、四阶累积量与等权组合在原输入及 DURA 输出下的匹配；不判断 TX 身份。

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

## 固定源域诊断

固定合法源训练池中前 512 条物理 ID 的干净 IQ 计算特征标准化；后 512 条不同物理 ID 作为配对检索对象，各检查三个预先生成的源域视图。比较接收 IQ 与固定 DURA 输出的 32 频带功率、21 维多时滞四阶统计，以及按特征维数平衡的组合。统计 top-1、top-5、平均倒数排名与匹配距离；不拟合任何网络，不读取 F 权重、F 输出、TX 标签、V 或目标。

干净参考库仅供源域诊断，物理记录检索可能反映调制内容，并非 TX 身份准确率或可部署查询库；源训练池可能为 DURA 所见，本结果也不是新盲测。正在训练的四阶统计恢复器保持固定方案，不按本诊断改变或取消正式七景测试。

GPU7 至少 8192 MiB 空闲、少于 3 个 GPU 进程且本任务同卡最多一 worker；先核对固定 DURA 与源缓存，再运行全部 512×3 对应视图。代码 e20acc7d1f89b7561afecf9584c374f16cc36df6；执行命令、来源与停止规则见 experiment.json。局部特征／检索检查及独立 P0/P1 审查通过，服务器运行待验证。

## 服务器诊断结果

发布包两端 SHA-256 均为 b91f3184bcc77e8e64f40a1c8c0fe5d8cbe7a89c228c70763496c2bf48f9ecf6；远端编译和导入通过。GPU7 启动前空闲 24110 MiB、无计算进程。实际 DURA 与缓存的 8 条源记录核对通过，完整源域匹配结果和日志已生成；任务较短，结束前未抓取独立的存活 PID/GPU 快照。表格及科学边界见 [源域统计分析](analysis.md)。

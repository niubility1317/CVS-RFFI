# BranchMetric类内度量与单样本代理support诊断

- run_id：`20260929-phase2-d92-branch-metric-support-m4-r01`
- group_id：`d92-branch-metric-support-development`；类别：`diagnostic`；阶段：`Phase2-support-only-diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定交互表示，对比岭回归、最近类中心与类内度量；完整物理OOF及当前support内所有1-shot anchors。

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

固定三臂与两类support诊断已预登记；完整4800父任务、3600OOF、42000单样本代理anchors，真K1不伪造holdout。缓存身份、CPU资源及新输出路径只读preflight VERIFIED；尚未发布或启动。

实际算法配置已冻结；核心25项及既有interaction29项、entry/summary44项和3子测试、CPU编排15项通过，集合按范围记录不当成独立总数。分析发布闭包P1已补齐，隔离目录导入验证通过；独立审查收尾中，尚未启动。

运行已独立读回VERIFIED：runtime7099da7ab85170034de90d41c2471c4572860207，supervisor168153及四CPU worker live且argv/CWD一致，readback_1790671911.json。全矩阵结束后一次完整汇总，不按中途表现改变配置。

完整support诊断已ANALYZED：4800父任务、3600OOF、42000proxy、63600分解。OOF A通过，proxy B失败，不推进query或替换现方法。解释见support_interpretation.md；保留全部原始产物和各新增类规模分层。

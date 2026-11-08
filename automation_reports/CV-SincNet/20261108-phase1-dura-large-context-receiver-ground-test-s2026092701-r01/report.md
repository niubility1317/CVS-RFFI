# DURA 源接收机均衡七景测试

- run_id：`20261108-phase1-dura-large-context-receiver-ground-test-s2026092701-r01`
- group_id：`phase1-dura-large-context-receiver`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

归一化 DURA 全局修正，源训练按公开接收机频率反比均衡，不在推理时使用 RX。

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

## 预登记补充

源均衡网络固定第 20 轮权重后，接入冻结的 Tri-SIG-TIME SSL Main split 默认分类头。clean 和六种 practical 各 168000 条，全部预测完成后进行独立 truth-last 评分，与 DURA 逐场景配对比较；同时报告相对 raw clean 的 3 个百分点及相对 3% 两种口径。该测试是已多次观察的地面模拟单种子测试。

## 七场景正式结果

独立评分已完成。六个受损场景均低于 DURA，低城区 62.8327% 对 68.8012%；每景、净纠错和 TX/RX/day 见 [七景分析](../20261108-phase1-dura-large-context-receiver-ground-test-s2026092701-r01/analysis.md)。

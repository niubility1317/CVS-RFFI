# PAMF第三阶段：源域物理分支未知证据

- run_id：`20261029-phase3-source-branch-s2026092701-r01`
- group_id：`pamf-phase3-source-branch`；类别：`cvs`；阶段：`Phase3-source-physical-branch-development`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定自有四类CE20源模型，8分支乘2种距离，共16条源侧规则；零模型更新、零目标访问。

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

## 发布前证据

实际code/config commit：`df34171b8dc9942ec236e4504cac61768d70997a`。95项Phase3测试通过；本候选一次独立P0/P1审查PASS，审查者另运行22项候选测试通过。真实own-checkpoint smoke为launcher首步，仅先读两条合法L，PASS立即继续。既有N607 preflight通过，选择空闲/有余量卡，每卡至多新增2个本任务worker，不停止其他owner。尚无真实本轮分支结果。

## 独立完成读回与结果

VERIFIED/ANALYZED_SOURCE_ONLY。真实源smoke PASS；L3360/V27000八分支导出与16规则均完整，零target/零训练更新/默认头不变。16规则全部NO_PROMOTION；全V最高未知拒识3.378%，RX8最高7.278%，不扩正式target。流水线实测11.137秒。详细结果见docs/PAMF_PHASE3_BRANCH_EVIDENCE_RESULTS_20261029.md；本地analysis_summary.json和完整日志保留。

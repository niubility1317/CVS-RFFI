# D92固定Phase1融合前分支support信息探查

- run_id：`20260929-phase2-d92-branch-support-probe-m4-r01`
- group_id：`d92-fixed-phase1-branch-support-information`；类别：`diagnostic`；阶段：`Phase2-support-only-diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定模型和单原始view，完整support矩阵比较现有表示、重复背景控制、加入时间/频率/PA分支的物理OOF增量；不读取query样本或成绩，不产生部署分类器。

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

PLANNED: complete current-row support-only six-arm probe; core/export/entry and orchestration synthetic validation passed. Resource/output/capsule binding preflight VERIFIED without IQ access or data revalidation. Awaiting unique P0/P1 review and pushed release; not launched.

Publication technical failure before extraction/launch: Windows Path serialized remote archive with backslashes. Independent readback_1790620548.json shows no supervisor/children/startup; remote intended POSIX archive exists and is intact (SHA256 15d195442be83ebb4a285986a217494d2cbe7a158a70d8f8a8c13ba19b345d24). Fix uses PurePosixPath; explicit staged-commit recovery requires absent release/run, matching staged bytes, and identical runtime paths versus the original pushed commit f7f19e87042b22c499ca2cee3d01fd0334dc8d47. No experiment has started; preserve original landing evidence.

RUNNING VERIFIED readback_1790620659.json: supervisor PID2574293 and support exporter PID2574301 live with matching argv/CWD. Actual runtime commit f7f19e87042b22c499ca2cee3d01fd0334dc8d47. Only publication path serialization changed; original runtime archive retained. No repeat launch, no query access.

Progress VERIFIED by readback_1790620714.json: all eight support exports finished; four rx3 CPU probe children are live with exact argv/CWD, rx1 probes queued after exports. No failed lane. Do not repeat launch or read this progress as completion.

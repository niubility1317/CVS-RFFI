# D92固定Phase1融合前分支support信息探查

- run_id：`20260929-phase2-d92-branch-support-probe-m4-r01`
- group_id：`d92-fixed-phase1-branch-support-information`；类别：`diagnostic`；阶段：`Phase2-support-only-diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定模型和单原始view，完整support矩阵比较现有表示、重复背景控制、加入时间/频率/PA分支的物理OOF增量；不读取query样本或成绩，不产生部署分类器。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

完整support探查与全量汇总已VERIFIED。8个模型/cohort行全部结束，覆盖4800个任务，其中1200个K1任务只有数值诊断，3600个任务有物理support OOF；六臂共64800次解析分解，每臂562800条留出预测。未读取query样本、truth或成绩，没有输出部署分类器。

完整证据见[summary.json](results/support_summary/summary.json)、[执行证据](results/support_summary/analysis_execution.json)和[分层解释](support_interpretation.md)。新旧类联合任务中，zfft_aux相对zfft的旧类/新类/H均值增加5.025/10.221/9.464个百分点；相对重复背景对照仍增加4.342/9.047/8.231个百分点。各K、模型seed、support seed、RX/scene与K×新类规模的边际均值均正，但存在任务级退化；K1没有OOF证据。这些是support信息增量，不能写成query改善或全面目标已完成。

远端runtime commit为f7f19e87042b22c499ca2cee3d01fd0334dc8d47；独立分析commit为63aac180dbbd200ba3954929f4a457386aff7fd8，原runtime未改变。readback_1790620888.json确认原进程全部退出。实际run墙钟121.324秒；累计probe调用345.523秒，二者口径不同。新source payload为0B；4份不同完整训练checkpoint合计63971616B，已有部署状态未知，不能当新增卫星传输量。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

PLANNED: complete current-row support-only six-arm probe; core/export/entry and orchestration synthetic validation passed. Resource/output/capsule binding preflight VERIFIED without IQ access or data revalidation. Awaiting unique P0/P1 review and pushed release; not launched.

Publication technical failure before extraction/launch: Windows Path serialized remote archive with backslashes. Independent readback_1790620548.json shows no supervisor/children/startup; remote intended POSIX archive exists and is intact (SHA256 15d195442be83ebb4a285986a217494d2cbe7a158a70d8f8a8c13ba19b345d24). Fix uses PurePosixPath; explicit staged-commit recovery requires absent release/run, matching staged bytes, and identical runtime paths versus the original pushed commit f7f19e87042b22c499ca2cee3d01fd0334dc8d47. No experiment has started; preserve original landing evidence.

RUNNING VERIFIED readback_1790620659.json: supervisor PID2574293 and support exporter PID2574301 live with matching argv/CWD. Actual runtime commit f7f19e87042b22c499ca2cee3d01fd0334dc8d47. Only publication path serialization changed; original runtime archive retained. No repeat launch, no query access.

Progress VERIFIED by readback_1790620714.json: all eight support exports finished; four rx3 CPU probe children are live with exact argv/CWD, rx1 probes queued after exports. No failed lane. Do not repeat launch or read this progress as completion.

ARTIFACTS_COMPLETE VERIFIED readback_1790620888.json: all 8 lanes/4800 episodes complete; no live experiment. Analysis uses separate release d92_branch_support_probe_analysis_20260929_r01 at commit 63aac180dbbd200ba3954929f4a457386aff7fd8; full streaming audit pending.

ANALYZED VERIFIED: full support_summary and support_interpretation.md retained. No live processes, query access or query performance claim. Proceeding with one support-justified branch candidate, not repeating this diagnostic.

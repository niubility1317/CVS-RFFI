# DRIFT唯一复现版本收敛与RIEI继续优化报告

## 任务与边界

- 时间：2026-07-14 17:15+08:00
- 操作者：Codex
- 用户要求：将DRIFT的MSE mean版本固定为唯一复现版本，删除其他版本；继续优化RIEI，目标为复现期刊Table III。
- 删除边界：删除可执行代码面中的旧DRIFT论文复现launcher，不删除Git历史、`code/snapshots`、实验报告、训练日志、metrics、checkpoints或N607 run产物。这些是结果可追溯证据，不再作为可执行版本。
- DRIFT唯一配置：`drift_day1`，batch256，random抽样，每TX/RX训练800、验证200、测试200，仅信道均衡、RMS off，MSE reduction=`mean`、cap=`0`、`lambda_mse=0.020`，200epoch，epoch200 final，五seed=`1337,2024,3407,4242,7777`。
- RIEI目标：期刊Table III完整12行；发现阶段固定第1行`1-1,7-7→1-19`，比较优化器、loss reduction、RMS和feature norm，正式指标为200epoch last5，禁止target-oracle选epoch。

## Traceability

|ID|来源|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|D-01|用户要求|建立唯一DRIFT论文复现入口并固定V206 mean配置|`code/scripts/launch_drift_paper_reproduction.sh`、`baselines/README.md`|verified|`bash -n`、5-job dry-run、聚焦pytest通过|不改变CVS扩展协议入口|
|D-02|用户要求|删除旧DRIFT论文复现launcher|`code/scripts/*paper_repro*20260714.sh`、旧v2 discovery/confirm launcher|verified|本地、Git镜像和N607精确文件清单反向审计通过|保留报告、日志、metrics、checkpoint、run与snapshot作为审计证据|
|D-03|论文v2口径|完整分析五seed epoch200 final并判断稳定性|DRIFT v2报告、本报告|verified|1000条epoch、3495行日志、5份metrics完整解析|final=72.75±5.93%，均值差-0.79pp；SD偏高|
|R-01|RIEI论文Table III|修复last5、RMS、优化器和reduction偏差|RIEI parity代码、wrapper、launcher|verified|`ssr-gpu`聚焦测试15 passed；bash-n/dry-run通过|启动前重新验证远端hash|
|R-02|用户要求|DRIFT结束后启动RIEI 8候选同row消融|N607独立run/log根|verified|容量门、hash、远程bash-n/dry-run及5分钟健康检查均通过|8 trainer分布于GPU0–7，当前硬错误0|
|R-03|RIEI论文Table III|胜出配置重跑完整12行并逐行比较|后续12行确认launcher及RIEI报告|deferred|等待R-02完整200epoch结果|成功阈值MAE≤3pp且≥10/12进入±2SD|

## 当前证据

- DRIFT五seed确认run`paper_repro_drift_v2_confirm_v206_20260714_164900`已5/5完成，训练与queue均退出，远端硬错误0。
- 已拉取并完整解析5×200=1000条epoch记录及15个日志/调度文件共3495行；未见Traceback、RuntimeError、OOM、Killed、NaN或Inf。
- RIEI parity发现矩阵已于17:37在N607启动；17:42时8/8 trainer健康运行，进度epoch13–15/200，硬错误0。

## 本地收敛验证

- 删除的旧可执行入口：`launch_optimizer_20260605_143743_paper_baseline_riei_drift.sh`、`launch_paper_repro_20260605_145347_riei_drift.sh`、`launch_paper_repro_original_matrix_20260714.sh`、`launch_paper_repro_repaired_matrix_20260714.sh`、`launch_paper_repro_fixopt_matrix_20260714.sh`、`defer_launch_paper_repro_fixopt_20260714.sh`、`launch_drift_v2_repair_matrix_20260714.sh`、`launch_drift_v2_confirm_v206_20260714.sh`。
- 唯一新增入口：`code/scripts/launch_drift_paper_reproduction.sh`。
- 验证：canonical launcher `bash -n`通过；5-job dry-run固定mean/no-cap/batch256/random/no-RMS/final；`ssr-gpu`聚焦测试`15 passed`。首次`conda run`因本机GBK包装输出异常失败，按项目规则用同一`ssr-gpu`解释器串行复跑通过。

## RIEI启动前状态

- DRIFT确认run已完全退出；N607仅GPU3保留1个Phase1 compute，RIEI每GPU1个job的容量门可通过。
- RIEI本地代码、launcher、hash和8-job dry-run已重新验证；同步、远程校验、正式启动和5分钟健康检查已完成。

## 反向审计与剩余风险

- 状态计数：`verified=5`、`implemented=0`、`deferred=1`、`blocked=0`、`rejected=0`。
- 唯一延后项是R-03：当前8候选只在Table III第1行选型，必须等完整200epoch last5结果后才能固定配置并执行12行确认。
- 最高风险：DRIFT五seed聚合均值达标但seed标准差`5.93pp`；RIEI尚未产生本轮正式last5结果。二者都不得用target peak或单epoch极值替代论文口径。

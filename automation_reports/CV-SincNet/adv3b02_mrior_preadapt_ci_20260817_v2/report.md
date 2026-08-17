# ADV3B02 MRIOR预适应CI对比v2启动报告

- Run ID：`adv3b02_mrior_preadapt_ci_20260817_v2`
- 状态：`LOCAL_VERIFIED / READY_FOR_N607_RELEASE`
- 冻结提交：`a9df3965fc2a6322b3943603ff98a94082c081cd`
- 目标：对每个`receiver/seed/new_class_count/K/scene`的ADV3B02基座先执行MRIOR-SDA，再以冻结适应模型运行原CSIL/MoPC-HR类增量注册。
- 对照：不可变v7无预适应同排结果；本run不改变CSIL/MoPC-HR参数或注册逻辑。

## v1停止与v2修复

v1在N607首次生成plan时因真实v7不同new-count包的target-old support身份不同而fail-closed停止；未生成plan、run root、预测、评分或性能结果。v2不复用v1路径，改为1200个new-count-specific预适应job，并以全新run ID启动。

## 冻结矩阵与输入

- `protocol_schema=p2_min_v1`；比较方法保留明确source访问例外，新类support/query仍为固定LEO weak IQ。
- 预适应job：`5 receiver×5 seed×4 new-count×4 K×3 scene=1200`；两种downstream方法在同一package内共享。
- 注册cell：800；预期scene row：2400。
- MRIOR锁：200steps、Adam lr=0.0006、estimate=7、target CE=1.0、DV-KL=0.005、mu=0.5。
- v7授权plan：`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_unfrozen_paperfull_ci_20260723_v7/protocol_plan/paper_full_plan_authorized.json`，SHA256=`1c5fb08231bc9d150d625e5360162c3ee287bdf778ac1c16a069ac187b96d65b`。
- source cache：`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_three_da_leoweakonly_20260715_v1/phase1_caches/source/cache_set.json`，SHA256=`dea3bdb01d4f5791d7e92a01dbdcdb7f3d66b26bf134a375264b88eff8c6e4c4`。

## 本地验证与同步

`tests/test_build_adv3b02_mrior_preadapt_ci_plan.py`和`tests/test_run_adv3b02_mrior_preadapt_ci_plan.py`共17项通过；4个目标文件`py_compile`通过；`git diff --check`通过。同步目标为N607项目根中的同名相对路径。N607预检已确认项目根可见、8张RTX3090空闲、无活动compute job。

## N607命令与路径

- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- CWD：`/home/szu2070436088/2510044040/CV-SincNet`
- PLAN：`/home/szu2070436088/2510044040/CV-SincNet/protocol_plans/adv3b02_mrior_preadapt_ci_20260817_v2/mrior_preadapt_ci_plan.json`
- RUN_ROOT：`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_mrior_preadapt_ci_20260817_v2`
- LOG_ROOT：`RUN_ROOT/logs`
- 顺序：生成external PLAN→`prepare`→`preadapt_smoke`运行6个job→`smoke`运行4个cell→PASS receipt授权→8个`preadapt_shard`运行全部1200job。
- GPU：smoke用GPU0；正式8个分片分别使用GPU0至GPU7，每GPU一个本run进程。
- 预期artifact：plan与plan SHA、`run_root_identity.json`、6个smoke预适应artifact/receipt、4个smoke prediction/score/command receipt、`smoke_receipt.json`、8个PID与日志、最终1200个预适应artifact/receipt。

## 停机与判定

仅在P0协议/覆盖/错误hash/错误checkout/launcher-wide fault，或两个不同job在预测前出现相同normalized exception fingerprint时停止本run并标记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。绝不根据accuracy、H、BA或其他性能值停机。fresh-run retry未授权。技术smoke成功不等于性能结果；800-cell全量注册与paired分析在预适应闭合后继续。

## 四状态报告模板

| method | receiver | seed | K | new count | scene | DA0_REG0 old | DA1_REG0 old | DA0_REG1 old/new/H | DA1_REG1 old/new/H | DA effect |
|---|---|---:|---:|---:|---|---:|---:|---|---|---|
| 待artifact | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | `DA1_REG1-DA0_REG1` |

`REG0`的new accuracy与`H_old_new`必须为`N/A`；所有数值必须同排配对，不跨方法拼接。

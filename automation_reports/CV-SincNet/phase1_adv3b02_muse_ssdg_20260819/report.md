# ADV3B02-MUSE-SSDG Phase1最小预登记

## 候选矩阵

| 候选 | 固定基座 | 能力 | seed | epoch | source角色比例 | checkpoint选择 |
|---|---|---|---:|---:|---|---|
| M0 | `ADV3B02_CORE90_SOFT_E200` | 同协议ADV3B02控制；不进入MUSE能力路径 | 392002 | 200 | `0.07/0.63/0.15/0.15` | `final_only` |
| M1 | 同M0 | 基础domain/GRL/self/nuisance | 392002 | 200 | 同M0 | `final_only` |
| M2 | 同M0 | M1+fusion+H/M/L路由 | 392002 | 200 | 同M0 | `final_only` |
| M3 | 同M0 | M2+satellite student+cross-receiver+classification prototype | 392002 | 200 | 同M0 | `final_only` |

四个候选固定同一`tx_rx_day_1_7_2`数据split及`L_s/U_s/V_cal/V_select`角色定义，均以`len(U_s loader)`作为每epoch optimizer step预算。M0只按该长度循环L_s，不读取U_s batch、不计算U_s损失、不创建MUSE state。四臂共同启用ADV3B02 PAIC guard：`enabled=true`、`sat_ce_delta=0.12`、`grad_delta=3.0`、`reliable_drop=0.01`、`cooldown_epochs=1`、`sat_scale=0.75`。

## Commit

- Task 7基线提交：`198ba655a52f04bb63cbca4e92e6dedc936227af`。
- Task 7首轮交付提交：`530b077afb6204205cf5075074d6b471f144bfb8`。
- Fix round 1交付提交：本报告、launcher、训练步预算修复与聚焦测试所在提交；最终OID由Git提交与远端分支回读记录。

## 命令

```bash
bash code/scripts/launch_phase1_adv3b02_muse_ssdg_20260819.sh --only=M0,M1,M2,M3
```

本次Task 7 fix round 1仅执行本地`bash -n`、pytest、`--dry-run --only=M3`及临时fake trainer/evaluator非dry-run控制流，不连接N607、不启动真实训练。

## 环境与CWD

- 计划环境：N607普通账户，`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- 计划CWD：`/home/szu2070436088/2510044040/CV-SincNet`。
- 本地验证环境：`C:/Users/lh594/.conda/envs/ssr-gpu/python.exe`。
- 本地验证CWD：`E:/type10-7/github_publish/CVS-RFFI-repo/.worktrees/adv3b02-muse-ssdg`。

## 输入、输出与GPU

- 输入：`${ROOT}/Dataset_WigSig/ManySig.pkl`。
- 输出根：`${ROOT}/runs/phase1_adv3b02_muse_ssdg_20260819/{M0,M1,M2,M3}`；已存在的候选根禁止覆盖。
- GPU：默认`GPU=0`，所有子命令映射为`CUDA_VISIBLE_DEVICES=${GPU}`与进程内`cuda:0`。

## 停止规则

- 训练命令非零退出、`final_ssdg.pth`缺失或为空时停止当前候选并保留全部产物。
- 联合clean+三LEO评测命令失败或联合artifact为空时写`EVAL_FAILED_JOINT`并保留训练产物；拆分时任一场景缺失、计数非法、准确率与计数不一致或对应日志/metrics为空时停止，写`EVAL_FAILED_<SCENARIO>`。
- 不因中间或最终性能高低停止。

## 预期artifact

每个候选根必须包含非空`train.log`、`config.json`、`final_ssdg.pth`、`eval_clean.log`、`eval_leo_clear_weak.log`、`eval_leo_low_elev_weak.log`、`eval_leo_rain_weak.log`、`metrics_clean.json`、`metrics_leo_clear_weak.json`、`metrics_leo_low_elev_weak.json`、`metrics_leo_rain_weak.json`。真实评测器只调用一次并生成联合JSON；控制层从真实row计数重算四个场景aggregate，四份JSON顶层`scenario`、`aggregate.scenario`和row语义必须一致。仅当四组评测日志与metrics均非空时，`status.txt`才写`ARTIFACTS_COMPLETE`。

## Task 8：追踪闭合与发布准备

### 状态

Tasks 1–7实现已完成Task 8本地聚焦验证和正反追踪。当前结论是`LOCAL_VERIFIED_FOR_RELEASE_PREPARATION`，不是`ARTIFACTS_COMPLETE`或`ANALYZED`：尚未连接N607、未启动M0–M3真实训练，也没有clean或三种LEO弱场景的性能结果。

### 完整聚焦测试

- 环境：`C:/Users/lh594/.conda/envs/ssr-gpu/python.exe`；解释器前缀读回为`C:/Users/lh594/.conda/envs/ssr-gpu`。
- CWD：`E:/type10-7/github_publish/CVS-RFFI-repo/.worktrees/adv3b02-muse-ssdg`。
- 文件映射：brief列出的12个测试文件均实际存在，无需使用等价文件替换。
- 命令：`python -m pytest code/tests/test_muse_ssdg_schedule.py code/tests/test_muse_ssdg_routing.py code/tests/test_muse_ssdg_losses.py code/tests/test_muse_ssdg_memory.py code/tests/test_muse_ssdg_training_heads.py code/tests/test_muse_ssdg_train_integration.py code/tests/test_muse_ssdg_satellite.py code/tests/test_muse_ssdg_checkpoint.py code/tests/test_phase1_muse_launcher.py code/tests/test_meta_ssl_pseudo_gate.py code/tests/test_concat_sat_channel_aug.py code/tests/test_phase1_p1_protocol.py -q`。
- 结果：退出码0；12个文件共收集107项，107项全部通过；测试进程未输出warning，也没有warning升级为error。

### 真实checkpoint无query smoke

- checkpoint：`E:/type10-7/automation_reports/CV-SincNet/qknnv42_strict_dual125_20260714_183556/artifacts/best_joint_safe_ssdg.pth`，8,582,116字节。
- checkpoint SHA-256：`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`。
- 路径发现方式：先在仓库报告中定向检索已登记的ADV3B02路径，再仅对3个精确候选路径执行只读`test -f`；未扫描数据根、未连接N607。
- 输入边界：CPU、1个batch、batch size 2；输入为确定性构造的source-shaped张量；`dataset_read_count=0`、`support_input_count=0`、`query_input_count=0`、`target_truth_read_count=0`。
- 执行结果：严格重建0 missing、0 unexpected；前向输出`tx_logits=[2,6]`、`z_id=[2,160]`且有限；反向、optimizer step、MUSE训练态保存和重新加载全部完成；退出码0。
- 输出artifact：`E:/type10-7/local_artifacts/adv3b02_muse_ssdg_task8_20260820/m3_real_checkpoint_no_query_smoke.pt`，274,118字节；第二个独立进程以`weights_only=True`回读schema、batch数、checkpoint严格加载标志和零query/truth计数，全部一致。
- 非失败关注：模型前向输出1条`torch.cuda.amp.autocast`弃用`FutureWarning`。该warning来自既有`code/model.py`调用，不影响本次退出码和数值有限性；完整聚焦pytest没有输出该warning。

该smoke只验证真实历史ADV3B02 checkpoint与M3训练路径、optimizer和MUSE state回环兼容，不使用真实source batch，也不产生准确率、DG收益、LEO鲁棒性或晋级证据。

### 18项正向追踪与反向审计

- 逐项状态：MUSE-001至014及017为`verified`；MUSE-015、016、018为`implemented`；`pending=0`。
- `implemented`三项的剩余证据均来自真实运行：四场景metrics/log、完整run artifact和实际telemetry/泄漏探针。实现行为已由launcher或telemetry测试验证，但未用fake artifact替代真实结果。
- 汇总：总要求18，`verified=15`、`implemented=3`、`pending=0`；实现追踪闭合18/18。
- 生产文件审计范围：`0e1019beb8f9c3217b4ae84f1a56a4be6dd5ba9e..4c66489ea058f5fe8401c29a237a58708bd7451f`。
- 反向结果：`code/cvsrffi/muse_ssdg.py`映射MUSE-003至012、017；`code/SSDG/train_ssdg.py`映射MUSE-001至013、017、018；launcher映射MUSE-014至016。3/3个新增或修改生产文件均有规范来源，未发现需要删除或重新审批的规范外生产逻辑。
- 完整逐项证据见`analysis/adv3b02_muse_ssdg_traceability_20260819.md`。

### 单一release归档准备

拟定归档名：`adv3b02_muse_ssdg_code_4c66489ea058.tar.gz`。归档仅包含Tasks 1–7在代码身份`4c66489ea058f5fe8401c29a237a58708bd7451f`下的3个生产文件：

1. `code/cvsrffi/muse_ssdg.py`
2. `code/SSDG/train_ssdg.py`
3. `code/scripts/launch_phase1_adv3b02_muse_ssdg_20260819.sh`

Task 8不创建归档、不执行SSH/SCP、不启动实验。后续唯一runner在N607 preflight通过后创建这一个归档，只做一次本地/远端归档SHA比较，并在远端执行两个Python文件的`py_compile`和launcher的`bash -n`；不增加成员SHA、seal、receipt或额外发布gate。

### Task 8关注与下一状态

- 最高风险剩余项是尚无真实M0–M3单seed训练及其clean/三LEO逐场景结果；不能据当前本地证据判断性能、晋级或发表价值。
- MUSE-015、016、018必须在真实训练与评测后从`implemented`更新为`verified`，并把同row结果追加到本报告。
- release归档创建、N607资源/路径preflight、单次SHA比对、远端编译、启动后PID/CWD/cmdline/GPU/log增长检查均留给后续唯一runner；本Task未越权执行。

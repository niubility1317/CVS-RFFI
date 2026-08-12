# D92 E0 FULL D42 TPCE Pareto-safe Hard11 v3实验报告

## 1.实验身份

|字段|冻结值|
|---|---|
|run ID|`d92_e0_full_d42_tpce_hard11_20260812_v3`|
|候选|`E0_FULL_D42_TAIL_PAIR_CODE_EXCHANGE`；revision=`pareto_safe_greedy_v2`|
|科学commit|`c6f1ff0f9f9ef5e8eebb5c0e8e8023256463ba0d`|
|目标|在10个最难performance outer上，相对E0_FULL_ONLY让八项指标同排严格改善，同时维持低注册开销|
|矩阵|10 performance+1 K1 liveness；11 jobs；33 scene-arm；8 shards|
|协议|`p2_min_v1`；复用`VALIDATED_ONCE`；query零fit/update/selection/truth/role/quota/global reassignment|
|状态|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`|

## 2.假设与比较目标

v2真实K10 smoke证实：原66个原子同步发布会提高pooled-new tail且不增加双向hinge，但会持续降低部分旧类tail。v3保持同一E0 FULL fit、D42原子、固定lower-Q20 tails和全部阈值，只改为确定性Pareto安全贪心子集：每个候选只改两个`coef2_qint8`单元；每个accepted prefix均materialize真实D42 state、运行真实`_score`并复核六旧类tail、pooled-new cross/all和双向hinge；最终再次复核。无安全子集或任何真实守卫失败即byte-exact回退E0。

唯一performance比较基线是完整125的E0_FULL_ONLY同outer历史artifact。八项晋级方向为：`H_old_new`、old balanced accuracy、`c_old_acc`、old floor、seen-new均提高；average forgetting、new→old、old→new均降低。Hard10只决定是否进入完整125，不形成正式推广结论。

## 3.本地实现与验证

变更覆盖TPCE核心、query正式收据、runner计数闭包、method lock及设计追溯。K>2仍为two-state fit=2、actual FULL fit=1；TPCE新增fit=0；K1严格D92 FULL alias 3/3；query MAC和persistent state不变。

- 聚焦回归：93 passed。
- `py_compile`、config JSON、runner/analyzer CLI、`git diff --check`通过。
- 独立审查：P0=0、P1=0、APPROVE。
- C=26、K=10、156原子本地固定基准：5次核心wall=`[35.017,34.797,38.937,32.868,34.870]`ms，中位34.87ms；active，selected=4。该基准仅是实现资源预筛，不是N607性能结论。
- 资源收据包含最坏三角候选两列解析评估、每个accepted prefix和final的完整真实D42 support评分、坐标比较及瞬时内存包络。

## 4.发布输入

|文件|大小|SHA256|
|---|---:|---|
|`d92_tpce_runtime_closure_c6f1ff0f.tar.gz`|5,100,979B|`24aee361d90ec3840d87fe1d2bc19bea299ea0c5e57921ec07eb931dd842a9b4`|
|`configs/stage2_d92_full_d42_tpce_hard11_v1.json`|7,177B|`c57f382c3bd31754c0f391fa18b5b2f615ee0b243b0744ac989e47b0917f126d`|
|`launch.sh`|3,764B|`2c69005d67e35d4dee2defd92543e9af41764df0dcef14181ba0b15d2020b47a`|

本地`bash -n`因Windows路径映射失败，不能作为脚本失败；runner必须在N607落地后以远端`bash -n`确认。

## 5.N607冻结路径与命令

- source：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_tpce_source_snapshot_c6f1ff0f_20260812_v3`
- output：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_d42_tpce_hard11_20260812_v3`
- logs：`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_d42_tpce_hard11_20260812_v3`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 唯一启动命令：

```text
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_tpce_source_snapshot_c6f1ff0f_20260812_v3 && nohup bash ./launch.sh >./launch_driver.out 2>./launch_driver.err </dev/null &
```

## 6.发布门、停止规则和预期artifact

### 6.1 Runner落地记录（2026-08-12）

- 直连N607只读预检通过：普通账号`szu2070436088`、项目根可见、8张RTX3090可见且启动前约1MiB显存/卡。
- source/archive、config和launch已按冻结映射串行SCP；远端尺寸、SHA256、archive仅`code/`入口、config JSON、`bash -n`、冻结Python（3.10.19，torch 2.1.0+cu121，CUDA可用）和context检查均通过。
- 机械异常：预创建的本run output/log空目录触发`launch.sh`的`test ! -e`门；已只读证明两根目录为空、非目标run、无同run进程后，以精确非递归`rmdir`移除，并复核为`ABSENT`。未删除source、文件或其他run；不计重试。

K10真实checkpoint truth-free smoke在8shard前执行。三场景必须TPCE active、无fallback、prefix/final真实support guard通过、fit=2/1、四项原子计数闭合、state/prediction/COMMIT闭合且query所有禁用访问为false，才允许启动Hard10。系统性停止仅限协议/安全、wrong checkout/hash、overwrite、launcher确定性错误或至少两个distinct outer同一pre-prediction异常；绝不按中间性能停止。`fresh_run_retry=false`。

成功预期：11 job receipts、22 formal before/after prediction+COMMIT+fit/resource/execution、11 score、8 shard summaries；smoke另有2套诊断artifact。完成后完整取回source/output/logs及manifest引用truth sidecar，再由主代理执行独立analyzer。

## 7.性能与资源裁决

八项均值必须严格朝优方向，且达到冻结幅度门：H≥+1pp、old BA≥+1.5pp、`c_old_acc`≥+1pp、old floor≥+4pp、seen-new≥+0.5pp、forgetting≤-1.5pp、两向混淆各≤-0.5pp。任一方向未改善直接`REJECT_ROUTE`；方向全对但幅度、稳定性或target资源未全过为`REVISE_ONCE`；全部门通过才`ADVANCE_TO_TARGET125_CANDIDATE`。

资源硬门：registration wall P90≤150ms、同排wall ratio中位≤1.50、peak≤E0+512KiB、query MAC/state exact、D92 component-fit proxy降幅≥80%；target为P90≤120ms、ratio≤1.25。

## 8.Runner closure（2026-08-12）

### 8.1最终状态

`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。唯一detached命令只执行一次；prepare成功，K10 truth-free smoke失败，8个formal shard未启动。未运行analyzer、scorer或任何性能解释；`fresh_run_retry=false`。

### 8.2失败触发与smoke结构证据

`logs/smoke.err`保留完整异常：`D92D92TPCEHard11RunnerError: fit audit TPCE K>2 candidate did not activate`。三场景after fit audit均为`d92_e0d_tpce_active=false`、`fallback_active=true`、`fallback_reason=support_guard_failed`、`support_guard_pass=false`、`applied=0`；query fit/update/selection/truth/role/quota/global字段均为`false`。smoke输出保留before/after两套各5项诊断artifact（prediction、COMMIT、fit、resource、execution），不将其视为formal结果。

三场景均生成66个pair-exchange原子，greedy分别选择39/35/39个，但最终严格七组tail门仍未闭合：clear的最差旧类gain为`-0.0007648468`且pooled-new gain为`-0.0001022339`；low_elev分别为`-0.0004673004`和`-0.0003410339`；rain最差旧类gain为`-0.0006389618`。双向hinge delta均为0。注册wall分别约204.626/197.028/208.356ms，已超过150ms硬门。因此本轮同时给出科学不可行性和资源失败证据，不进入Hard10。

### 8.3完整取回与闭包

远端source、output、logs已完整SCP到`E:\type10-7\local_artifacts\d92_e0_full_d42_tpce_hard11_20260812_v3`，并按独立目录整理；manifest引用的11个truth sidecar已逐SHA取回到`truth_sidecars/`。远端/本地树统计一致：source=1337 files/70,881,759B/tree SHA `f5ddd6c660d87ff2620c5250e7087b9203fd2e1e81cc1169ac71fd6431fbad63`；output=13 files/762,506B/tree SHA `1a892fcf82d5df0708eec89d7f89232079cbafe69fd55657d8b1292f38b1ec0f`；logs=6 files/1,824B/tree SHA `9534edd70ffe0e9f262901205e62cce1a193df7b519379e233f74a48a6a3e7a7`。truth sidecar=11 files/5,306,510B，manifest逐项SHA匹配。启动前误创建的本run空output/logs目录已在launch前精确非递归移除；未删除source、文件或其他run。

### 8.4资源与清理

launcher退出后无同run进程；8张GPU均回到约1MiB显存、0%利用率。每次SSH/SCP后本地`ssh.exe`/`scp.exe`进程及到N607/桥接主机TCP22均为0。远端artifact全部保留。

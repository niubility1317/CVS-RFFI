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
|状态|`LOCAL_VERIFIED / READY_TO_LAND`|

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

K10真实checkpoint truth-free smoke在8shard前执行。三场景必须TPCE active、无fallback、prefix/final真实support guard通过、fit=2/1、四项原子计数闭合、state/prediction/COMMIT闭合且query所有禁用访问为false，才允许启动Hard10。系统性停止仅限协议/安全、wrong checkout/hash、overwrite、launcher确定性错误或至少两个distinct outer同一pre-prediction异常；绝不按中间性能停止。`fresh_run_retry=false`。

成功预期：11 job receipts、22 formal before/after prediction+COMMIT+fit/resource/execution、11 score、8 shard summaries；smoke另有2套诊断artifact。完成后完整取回source/output/logs及manifest引用truth sidecar，再由主代理执行独立analyzer。

## 7.性能与资源裁决

八项均值必须严格朝优方向，且达到冻结幅度门：H≥+1pp、old BA≥+1.5pp、`c_old_acc`≥+1pp、old floor≥+4pp、seen-new≥+0.5pp、forgetting≤-1.5pp、两向混淆各≤-0.5pp。任一方向未改善直接`REJECT_ROUTE`；方向全对但幅度、稳定性或target资源未全过为`REVISE_ONCE`；全部门通过才`ADVANCE_TO_TARGET125_CANDIDATE`。

资源硬门：registration wall P90≤150ms、同排wall ratio中位≤1.50、peak≤E0+512KiB、query MAC/state exact、D92 component-fit proxy降幅≥80%；target为P90≤120ms、ratio≤1.25。


# D92-BE-2x2-Hard12-v1实验报告

|字段|值|
|---|---|
|实验ID|`d92_be_2x2_hard12_20260811_v1`|
|登记时间|2026-08-11 15:03:40+08:00|
|操作方|Codex primary；N607唯一runner待交接|
|当前状态|`ANALYZED_NO_STRICT_PARETO_PROMOTION`|
|目标|在D92共同路径内验证删除注册后B/E能否同时提升`H_old_new`并降低注册计算|
|声明范围|`DEVELOPMENT_ONLY_COVERAGE_CONSTRAINED_STRESS_SCREEN`|

## 假设与比较对象

四臂为`FULL/B0/E0/B0E0`。所有臂固定A=288维联合特征、C=旧/新0.5/0.5任务均衡协方差、D=full/block3+LOO可靠性融合和F0 FP32仿射头；B/E只在注册后且`K>2`时切换，注册前及`K<=2`走FULL精确路径。`B0E0`是唯一可晋级候选，只有严格性能门和严格资源门全部通过才可进入完整Target125确认。

## 矩阵与协议

- 协议：`p2_min_v1`；复用已有`VALIDATED_ONCE`数据和封存D92 package，不重复数据验证。
- 上下文：`target125_context.json`，SHA256=`067a6365e9c859161407ab62ba6349d7beb93083f59f78fd7c780c6d8924731f`。
- Hard12 selection SHA256=`95d94d586f5084d4982d67ec6402c4244f80e818ef3f95a5a03771085a6885a4`。
- 12outer×4arm=48job；每job固定`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`，合计144 scene-arm。
- 两条K1只做liveness；其余10outer进入严格性能与资源判定。
- query逐样本面对全部已注册类；zero-fit、zero-update、zero-selection；预测子进程不接收truth路径，评分子进程只在before/after prediction提交后读取truth。

## 本地版本与改动

- 工作树：`E:\type10-7\code\snapshots\d92_125wt`。
- 分支：`codex/d92-be-hard12-strict-pareto-20260811`。
- 方法/runner提交：`afde865b123f7236e5d3724745f223773093c215`；严格汇总器提交：`83fe1565af690d130f46d66ab856700c7aa1a2b8`。
- 预注册报告/启动器提交：`a8e310c2`。
- method lock：`configs/stage2_d92_be_2x2_hard12_v1.json`，SHA256=`282d4343adcecc124d76bcbafae8d3c473d7301d0a455797f1bbce57bb2af520`。
- 关键实现：注册资源计量、D92注册后B/E开关、arm truth-free evaluator、Hard12选择与package定位、prediction/scorer隔离启动器。
- 根目录`E:\type10-7`不是Git仓库；本报告已从Git工作树逐字镜像到根目录要求路径。

## 本地验证

在`ssr-gpu`环境串行运行39项聚焦测试，结果全部通过：

```text
python -m pytest tests/test_stage2_registration_resource_probe.py
  tests/test_stage2_d92_registration_balanced_covariance.py
  tests/test_probe_d92_registration_balanced_covariance.py
  tests/test_stage2_d92_be_slim.py
  tests/test_stage2_d92_be_query_evaluation.py
  tests/test_run_d92_be_prediction.py
  tests/test_stage2_d92_be_hard12.py
  tests/test_run_d92_be_hard12.py
  tests/test_stage2_d92_be_analysis.py
  tests/test_stage2_d92_role_oracle_query_evaluation.py
  tests/test_run_d92_role_oracle_125.py -q
结果：39 passed
```

新增和修改的10个Python入口已通过`py_compile`，`git diff --check`通过。Hard12 manifest本地只读构造闭合为48job、144 scene-arm；远端`prepare`会对实际12个源job的四组seal和truth sidecar做存在性验证。

独立Terra/max审查在提交`25e68eaf`上给出`P0=0、P1=0、VERDICT=APPROVE_RELEASE`，并独立复跑同一38项聚焦测试全部通过。P2仅要求runner在首波健康检查中汇总8个shard异常指纹，不阻塞发布。

## N607落地与运行锁

|项目|冻结值|
|---|---|
|远端Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|项目根|`/home/szu2070436088/2510044040/CV-SincNet`|
|源码快照|`runs/d92_be_source_snapshot_20260811_v1`|
|上下文|`runs/d131_d92_lite160_qtie_target125_20260804_r3/prepared/target125_context.json`|
|真实smoke输出|`runs/d92_be_truthfree_smoke_20260811_v1`|
|矩阵输出|`runs/d92_be_2x2_hard12_20260811_v1`|
|日志根|`logs/d92_be_2x2_hard12_20260811_v1`|
|GPU|shard0–7分别绑定GPU0–7，child使用`cuda:0`|
|CPU|每job BLAS/OMP线程2，interop线程1|

实际启动器为同目录`launch.sh`。runner先执行一次本地N607只读预检，检查输出根不存在和当前GPU/进程占用。为避免Python完整包覆盖局部namespace导致新模块不可见，runner把当前Git提交中的完整`code/cvsrffi`和`code/scripts`打成一个运行闭包归档，同步到全新源码快照；另同步method lock和启动器。启动器先核对这一个归档、确认`cvsrffi/scripts`目标尚不存在，再解包并执行所有本轮入口都来自新快照的真实import closure；method lock保留单独身份核对，不做整树或逐文件SHA。随后依次执行：

|本地文件|N607目标|最低身份值|
|---|---|---|
|`E:\type10-7\code\snapshots\d92_be_runtime_closure_bf05869b.tar.gz`|`runs/d92_be_source_snapshot_20260811_v1/runtime_closure_bf05869b.tar.gz`|SHA256=`183b04e256ef94a4a946bf17b114ce0de51606f94751af9681b0ea700af61c04`|
|`configs/stage2_d92_be_2x2_hard12_v1.json`|`runs/d92_be_source_snapshot_20260811_v1/configs/stage2_d92_be_2x2_hard12_v1.json`|SHA256=`282d4343adcecc124d76bcbafae8d3c473d7301d0a455797f1bbce57bb2af520`|
|同目录`launch.sh`|`runs/d92_be_source_snapshot_20260811_v1/launch.sh`|SHA256=`fa0b0cb01ac856d860132246c85a55f841d9abfdc15cd391b8dadc4a86bc5730`|

1. `prepare`：验证12个源D92 job并独占写48job manifest；
2. `smoke`：`rx_3_19__seed_713104__k_1__new_20/FULL`真实sealed checkpoint链，无truth参数；
3. smoke通过后才启动8个shard；primary不重复启动。

## 期望artifact

- `matrix_manifest.json`；
- 8份shard events、summary及外层stdout/stderr；
- 48份`job_receipt.json`；
- 每job两份prediction、COMMIT、fit/resource audit；
- 每job一份独立`diag_cosine_score.json`；
- 完成后本地生成严格Pareto `summary.json/gates.json`及配对表。

## 健康停止与成功标准

只在P0协议/安全违规、错误checkout或输入身份、输出覆盖风险、launcher级确定性故障，或至少两个不同row在prediction前出现相同异常指纹时技术停止。不得依据H、accuracy、floor或遗忘中间值停止。

`B0E0−FULL`必须同时满足：平均`ΔH>=0.005`、至少8/10 outer的`ΔH>=0`、平均旧类balanced accuracy/seen-new/old floor均不下降、平均forgetting不增加、`DA0_REG0`预测逐值相同、K5/K10 fit数48→24和88→44、配对注册wall与增量peak working set中位降幅均至少40%、query路径与MAC不增加。任一失败即`NO_STRICT_PARETO_PROMOTION`。

## 已知风险

- 历史B/E效应很小，`B0E0`可能只通过效率门而无法提升0.5pp H；不得因此放宽阈值。
- K5上下文曾以K10 pool表达；本实现明确定位原D92的真实K5 job目录，避免误把K10 package当K5运行。
- RSS为注册调用期间当前进程working set的1ms采样增量；只与同outer同节点同线程的FULL配对比较，不写成端到端推理内存。

## 运行后更新区

### 2026-08-11 15:34:06+08:00启动与首波

- 直连N607只读预检、GPU占用检查和四个不可覆盖目标检查通过；没有使用管理员账号或bridge。
- 三项最小同步身份全部匹配；远端关键入口`py_compile`通过，启动器完成归档解包和9模块新快照import closure。
- `prepare`生成48job manifest；真实sealed checkpoint、无truth的K1/FULL smoke通过，receipt确认query truth/fit/update/selection均为false。
- driver PID=`1755987`，CWD、cmdline和source root绑定一致；完成编排后正常退出。
- shard0–7分别启动为PID`1756430`至`1756437`，各自绑定GPU0–7；driver退出后shard转为PPID1，仍属预期detach状态。
- 首波已有20份`job_receipt`、40份prediction COMMIT；shard4已`4/4 PASS`，总体失败数0，stderr异常数0，无重复pre-prediction异常指纹、无P0/P1。
- GPU显存约544–564MiB/卡，GPU利用率抽样0%–58%，未超过冻结并发映射。runner继续使用短连接监控到8个shard全部结束。

完整结束后继续补充artifact closure、详细四臂同row表、严格Pareto门和最终建议。

### 2026-08-11 15:46:45+08:00完成与artifact closure

- 48/48份`job_receipt`、96/96份prediction COMMIT、96/96份prediction、48/48份score、96/96份fit audit、96/96份resource audit全部闭合。
- 8/8个shard summary均为`PASS`，`failed_job_count=0`、`failures=[]`；shard0–3各8/8，shard4–7各4/4。
- 所有远端`.err`为空，无`Traceback/Exception/ERROR/FAILED`和重复异常指纹；最终run-owned进程为0，GPU pmon为空。
- 产物已非覆盖式取回至`E:\type10-7\local_artifacts\d92_be_2x2_hard12_20260811_v1`；权威分析为其`analysis_r2`。第一次`analysis`保留为历史输出，其唯一差别是把数学零`-5.55e-18`误标为old-floor失败；提交`53b899a8`用`1e-12`边界容差修复显示，严格总裁决未变化。
- 无训练过程，故best epoch/checkpoint为`N/A`；四臂共用同一sealed Phase1 checkpoint和同一support/query输入。

### 四臂同row汇总

下表仅汇总10个performance outer的`DA0_REG1`，每个outer先对3个LEO场景等权，再跨outer平均；两条K1只用于liveness。耗时和增量working set为每臂跨outer的中位描述，不等同端到端query时延。

|候选|机制|H_old_new|旧类balanced accuracy|旧类floor|seen-new|forgetting|注册wall中位|增量peak中位|K5/K10 fit|结论|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|FULL|B开、E开|66.8562%|68.9444%|37.6667%|65.3667%|16.9444pp|2998.84ms|2.328MiB|48/88|参考臂|
|B0|仅删B ground center|66.8067%|68.8889%|37.5000%|65.3333%|17.0000pp|2934.96ms|2.430MiB|48/88|拒绝：几乎不省计算且略退化|
|E0|仅删E Fisher/Pareto|66.8448%|68.9444%|37.8333%|65.3417%|16.9444pp|1487.11ms|1.594MiB|24/44|保留为效率开发基线，不是性能晋级|
|B0E0|同时删B/E|66.8203%|68.9444%|37.6667%|65.3083%|16.9444pp|1438.97ms|1.430MiB|24/44|效率门通过，性能门失败|

### 因果解释臂对FULL

|臂|mean ΔH|H不差outer|mean Δold BA|mean Δseen-new|mean Δold floor|wall中位降幅|peak中位降幅|解释|
|---|---:|---:|---:|---:|---:|---:|---:|---|
|B0|−0.0496pp|5/10|−0.0556pp|−0.0333pp|−0.1667pp|6.88%|−1.34%|B不是有效算力杠杆，删除后内存还略增|
|E0|−0.0114pp|8/10|0.0000pp|−0.0250pp|+0.1667pp|54.04%|30.15%|E是主要效率杠杆，性能近似持平但未超过FULL|
|B0E0|−0.0359pp|5/10|0.0000pp|−0.0583pp|0.0000pp|58.02%|40.33%|删除B扩大E0的性能损失，只能效率-only|

### B0E0对FULL逐outer配对

|outer|K|ΔH|Δold BA|Δseen-new|Δold floor|Δforgetting|wall降幅|peak降幅|fit FULL→B0E0|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`rx_20_1__seed_713105__k_10__new_20`|10|0.0000pp|0.0000pp|0.0000pp|0.0000pp|0.0000pp|50.47%|44.86%|88→44|
|`rx_20_1__seed_713106__k_5__new_20`|5|0.0000pp|0.0000pp|0.0000pp|0.0000pp|0.0000pp|58.51%|32.21%|48→24|
|`rx_3_19__seed_713103__k_10__new_10`|10|+0.0909pp|+0.2778pp|0.0000pp|0.0000pp|−0.2778pp|59.08%|−9.65%|88→44|
|`rx_3_19__seed_713106__k_10__new_20`|10|+0.1087pp|0.0000pp|+0.1667pp|0.0000pp|0.0000pp|58.20%|47.50%|88→44|
|`rx_7_14__seed_713102__k_10__new_20`|10|+0.0529pp|+0.2778pp|−0.1667pp|0.0000pp|−0.2778pp|58.11%|40.38%|88→44|
|`rx_7_14__seed_713102__k_5__new_20`|5|−0.0322pp|−0.2778pp|+0.1667pp|0.0000pp|+0.2778pp|61.75%|45.17%|48→24|
|`rx_7_7__seed_713104__k_10__new_5`|10|−0.0443pp|+0.2778pp|−0.3333pp|+1.6667pp|−0.2778pp|57.94%|51.46%|88→44|
|`rx_7_7__seed_713105__k_10__new_5`|10|−0.1841pp|0.0000pp|−0.3333pp|0.0000pp|0.0000pp|50.40%|33.77%|88→44|
|`rx_8_8__seed_713104__k_10__new_10`|10|−0.1575pp|−0.2778pp|0.0000pp|−1.6667pp|+0.2778pp|50.88%|26.71%|88→44|
|`rx_8_8__seed_713105__k_5__new_20`|5|−0.1934pp|−0.2778pp|−0.0833pp|0.0000pp|+0.2778pp|53.67%|40.27%|48→24|

### 严格Pareto门

|门|观测|阈值|结果|
|---|---:|---:|---|
|mean ΔH|−0.0359pp|≥+0.5000pp|FAIL|
|ΔH非负outer|5/10|≥8/10|FAIL|
|mean Δold BA|0.0000pp|≥0|PASS|
|mean Δseen-new|−0.0583pp|≥0|FAIL|
|mean Δold floor|数学零|≥0|PASS|
|mean Δforgetting|0.0000pp|≤0|PASS|
|注册wall中位降幅|58.02%|≥40%|PASS|
|增量peak中位降幅|40.33%|≥40%|PASS|
|K5/K10 fit|48→24、88→44|精确闭合|PASS|
|query MAC增加|0|≤0|PASS|
|`DA0_REG0`预测|四臂逐值一致|一致|PASS|
|K1 alias|两outer四臂逐值一致|一致|PASS|

最终裁决：`NO_STRICT_PARETO_PROMOTION`。`B0E0`不能进入完整Target125确认，也不能表述为性能提升；它只证明注册wall和增量working set可同时下降。`E0`是更合理的下一轮效率底座：保留B、删除E，在几乎不损H的同时将注册耗时减半。下一轮只研究E0条件下的D几何瘦身/LOO support-only选择，目标是用单几何进一步降计算并将H推过FULL；本Hard12仍是`DEVELOPMENT_ONLY`，不得替代正式完整125确认。

### 异常与边界

两次异常均为runner只读监控命令的本地封装问题：一次远端引号EOF，一次`nvidia-smi`格式解析错误；均未改变远端状态、未遗留SSH连接，随后用LF-only/bash-s与简化GPU查询完成取证。实验本身无异常、无重启、无停止、无覆盖。`DA1_REG0/DA1_REG1`为`N/A`，因为本轮是head-only消融，没有引入域适应。

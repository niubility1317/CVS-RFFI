# D92-BE-2x2-Hard12-v1实验报告

|字段|值|
|---|---|
|实验ID|`d92_be_2x2_hard12_20260811_v1`|
|登记时间|2026-08-11 15:03:40+08:00|
|操作方|Codex primary；N607唯一runner待交接|
|当前状态|`LOCAL_VERIFIED_P0P1_REVIEWING`|
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

在`ssr-gpu`环境串行运行38项聚焦测试，结果全部通过：

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
结果：38 passed
```

新增和修改的8个Python入口已通过`py_compile`，`git diff --check`通过。Hard12 manifest本地只读构造闭合为48job、144 scene-arm；远端`prepare`会对实际12个源job的四组seal和truth sidecar做存在性验证。

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

待runner返回后补充PID/GPU/日志、artifact closure、详细四臂同row表、异常、严格门和最终建议。

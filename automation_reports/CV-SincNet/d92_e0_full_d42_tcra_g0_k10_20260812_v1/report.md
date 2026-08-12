# D92 E0 FULL D42 TCRA K10 G0实验报告

## 1.实验身份

|字段|冻结值|
|---|---|
|run ID|`d92_e0_full_d42_tcra_g0_k10_20260812_v1`|
|候选|`E0_FULL_D42_TAIL_CLASS_ROW_ASCENT`；candidate=`d92_e0_full_d42_tail_class_row_ascent`|
|科学commit|`b2934f62da56035ccd80032798280b6c7d2d74b7`|
|日期/执行者|2026-08-12；主代理设计与裁决，唯一N607 runner负责落地|
|状态|`LOCAL_VERIFIED / READY_TO_LAND`|
|目标|仅在固定K10真实checkpoint上验证三场景TCRA是否active、协议闭合且registration wall P90≤150ms|
|声明边界|`TRUTH_FREE_MECHANISM_G0 / NO_PERFORMANCE_RESULT`；不运行scorer，不读取accuracy/H/BA/floor/forgetting/混淆|

## 2.假设与冻结机制

TPCE v3在真实K10上生成66个pair原子并选择35–39个，仍有旧类或pooled-new tail为负且wall为197–208ms。TCRA移除同步降低competitor code的干扰：在E0 FULL一次fit得到的D42 state上，每个非空`class×block`只生成一个真实类行上升原子。tail由E0 support固定一次；每个拟接受prefix及final均以真实D42解码全头score复核六旧类tail、pooled-new cross/all和双向hinge；任何最终门失败均逐字节回退E0。

本地C26/K10可行性fixture中，TCRA generated=78、selected=7、active；7次核心wall为26.319–27.210ms，中位26.730ms。该结果只证明实现可行和资源预筛，不是目标数据性能。

## 3.固定数据与输入

- 协议：`p2_min_v1`；复用`VALIDATED_ONCE`，不重复数据验证。
- outer：`rx_7_7__seed_713106__k_10__new_5`；receiver=`7-7`；seed=`713106`；K=`10`；new class count=`5`。
- 场景：`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`。
- ground component：`/home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component`；manifest SHA256=`15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c`。

|sealed输入|SHA256|
|---|---|
|before enrollment seal|`e3da38668a1e6ec4053e65e669e2a1845bb43198891644d830950e4550b5cea9`|
|before apply seal|`736852188c32255647b8105bc7a68d4cc92ca73615e4734d0ed5f4bdd0f04473`|
|after enrollment seal|`2600a21ee9a2f95a8d17fa1f4d2263b0e04d243424e3257474502953ed6d9286`|
|after apply seal|`afbdc2ebae59fcc311b0cd44aafd27898d7c4af65c9ef03c1085154c8d13020a`|

## 4.本地实现与验证

科学改动仅覆盖TCRA core、E0D slim/query接线及三份测试。K>2 two-state component fit=2、actual FULL fit=1、TCRA postprocess fit=0；K≤2保持FULL exact alias。query fit/update/selection/truth/role/quota/global reassignment均为false，query MAC/state不增加。

- TDD：模块/行为/接线先RED，最终TCRA及相邻TPCE回归`67 passed in 30.21s`。
- fast/reference：最终`coef2_qint8`与完整receipt byte-exact一致。
- `py_compile`、`git diff --check`、冲突标记检查通过。
- 独立审查：P0=0、P1=0、`APPROVE`。
- 设计追溯：`analysis/d92_d42_tcra_traceability_20260812.md`。
- G0计划：`docs/superpowers/plans/2026-08-12-d92-d42-tcra-g0.md`。

## 5.发布输入和路径

|文件|大小|SHA256|
|---|---:|---|
|`d92_tcra_g0_runtime_b2934f62.tar.gz`|6,071,944B|`aeed383cb79892cc4c84c9a02bf9bc543503962dcffc10bbb3299c4bd94bb973`|
|`launch.sh`|8,635B|`9b0bfbc297c82febc21c0b3d0bce35218826e21c2a83722f75df6c0281987dd0`|

- remote source：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_tcra_g0_source_b2934f62_20260812_v1`
- remote output：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_d42_tcra_g0_k10_20260812_v1`
- remote logs：`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_d42_tcra_g0_k10_20260812_v1`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- GPU：GPU0；`CUDA_VISIBLE_DEVICES=0`、内部`--device cuda:0`；CPU threads=2。

唯一detached命令：

```text
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_tcra_g0_source_b2934f62_20260812_v1 && nohup bash ./launch.sh >./launch_driver.out 2>./launch_driver.err </dev/null &
```

## 6.G0健康门和期望artifact

launcher先验证archive、四个detached seal、ground manifest、运行时import closure和三条不可覆盖路径。唯一prediction入口不接收truth/scorer参数。预期输出为`diag/before`与`diag/after`下的`prediction_artifact.npz`、`COMMIT.json`、`execution_receipt.json`、`fit_audit.json`和`resource_audit.json`，以及logs中的`g0_validation.json`。

三场景必须同时满足：TCRA active；无fallback；final state SHA不等于E0；仅`coef2_qint8`改变；六旧tail与pooled-new cross严格大于冻结tol；pooled-new all不降；双向hinge不增；fit=2/1+postprocess=0；query七项禁用访问为false；registration wall最近秩P90（3行即max）≤150,000,000ns。任何一项失败只形成G0负结果，不进入Hard11，不读取性能，不以同run重试。

技术停止只限wrong hash/checkout、overwrite、协议泄漏、launcher/预测异常、OOM/NaN、缺prediction closure。runner必须完整取回source/output/logs并核树hash，结束后清理SSH连接和确认GPU释放。

## 7.后续决策

- `D92_TCRA_G0_ACTIVE_RESOURCE_PASS`：随后才实现独立单臂Hard11（10 performance+1 K1 liveness），并以八项同排指标裁决。
- 任一场景fallback或P90>150ms：记录并拒绝/修订TCRA，不建设Hard11。
- G0绝不形成“优于E0_FULL_ONLY”的性能声明。

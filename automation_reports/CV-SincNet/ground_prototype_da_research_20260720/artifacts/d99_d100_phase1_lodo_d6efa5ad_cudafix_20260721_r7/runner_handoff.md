# D99/D100 Phase1 LODO CUDA-fix r7 Runner Handoff

## 结论

- 实验ID：`d99_d100_phase1_lodo_d6efa5ad_cudafix_20260721_r7`
- 运行状态：`ANALYZED_NONFORMAL_LODO_DIAGNOSTIC`；进程正常退出，`runner.exit=0`。
- 结果状态：`NONFORMAL_LODO_DIAGNOSTIC`；`formal_phase1_lock=false`、`canonical_lock_artifact_write_allowed=false`。
- D99开发准入：K5、K10通过；K1、K20因最差类floor回退被拒。
- D100开发准入：K1/K5/K10/K20全部失败；所有K的64个候选均未让融合balanced NLL严格优于D99，K5/K10请求融合还存在逐receiver×pseudo-new局部退化。
- 最强可继续研发的分支：D99 K10，final outer-LODO相对D81的balanced accuracy为`87.32%→88.75%`、old为`87.32%→88.75%`、new为`87.32%→88.76%`、H为`86.45%→88.10%`、全局worst-class floor为`19.15%→36.26%`，balanced NLL为`0.9493→0.7090`。
- 关键缺陷：K5与K10虽aggregate floor上升，但D99各有`24/42`个receiver×pseudo-new pair的floor下降；当前aggregate准入门不足以保证局部反遗忘稳定性。
- 协议：只使用Phase1单次LEO观测档案和聚合ground知识；`target_rows_used=0`、`query_rows_used_for_selection=0`、`clean_or_raw_iq_used=false`。未访问target、未修改候选、未重试。

## 执行与版本证据

|项目|值|
|---|---|
|代码提交|`d6efa5ad`|
|预登记提交|`15c49902867b7c132bc0a70a37ff080baac39d54`|
|源码ZIP|`source_d6efa5ad.zip`，31,216,534B，4380成员|
|源码ZIP SHA-256|`701e124fbb53046c9361995f2a6141841853153eabaef2d6865ce3187bd5b82a`|
|配置SHA-256|`3241eb36d4f774f6e3751af7f7682060ce0a0e8204de18227870c133cebdb4e2`|
|包装器PID/Python PID|`1457448`/`1457450`，均已退出|
|GPU|物理GPU5；`CUDA_VISIBLE_DEVICES=5`；内部`cuda→cuda:0`smoke通过|
|CPU线程限制|OMP/MKL/OpenBLAS/NumExpr均为2；运行中Python约2.1核|
|开始/结束|`2026-07-21 04:11:49 CST`/`2026-07-21 04:23:10 CST`|
|时长|11分21秒|
|主结果|`output/d99_d100_phase1_lodo_blocked_diagnostic.json`，20,592,814B|
|主结果SHA-256|`6a7b6cb0ab9b0201fe99a7290067925ae7138490cd0b86e1255749a0eb7d46bf`|
|receipt SHA-256|`8af595bb3984a525472dd33232872c5b19e678ea4bbef74214a82a9c6ebff826`|

ZIP本地审计结果为`unsafe=0`、`symlink=0`、`duplicate=0`；远端整体SHA、4380成员数、5个模块SHA、`py_compile`、import、配置validator与CUDA正规化smoke全部通过。5个模块SHA为：

|模块|SHA-256|
|---|---|
|runner|`110295caa83ab0d7717e26b17b1d4ac33423337afaa8877067f64649d06c7ea1`|
|D100|`86c185ee13222bc0c97c4576984b9cd07f981201da4f0b62f8d4bc66970b4714`|
|D81 scorer|`b0a587d873e0d1db552dc12532c5f31544f570e28d5c6a9cfcca833ddbe7f257`|
|LODO|`aa99b3d726338481ed7f22f4acc5cdf2cfe4b2ef420e44da6f2ff2f674841e0e`|
|D99|`c166a5e375b0b8be5c95e678e63a6f04526474cd1a01544616829106af52f56f`|

## 每K准入与选参

|K|D99状态|D100状态|候选|有效参数摘要|首要拒因|
|---:|---|---|---:|---|---|
|1|`BLOCKED_D99_ADMISSION`|失败|34|`eta=.5,T99=.85,lambda=.08,TR=1.0,alpha=0`|D99全局worst floor下降2.37pp|
|5|`D99_ADMITTED`|失败|4|`eta=.25,T99=.85,lambda=.2,TR=.85,alpha=0`|D100 NLL不严格改善；selection中24/42 pair退化|
|10|`D99_ADMITTED`|失败|4|`eta=.25,T99=.85,lambda=.2,TR=.85,alpha=0`|D100 NLL不严格改善；selection中19/42 pair退化|
|20|`BLOCKED_D99_ADMISSION`|失败|34|`eta=.5,T99=.85,lambda=.08,TR=1.0,alpha=0`|D99全局worst floor下降23.95pp|

K5/K10冻结D99参数相同：`eta=.25`、`student_nu=3`、`kernel_volume_gamma=1`、`shared_h0=.35`、`scale_prior_strength=2`、`scale_min_ratio=.5`、`scale_max_ratio=2`、`d99_temperature=.85`、`lambda0=.2`、`ridge_temperature=.85`。D100请求`alpha=.2`，但未过门后统一强制为`alpha=0`；因此没有D100锁参，最终effective fused指标等于D99。

## D81→D99关键性能

K5/K10来自final outer-LODO；K1/K20没有入选候选，表中仅为selection阶段排名第一的blocked诊断候选，不能当作final outer-LODO结果。

|K|作用域|BA|balanced NLL|old|new|H|全局worst floor|判定|
|---:|---|---:|---:|---:|---:|---:|---:|---|
|1|blocked候选诊断|76.15%→78.94%|1.0673→0.7417|76.15%→78.94%|76.15%→78.94%|71.97%→75.31%|4.49%→2.13%|均值/NLL改善，但floor退化，拒绝|
|5|final outer-LODO|81.54%→88.21%|1.0341→0.7583|81.54%→88.21%|81.54%→88.23%|80.59%→87.63%|31.96%→38.46%|D99开发准入|
|10|final outer-LODO|87.32%→88.75%|0.9493→0.7090|87.32%→88.75%|87.32%→88.76%|86.45%→88.10%|19.15%→36.26%|D99开发准入；当前最强|
|20|blocked候选诊断|90.00%→85.82%|0.9041→0.5448|90.00%→85.82%|90.00%→85.82%|89.49%→84.42%|36.17%→12.22%|校准改善但识别/floor全面退化，拒绝|

K5相对D81的增量为：BA`+6.68pp`、old`+6.67pp`、new`+6.69pp`、H`+7.04pp`、全局floor`+6.50pp`、balanced NLL改善`0.2758`。K10增量为：BA`+1.43pp`、old`+1.43pp`、new`+1.44pp`、H`+1.65pp`、全局floor`+17.11pp`、balanced NLL改善`0.2402`。

## D100为什么没有收益

64个候选在每个K均满足aggregate双向rescue非零，但没有任何候选让请求融合的balanced NLL严格优于D99，因此D100通过数为`0/64`。此外：

- K5选中D99候选的selection诊断中，请求D100使BA`87.31%→87.05%`、NLL`0.7666→0.8342`、H`86.47%→86.28%`；虽floor`32.22%→37.78%`，但24/42 pair违反old/new/H/floor联合非降门。
- K10中，请求D100使BA`87.97%→87.87%`、NLL`0.7162→0.7903`、H`87.09%→87.02%`，floor保持`26.67%`；19/42 pair违反联合非降门。
- 在final outer-LODO的请求融合诊断中，K5和K10均有28/42 pair至少一个old/new/H/floor指标下降；effective alpha回退为0后才得到0个退化pair，但此时D100与D99完全相同。
- Ridge头确有互补纠错：K5 final中D99救回516个ridge错误、ridge救回393个D99错误，oracle union为89.95%；K10分别为620、306，oracle union为90.10%。问题不是没有互补性，而是固定凸融合无法把互补性转化为稳定的NLL和逐pair收益。

## Receiver×pseudo-new稳定性

下表每行是同一receiver下6个pseudo-new轮换fold的均值；floor列为6个fold中的最小floor。`floor回退`与`D100退化`为pair计数。

|K|receiver|mean rho|BA D81→D99|NLL D81→D99|old D81→D99|new D81→D99|H D81→D99|min floor D81→D99|floor回退|D100请求退化|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|5|1-1|.0722|68.69→88.19|1.119→.790|68.69→88.19|68.69→88.19|66.94→87.83|43.59→73.33|0/6|6/6|
|5|1-19|.0632|92.11→88.69|.895→.668|92.11→88.69|92.11→88.69|91.68→87.86|75.00→64.29|6/6|0/6|
|5|14-7|.0770|92.35→92.29|.829→.611|92.35→92.29|92.35→92.29|92.18→91.95|81.91→73.91|6/6|6/6|
|5|18-2|.0118|70.30→75.79|1.197→.987|70.30→75.79|70.30→75.79|69.26→73.97|45.74→38.46|6/6|4/6|
|5|19-2|.0768|94.15→90.01|.905→.674|94.15→90.01|94.15→90.01|94.07→89.67|85.42→77.08|6/6|6/6|
|5|2-1|.1300|66.28→91.65|1.326→.852|66.28→91.65|66.28→91.65|63.36→91.35|31.96→76.84|0/6|0/6|
|5|2-19|.1602|86.86→90.87|.968→.727|86.86→90.85|86.86→90.96|86.62→90.74|78.41→80.68|0/6|6/6|
|10|1-1|.0898|92.46→86.26|.870→.663|92.46→86.26|92.46→86.26|92.37→85.70|85.90→66.28|6/6|3/6|
|10|1-19|.0941|93.91→89.06|.887→.658|93.91→89.06|93.91→89.06|93.70→88.29|81.82→66.67|6/6|0/6|
|10|14-7|.0780|88.20→94.13|.753→.560|88.20→94.13|88.20→94.13|87.38→93.93|63.83→80.43|0/6|6/6|
|10|18-2|.0138|63.86→76.23|1.154→.946|63.86→76.23|63.86→76.23|59.60→74.08|19.15→36.26|0/6|3/6|
|10|19-2|.1292|92.78→91.10|.922→.676|92.78→91.10|92.78→91.10|92.54→90.87|77.08→80.21|0/6|6/6|
|10|2-1|.1538|91.81→92.50|1.038→.721|91.81→92.49|91.81→92.53|91.60→92.28|82.11→81.05|6/6|4/6|
|10|2-19|.1978|88.25→92.00|1.021→.740|88.25→92.00|88.25→92.00|87.95→91.58|75.64→70.45|6/6|6/6|

主要局部失败：

- K5最差new变化发生在receiver`18-2`、pseudo-new`20-19`：`63.74%→38.46%`，下降`25.27pp`；该pair的H下降`14.83pp`。
- K10最差new变化发生在receiver`1-1`、pseudo-new`20-19`：`88.37%→66.28%`，下降`22.09pp`；H下降`14.33pp`。
- K10按pseudo-new类跨receiver平均时，`20-19`的new为`80.05%→68.77%`，是最明确的注册塑性退化类；`8-20`也由`98.54%→96.35%`小幅下降。
- K5/K10的pair级rho与BA/floor增量相关性方向不稳定：K5约为`+.22/+.34`，K10约为`-.30/-.33`。因此当前coverage certificate不能单独预测正迁移，需要和D99 margin/floor风险共同建门。

## 类别轮换汇总

每行跨7个held receiver平均，列为new、old、H的D81→D99。

|K|pseudo-new|new|old|H|
|---:|---|---:|---:|---:|
|5|14-10|68.87→87.86|84.07→88.27|74.22→87.95|
|5|14-7|77.77→84.89|82.29→88.86|79.82→86.58|
|5|20-15|92.17→94.24|79.41→87.02|84.60→90.27|
|5|20-19|69.45→70.38|83.95→91.80|75.60→79.00|
|5|6-15|87.77→95.66|80.29→86.74|83.80→90.89|
|5|8-20|93.17→96.33|79.21→86.57|85.48→91.07|
|10|14-10|76.03→88.81|89.58→88.75|80.31→88.63|
|10|14-7|78.71→85.90|89.05→89.33|83.32→87.40|
|10|20-15|95.73→96.48|85.64→87.21|90.20→91.40|
|10|20-19|80.05→68.77|88.78→92.76|83.90→78.27|
|10|6-15|94.87→96.25|85.81→87.23|90.04→91.47|
|10|8-20|98.54→96.35|85.08→87.24|90.92→91.44|

## 资源与计算

|项目|值|解释|
|---|---:|---|
|候选episode评估|21,504|64候选、多K、多fold开发诊断|
|D99/D100解析状态构建|10,752|0 epoch、0 optimizer step|
|D81 episode fit|28次×20步=560步|D81固定全局ground basis不按pseudo-new fold重复拟合|
|最大选中D99+D100已知持久wire|33,070B|仅已知组件，非完整系统上界|
|最大query MAC上界|207,754/sample|仅D99+D100已知组件|
|最大可训练参数等价|1,734|D99+D100解析头|
|最大D99 fit瞬态上界|1,243,520B|D100和D81完整fit peak未知|
|固定D81 ground basis数值字节|18,032B|聚合Phase1知识|
|完整D99 ground bundle数值字节|6,930B|聚合Phase1知识|

`complete_combined_fit_peak_available=false`、`complete_combined_parameter_count_available=false`、`complete_combined_persistent_upper_bound_available=false`、`complete_combined_query_mac_available=false`，所以即使已知部分低于预算，也不能声明正式满足256KiB/80k参数门；资源状态为`NONFORMAL_PARTIAL_KNOWN_COMPONENTS_ONLY`。

## 协议和正式性边界

- 通过项：`phase1_only=true`、`single_leo_observation_archive=true`、`clean_or_raw_iq_used=false`、`target_rows_used=0`、`query_rows_used_for_selection=0`、`class_specific_hyperparameters=false`、所有类轮换为pseudo-new、K20是独立真实episode而非复制K10。
- D99局部ground知识按outer fold删除held receiver行，并删除pseudo-new类行；没有使用pseudo-new ground行。
- 局限：D81固定全局ground basis可能包含held receiver域，且不按outer fold重训，因此该实验只能声称“support adaptation和D99局部ground消融的pseudo-target receiver评估”，不能声称完整encoder/D81 basis的whole-method LODO。
- ground authority为`BLOCKED_DEVELOPMENT_GROUND_RELEASE`，`formal_phase1_eligible=false`；blocked inputs还包括`formal_feature_archive`和`independent_ground_authority_root`。
- 因此K5/K10的`D99_ADMITTED`仅是本次开发诊断内部准入，不是正式Phase1 lock，更不能直接替代target 125实验。

## 后续研发建议

1. 保留D99 K5/K10主干，但把准入从“aggregate floor非降”升级为receiver×pseudo-new分层约束，例如限制floor回退pair比例、下分位delta或receiver级max-regret；否则24/42局部floor回退会被均值掩盖。
2. 不继续固定alpha凸融合D100。Ridge的oracle rescue充足，但全候选NLL均变差；下一版应先做support-only可靠度/风险收缩或只在预注册pair-risk proxy安全时启用，而不是把互补性直接等同于可融合性。
3. 针对K10 pseudo-new`20-19`设计类对称、无ID Oracle的support几何风险特征，例如support内类间margin、D99/ridge cross-fit分歧和ground coverage联合门；不能按真实类别ID特判。
4. 当前rho对收益的相关方向随K翻转，不能单独控制ground transport强度。需要用Phase1 LODO重新拟合`rho+margin+floor-risk`共享门，并保持所有类、receiver使用同一公式。
5. 在正式125前仍需解决formal feature archive、独立ground authority和完整资源上界；本结果只支持下一轮Phase1开发，不支持正式性能发布。

## 回收产物

- `remote_output/d99_d100_phase1_lodo_blocked_diagnostic.json`：完整20.6MB原始结果。
- `remote_output/result.json`：runner结果receipt。
- `remote_log/`：完整命令、起止时间、PID、退出码和stdout。
- `remote_config.json`：远端配置副本。
- `aggregate_summary_4k.csv`：4个K的同rowaggregate表。
- `pair_metrics_84.csv`：K5/K10全部84个receiver×pseudo-new同row指标。
- `receiver_summary_14.csv`：2个K×7个receiver汇总。
- `pseudo_new_class_summary_12.csv`：2个K×6个pseudo-new类汇总。

最终远端状态：runner/Python均退出，GPU5为`0%/10MiB`。最终本地SSH状态：`ssh.exe=0`，至N607和lab bridge的`ESTABLISHED TCP22=0`。

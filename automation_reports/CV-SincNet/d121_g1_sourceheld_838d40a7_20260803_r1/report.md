# D121-RDCE＋LBR-qKNN固定四臂source-held G1发布报告

状态：`ARTIFACTS_COMPLETE / ANALYZED / REJECT_D121_REVISION_PERFORMANCE_WEAK`

## 1.实验身份与目标

- 实验ID：`d121_g1_sourceheld_838d40a7_20260803_r1`
- 时间：2026-08-03
- operator：主agent负责方法集成、数据分析与晋级判定；唯一Terra Max runner负责N607落地、预测封存、独立score和artifact回收
- 目标：在固定D104 source-held 63行矩阵上比较`M0/M_DA/M_HEAD/M_JOINT`，首次验证LBR性能；不进入Target，不扩展125矩阵。
- 假设：LBR在identity和冻结RDCE下都能提高K1总正确数且不降低汇总old/new净正确数与全矩阵floor。
- 历史因果依据：D106 RDCE域适应平均主效应为正，但旧RCMR联合floor转负；D121保留RDCE并只替换head。

## 2.固定四臂与晋级规则

|臂|域适应|head|
|---|---|---|
|`M0`|identity|原Student-t qKNN|
|`M_DA`|冻结D106 RDCE|原Student-t qKNN|
|`M_HEAD`|identity|D121 LBR-qKNN|
|`M_JOINT`|冻结D106 RDCE|在RDCE后重建LBR图|

固定矩阵：7个held receiver×K1/K5/K10的21行，加7个receiver×6个held class×K1的42行，共63行、252个arm-row prediction units。

晋级只比较：

- `HEAD_AT_ID=M_HEAD-M0`
- `HEAD_AT_DA=M_JOINT-M_DA`

对两个head效应分别要求：

1.完整同row集合的old correct净和不小于0；
2.`held_class!=None`行的seen-new correct净和不小于0；
3.完整矩阵old floor和all-class floor的全局最小值不降低；
4.K1固定row的total correct净和严格大于0。

这是矩阵汇总gate，不要求每一row指标非负。任一条件不满足即`REJECT_D121_REVISION_PERFORMANCE_WEAK`，不调参、不重跑、不扩矩阵。

## 3.协议与数据复用

- `protocol_schema=p2_min_v1`；predict无truth参数，query零fit/update/selection，无Target、role、quota或global assignment。
- 完整63行prediction manifest及每row SHA/receipt先封存，score才允许打开D104独立truth。
- 不重新prepare或验证数据，直接只读复用D106-r2已完成的21个D104 predictor packages及truth seal。
- 远端package root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_g1_sourceheld_b442472b_20260801_r2/packages`
- package manifest SHA256：`55780d103e2cb19b4446ef44033446182beac992b42c59753b6910b0910e6710`
- truth input seal SHA256：`c2818c2876e3e76e813137b8eb0acc681e2ce5a46e40e46311207500c4b143fc`
- truth SHA256：`09a719aae8f8196c93d8191c1e4c038ec84fbe9009d493a3aa83ac2360ad7d62`
- RDCE wire：`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/rdce_asset/d106_rdce_gtsm.asset.wire`
- RDCE wire SHA256：`20e44cb0eb2f5698e6d5f9029b63cf296ffbf4716edb999fed6743c8671bd795`
- 参数扫描数：0。

## 4.本地版本与验证

根目录`E:\type10-7`不是Git仓库。本报告同步镜像到Git工作树：
`E:\type10-7\code\snapshots\cdom_scxmap_d92_glf_r1_wt\automation_reports\CV-SincNet\d121_g1_sourceheld_838d40a7_20260803_r1\report.md`。

- 分支：`codex/stage2-da25-r1`
- G0 pass报告commit：`c4c3c69e`
- G1 gate澄清commit：`9d1ef7a5`
- G1实现commit：`838d40a743d4dc5401dc2095475a75107bcc084b`
- 禁止push。

|文件|用途|SHA256|
|---|---|---|
|`code/scripts/run_d121_g1_sourceheld_one_shot.py`|四臂predict与独立score|`bf61c1e5f78dc3900d89f940d98099851f4985615b7d633196a72b00dbffdb38`|
|`tests/test_run_d121_g1_sourceheld_one_shot.py`|63行、四臂、truth seal与gate测试|`a0288759f011b58ff55a844d3bf4a9181f7b2e45188e42b9a56707d8818a4a1f`|
|`d121_g1_source_838d40a7.zip`|实现commit的`code/` archive|`2c763986641fc94d51bd6dd035ee1bec87c495dcaec60fff9c94b144b03e6ffe`|

source archive本地路径：
`E:\type10-7\automation_reports\CV-SincNet\d121_g1_sourceheld_838d40a7_20260803_r1\input\d121_g1_source_838d40a7.zip`，大小7,985,344bytes。

验证：

- 实现agent：新聚焦测试3/3通过。
- 主agent：D121＋继承D106/D112生命周期测试12/12通过。
- 独立Terra Max审查：`MERGE / P0=0 / P1=0 / P2=0`，其聚焦及继承验证14 passed。
- 真实21 packages＋真实RDCE wire无truth predict smoke：63行、四臂、252 units，`query_truth_access=false`、`query_state_updates=0`、`target_access=false`。
- 本地smoke prediction manifest SHA256：`981f1db3ef36155899dc8c826254f73dbbfe01920361c60c048dc2064f007228`。该smoke未打开truth，不是性能结果。

## 5.N607不可覆盖预登记

远端run root：
`/home/szu2070436088/2510044040/CV-SincNet/runs/d121_g1_sourceheld_838d40a7_20260803_r1`

- source：`<run-root>/source`
- CWD：`<run-root>/source`
- logs：`<run-root>/logs/runner.log`
- PID：`<run-root>/logs/launch.pid`
- exit：`<run-root>/logs/runner.exit`
- predictions：`<run-root>/predictions`，启动前必须ABSENT
- scores：`<run-root>/scores`，启动前必须ABSENT
- GPU：`CUDA_VISIBLE_DEVICES=0`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`

固定predict command：

```text
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_d121_g1_sourceheld_one_shot.py predict --package-root /home/szu2070436088/2510044040/CV-SincNet/runs/d106_g1_sourceheld_b442472b_20260801_r2/packages --rdce-asset-wire /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/rdce_asset/d106_rdce_gtsm.asset.wire --rdce-wire-sha256 20e44cb0eb2f5698e6d5f9029b63cf296ffbf4716edb999fed6743c8671bd795 --run-id d121_g1_sourceheld_838d40a7_20260803_r1 --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/d121_g1_sourceheld_838d40a7_20260803_r1/predictions
```

predict完整exit0且63行/252 units封存后，固定score command：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_d121_g1_sourceheld_one_shot.py score --prediction-root /home/szu2070436088/2510044040/CV-SincNet/runs/d121_g1_sourceheld_838d40a7_20260803_r1/predictions --truth-json /home/szu2070436088/2510044040/CV-SincNet/runs/d106_g1_sourceheld_b442472b_20260801_r2/packages/scorer_only/truth.json --truth-input-seal-json /home/szu2070436088/2510044040/CV-SincNet/runs/d106_g1_sourceheld_b442472b_20260801_r2/packages/scorer_only/truth_input_seal.json --truth-open-event-json /home/szu2070436088/2510044040/CV-SincNet/runs/d121_g1_sourceheld_838d40a7_20260803_r1/scores/truth_open_event.json --output-json /home/szu2070436088/2510044040/CV-SincNet/runs/d121_g1_sourceheld_838d40a7_20260803_r1/scores/held_scores.json
```

## 6.runner与技术stop

唯一runner必须：

1.直连N607只读preflight，核对普通账户、GPU/进程、r1 root ABSENT、复用package/wire/truth-seal的文件/hash；
2.同步固定source zip，远端核对zip和G1脚本SHA并`py_compile`；
3.不可覆盖detached启动predict，核PID/CWD/cmdline/log；predict完成后先验证63行/252 units及无truth状态；
4.仅在prediction seal完整后启动score；score结束后回收package manifest引用、prediction manifest、63 row JSON、truth-open event、held scores、log/PID/exit/hash清单；
5.每个SSH/SCP短连接后验证`ssh.exe=0`及N607/lab TCP22清零。

仅技术错误可停止：wrong hash/path、output已存在、query leakage、prediction coverage不完整、非零exit、确定性异常。不得根据accuracy、effect或promotion gate中途停止，也不得为结果不好而重跑。

fresh-run retry未授权；技术失败保留partial并记`NO_PERFORMANCE_RESULT`。

## 7.完成后分析

主agent只用完整`held_scores.json`做same-row数据分析，至少报告：

- 两个head效应的old/new净correct；
- K1 total correct净和；
- old/all floor全局最小值及差值；
- 63行按receiver、held class、K的正负分布；
- 四臂的old BA、seen-new、H、floor和correct count；
- `promotion_allowed`与每个冻结criteria。

性能弱则立即关闭D121并研发下一原理方法；通过才允许进入后续Target25，不运行125矩阵。

## 8.N607执行闭合

- predict：wrapper PID58978、child PID58980；CWD、cmdline、`CUDA_VISIBLE_DEVICES=0`与冻结命令匹配，exit0。
- prediction seal：63个row JSON、252个arm-row units；manifest SHA256=`71ef06e7ecf2bdbb7598e5bbd630860474ed9753e0903c8c95821835493f55c5`；prediction set receipt=`d500b7672b2ec5347a4c155c82c9cc74191377c6be3c4d404ec5fd97e4c314d8`。
- 本地用`ssr-gpu`绝对Python复算manifest、逐row文件SHA与receipt全部通过；`query_truth_access=false`、`query_state_updates=0`、`query_selection_count=0`、`target_access=false`。
- score只在prediction seal闭合后启动：PID62064、child62066；CWD/cmdline匹配，exit0，log异常指纹0。
- `truth_opened_after_all_predictions_committed=true`；held scores SHA256=`b6df00cc10aab5120ac2858b61617eb2ca87b63c15b711dd6fe5d9915b39c607`；score set receipt=`1639dd31fc2bd6f6bdc0ed3e13c40934102ab6ba6197d54811821f893f8d605c`。
- 最终predict/score PID均退出；8张GPU均0%利用率、约1MiB显存占用；`ssh.exe=0`，N607/lab TCP22连接=0；无重跑。

## 9.四臂完整row均值

下表是63行的row均值，seen-new与H只对42个`held_class!=None`行取均值；不是从不同run拼接的单项极值。

|臂|old BA|seen-new|H|old floor|all floor|平均correct/row|
|---|---:|---:|---:|---:|---:|---:|
|`M0`|83.6560%|83.7772%|82.2378%|57.9803%|56.4199%|288.9683|
|`M_DA`|83.9163%|84.1404%|82.6826%|58.2627%|56.8637%|289.8889|
|`M_HEAD`|83.5597%|83.7369%|82.1950%|57.7745%|56.2410%|288.6508|
|`M_JOINT`|83.8169%|84.0194%|82.5219%|57.7496%|56.2699%|289.5397|

## 10.同row因果效应

|效应|Δold BA|Δseen-new|ΔH|Δold floor|Δall floor|总correct净和|
|---|---:|---:|---:|---:|---:|---:|
|`DA_AT_BASE=M_DA-M0`|+0.2604pp|+0.3632pp|+0.4447pp|+0.2824pp|+0.4438pp|+58|
|`HEAD_AT_ID=M_HEAD-M0`|−0.0963pp|−0.0404pp|−0.0428pp|−0.2058pp|−0.1789pp|−20|
|`HEAD_AT_DA=M_JOINT-M_DA`|−0.0995pp|−0.1211pp|−0.1606pp|−0.5131pp|−0.5938pp|−22|
|`JOINT_VS_M0=M_JOINT-M0`|+0.1609pp|+0.2421pp|+0.2841pp|−0.2307pp|−0.1500pp|+36|

解释：`M_JOINT`的BA、seen-new和H仍高于`M0`，完全来自RDCE正主效应覆盖了LBR负效应；不能据此把LBR判为正收益。两个匹配head效应都为负，且RDCE下floor损失更大。

## 11.冻结gate

|head效应|old correct净和|seen-new correct净和|K1 total correct净和|old floor全局min差|all floor全局min差|晋级|
|---|---:|---:|---:|---:|---:|---|
|`HEAD_AT_ID`|−19|−1|−7|0.0000pp|0.0000pp|否|
|`HEAD_AT_DA`|−19|−3|−21|0.0000pp|0.0000pp|否|

两个全局min floor差为0只因为baseline与candidate都已有0 floor，属于饱和而非保护成功；row均值和负row分布均显示floor实际变差。两个head效应的old/new净正确数与K1严格正条件全部失败，因此：

- `promotion_allowed=false`
- `promotion_decision=REJECT_D121_REVISION_PERFORMANCE_WEAK`

## 12.负效应覆盖范围

|效应/指标|正row|零row|负row|净和|
|---|---:|---:|---:|---:|
|`HEAD_AT_ID` total correct|2|48|13|−20|
|`HEAD_AT_ID` old correct|2|49|12|−19|
|`HEAD_AT_ID` seen-new correct|0|41|1|−1|
|`HEAD_AT_ID` old floor|2|56|5|−0.1297|
|`HEAD_AT_ID` all floor|2|57|4|−0.1127|
|`HEAD_AT_DA` total correct|10|28|25|−22|
|`HEAD_AT_DA` old correct|10|31|22|−19|
|`HEAD_AT_DA` seen-new correct|2|37|3|−3|
|`HEAD_AT_DA` old floor|4|43|16|−0.3233|
|`HEAD_AT_DA` all floor|3|42|18|−0.3741|

按K的total correct净和：

|效应|K1|K5|K10|
|---|---:|---:|---:|
|`HEAD_AT_ID`|−7|−8|−5|
|`HEAD_AT_DA`|−21|0|−1|

LBR没有只在某个K偶发失败：identity下三个K全负；RDCE下K1大幅负、K5为0、K10仍负。

## 13.receiver与held-class诊断

|held receiver|`HEAD_AT_ID` total correct净和|`HEAD_AT_DA` total correct净和|
|---|---:|---:|
|`1-1`|+2|−1|
|`1-19`|−1|+8|
|`14-7`|+1|−11|
|`18-2`|−11|−9|
|`19-2`|−8|−8|
|`2-1`|−2|0|
|`2-19`|−1|−1|

主要损失集中在receiver`18-2`和`19-2`；RDCE只在`1-19`使LBR局部转正，不能抵消其他receiver的广泛负效应。按held class汇总，identity下六个held class的total correct均为−1；RDCE下六个held class均为−3，说明问题不是单一类ID特例。

## 14.artifact与最终裁决

|artifact|SHA256|
|---|---|
|`predictions/prediction_manifest.json`|`71ef06e7ecf2bdbb7598e5bbd630860474ed9753e0903c8c95821835493f55c5`|
|`scores/truth_open_event.json`|`9dd42ef0d9f68de47d19a6696660ccf3f8e13e2d92b73ff646517c6b7044ddc6`|
|`scores/held_scores.json`|`b6df00cc10aab5120ac2858b61617eb2ca87b63c15b711dd6fe5d9915b39c607`|
|`logs/runner.log`|`5ce774e2887e41e5528f6d73abaf000bb1c5ee146510d89905a8dfc050413adc`|
|`logs/score.log`|`46c43f9ef3402c71d092d7fb8fd087424e8cdafb0117137cd4323c3f4a0b5bbe`|
|`hash_manifest_and_cleanup.txt`|`960fd6c7a6ce67c41c9ba3db7c59d08a23fd700ea152feeeb165d591e9b4da20`|

最终裁决：永久关闭D121-LBR当前revision，不进入Target25，不修改rival数、强度、阈值或温度，不运行125矩阵。保留D106 RDCE作为已再次复现的正域适应因素；下一轮从新的分类原理出发，而不是调LBR参数。

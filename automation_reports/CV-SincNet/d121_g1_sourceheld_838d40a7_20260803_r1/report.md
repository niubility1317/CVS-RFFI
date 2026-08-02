# D121-RDCE＋LBR-qKNN固定四臂source-held G1发布报告

状态：`LOCAL_VERIFIED / PREREGISTERED / NOT_YET_LANDED / NO_PERFORMANCE_RESULT`

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

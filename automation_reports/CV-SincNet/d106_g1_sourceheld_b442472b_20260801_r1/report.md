# D106 G1 source-held四臂实验报告

## 1.实验标识与目标

- 实验ID：`d106_g1_sourceheld_b442472b_20260801_r1`
- 预登记时间：2026-08-01
- 操作方：主agent负责方法冻结与结果分析；唯一N607 runner负责落地、启动、监测与artifact回收。
- 状态：`LOCAL_VERIFIED / PREREGISTERED / NOT_LAUNCHED`
- 目标：在冻结D104 source-held split上一次性比较`M0`、`M_DA`、`M_HEAD`、`M_JOINT`，形成63个matched rows、252组预测的首份D106 held性能证据。
- 边界：本实验不是Target25、不是P2正式性能，也不替代后续125矩阵；禁止从中挑选有利receiver、K、held class或单项极值。

## 2.假设与比较

|臂|DA输入|分类头|同row比较作用|
|---|---|---|---|
|`M0`|identity|Student-t qKNN|共同基线|
|`M_DA`|RDCE|Student-t qKNN|估计DA主效应|
|`M_HEAD`|identity|RCMR-2V qKNN|估计HEAD主效应|
|`M_JOINT`|RDCE|RCMR-2V qKNN|估计联合效果与交互|

主要假设：`M_JOINT`相对`M0`在同row的`H_old_new`和`old_floor`方向为正，同时用`DA_AT_BASE`、`DA_AT_HEAD`、`HEAD_AT_ID`和`HEAD_AT_DA`分解增益来源。所有63行完成后统一分析；任何中途准确率不得触发停机或选臂。

## 3.冻结矩阵与数据

- `split_id=d104_source_seed104713_v2`
- receiver数：7；注册类数：6；K=`1,5,10`。
- 一般行：7 receiver×3 K=21行。
- held-class行：7 receiver×6 held class×K1=42行。
- 总计：63 matched rows×4 arms=252预测单元。
- source-val scorer archive：`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/input/d104_split/scorer_only/source_val/features.npz`，SHA256=`20e7f6c8789d55d7db56f80ea885131bfff9eeccafa5c11087fc9d3d87f7b408`，大小6,191,766bytes。
- scorer manifest：同目录`manifest.json`，SHA256=`6d57b511e95382b0c1ebfad91b2a0de4be5c08d5896e0625aa3a4dfa110b5134`。
- RDCE wire：`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/rdce_asset/d106_rdce_gtsm.asset.wire`，SHA256=`20e44cb0eb2f5698e6d5f9029b63cf296ffbf4716edb999fed6743c8671bd795`。

## 4.本地版本与验证

- Git仓库：`E:\type10-7\code\snapshots\cdom_scxmap_d92_glf_r1_wt`
- 分支：`codex/stage2-da25-r1`
- 实现提交：`b442472b4c3d12cc0fd576774d485b0816b082d0`
- 新增文件：`code/scripts/run_d106_g1_sourceheld_one_shot.py`、`tests/test_run_d106_g1_sourceheld_one_shot.py`。
- CLI SHA256：`500d22b473b6b803706471c1aad7798bf414486ca4f103a906d72ed6fd4dbc90`。
- 测试SHA256：`ce89c32cb85605d4ec1a5264ad9c946d7148d7c9a8535c731486600aa697ef28`。
- 验证：`ssr-gpu`环境下`py_compile`通过；G1、四臂、RCMR、RDCE asset/runtime相关回归90项全部通过；`git diff --check`通过。
- 独立复核：`P0=0、P1=0、RELEASE`；P2不阻塞。
- release zip：`E:\type10-7\automation_reports\CV-SincNet\d106_g1_sourceheld_b442472b_20260801_r1\release\d106_g1_b442472b.zip`，SHA256=`01a63e43abcf893b24546f74fb37f861f0bdb6eba3a2a3944f823e2f64e908a1`。
- 根目录`E:\type10-7`不是Git仓库；本报告同步镜像到上述Git工作树的同名`automation_reports`目录并单独提交。

## 5.N607发布合同

- 远端项目根：`/home/szu2070436088/2510044040/CV-SincNet`
- 不可覆盖run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_g1_sourceheld_b442472b_20260801_r1`
- Python环境：`ssr-gpu`对应的N607 Python；工作目录为run root内解压的`release`。
- 计算分配：本实验消费已冻结特征并使用NumPy推断，无训练GPU需求；不占用GPU训练配额。
- 日志：`logs/run.out`；PID：启动后回填；退出码：`logs/run.exit`。
- 预期artifact：`packages/package_manifest.json`、21个predictor NPZ、`packages/scorer_only/truth.json`、`truth_input_seal.json`、`predictions/prediction_manifest.json`、63个row JSON、`scores/truth_open_event.json`、`scores/held_scores.json`。

冻结子命令依次为：

```text
python code/scripts/run_d106_g1_sourceheld_one_shot.py prepare --source-val-archive <r7-features.npz> --source-val-manifest <r7-scorer-manifest.json> --output-dir <run-root>/packages
python code/scripts/run_d106_g1_sourceheld_one_shot.py predict --package-root <run-root>/packages --rdce-asset-wire <r7-rdce-wire> --rdce-wire-sha256 20e44cb0eb2f5698e6d5f9029b63cf296ffbf4716edb999fed6743c8671bd795 --rcmr-method-lock configs/d106_rcmr_2v_method_lock_20260801.json --rcmr-method-lock-sha256 be452cc52da8e5c43d3addc73568580d63a83f146310ec3559bb5daa99076b0c --output-dir <run-root>/predictions
python code/scripts/run_d106_g1_sourceheld_one_shot.py score --prediction-root <run-root>/predictions --truth-json <run-root>/packages/scorer_only/truth.json --truth-input-seal-json <run-root>/packages/scorer_only/truth_input_seal.json --truth-open-event-json <run-root>/scores/truth_open_event.json --output-json <run-root>/scores/held_scores.json
```

## 6.健康检查、停止规则与成功条件

- 启动后核验唯一主PID、CWD、cmdline、run-root绑定和日志增长；每个阶段完成后核验输出数量与不可覆盖状态。
- `predict`必须在`score`前完整结束，并形成63行、252臂、完整receipt；score进程随后才允许开启truth。
- 仅P0协议/安全错误、不可覆盖风险、错误checkout/SHA、或至少两个不同阶段/row出现相同确定性零预测异常指纹时停止精确run-owned进程树。
- 禁止因准确率、H、floor或任何中间性能值停止、重启或修改方法。
- 技术成功：prepare、predict、score均退出0；63/252闭包、21包、truth-open时序和所有SHA/receipt核验通过。
- 性能分析：完整报告每个matched row的四臂`old_balanced_accuracy`、`seen_new_accuracy`、`H_old_new`、`old_floor`、`balanced_accuracy`、`correct_count/query_count`及五组same-row effects；不使用边际最大值代替联合行。

## 7.风险与完成后检查

- source-held是非Target诊断；即便方向为正，也只能支持进入Target25或方法修订，不能直接宣称达到项目目标。
- K1 held-class行用于old/new分解；一般K1/K5/K10行用于整体receiver/K稳定性，两类行不得混为一个单指标排名。
- 完成后回收完整score、prediction manifest、truth-open event、run log、PID/exit和SHA清单；主agent再与D62、D91、D92、SVRN已有同边界证据分层对比。

## 8.结果表（待完成后回填）

|row|receiver|held class|K|arm|old BA|seen-new|H|old floor|all floor|correct/query|判定|
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
|待运行||||||||||||`NOT_RUN`|

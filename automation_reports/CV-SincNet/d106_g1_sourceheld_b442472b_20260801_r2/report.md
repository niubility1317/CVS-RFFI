# D106 G1 source-held四臂实验报告（r2）

## 1.标识、状态与目标

- 实验ID：`d106_g1_sourceheld_b442472b_20260801_r2`
- 时间：2026-08-01；主agent负责方法与分析，唯一Terra Max N607 runner负责发布。
- 状态：`LOCAL_VERIFIED / PREREGISTERED / NOT_LAUNCHED`
- 目标：在冻结D104 source-held split上完整运行`M0/M_DA/M_HEAD/M_JOINT`，获得63个matched rows、252组预测的D106 G1 held性能证据。
- r1处置：`d106_g1_sourceheld_b442472b_20260801_r1`仅完成landing，Git archive将受Git文本归一化的method-lock写成LF字节，远端SHA=`0a174521…`，不等于外部冻结SHA=`be452cc5…`；未启动、未生成package/prediction/score、未开启truth，永久标记`LANDED_NOT_STARTED_WRONG_LOCK_HASH_SEMANTIC / NO_PERFORMANCE_RESULT`。
- r2唯一修复：除原Git archive外，额外从本地已验证工作树精确SCP冻结method-lock字节到新run的release路径；不改方法、矩阵、输入或数值。

## 2.假设、四臂与完整矩阵

|臂|表示|分类头|作用|
|---|---|---|---|
|`M0`|identity|Student-t qKNN|基线|
|`M_DA`|RDCE|Student-t qKNN|DA主效应|
|`M_HEAD`|identity|RCMR-2V qKNN|HEAD主效应|
|`M_JOINT`|RDCE|RCMR-2V qKNN|联合效果|

- split：`d104_source_seed104713_v2`；7 receivers；6 classes；K=`1,5,10`。
- 21个receiver×K一般行，加42个receiver×held-class×K1行，共63行×4臂=252预测单元。
- 假设：`M_JOINT`相对`M0`的同row`H_old_new`和`old_floor`方向为正；同时完整报告`DA_AT_BASE/DA_AT_HEAD/HEAD_AT_ID/HEAD_AT_DA/JOINT_VS_M0`。
- 禁止按中间性能停机、选row、选receiver、选K或选单项极值。

## 3.冻结输入与版本

- source-val archive：`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/input/d104_split/scorer_only/source_val/features.npz`，SHA=`20e7f6c8789d55d7db56f80ea885131bfff9eeccafa5c11087fc9d3d87f7b408`。
- scorer manifest：同目录`manifest.json`，SHA=`6d57b511e95382b0c1ebfad91b2a0de4be5c08d5896e0625aa3a4dfa110b5134`。
- RDCE wire：`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/rdce_asset/d106_rdce_gtsm.asset.wire`，SHA=`20e44cb0eb2f5698e6d5f9029b63cf296ffbf4716edb999fed6743c8671bd795`。
- Git仓库：`E:\type10-7\code\snapshots\cdom_scxmap_d92_glf_r1_wt`；分支`codex/stage2-da25-r1`。
- 实现commit：`b442472b4c3d12cc0fd576774d485b0816b082d0`；r1预登记commit：`a81ac1dea8974545190d8c3d6533294e10d26914`。
- release zip：r1报告目录下`release/d106_g1_b442472b.zip`，SHA=`01a63e43abcf893b24546f74fb37f861f0bdb6eba3a2a3944f823e2f64e908a1`。
- 精确覆盖文件：`E:\type10-7\code\snapshots\cdom_scxmap_d92_glf_r1_wt\configs\d106_rcmr_2v_method_lock_20260801.json`，SHA=`be452cc52da8e5c43d3addc73568580d63a83f146310ec3559bb5daa99076b0c`。
- CLI SHA=`500d22b473b6b803706471c1aad7798bf414486ca4f103a906d72ed6fd4dbc90`；相关回归90项通过；独立复核`P0=0、P1=0、RELEASE`。
- 根目录`E:\type10-7`非Git；本报告镜像到Git工作树同名目录并提交。

## 4.N607发布与命令

- 远端run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_g1_sourceheld_b442472b_20260801_r2`，创建前必须`ABSENT`。
- 环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD=`<run-root>/release`；本实验为冻结特征NumPy推断，不占训练GPU。
- 同步：zip→`<run-root>/input/d106_g1_b442472b.zip`；本地method-lock→`<run-root>/release/configs/d106_rcmr_2v_method_lock_20260801.json`，覆盖只发生在启动前的新run release副本，并在启动前验证SHA。
- 日志/PID/退出：`logs/run.out`、`logs/run.pid`、`logs/run.exit`。
- 顺序命令：

```text
python code/scripts/run_d106_g1_sourceheld_one_shot.py prepare --source-val-archive <features.npz> --source-val-manifest <manifest.json> --output-dir <run-root>/packages
python code/scripts/run_d106_g1_sourceheld_one_shot.py predict --package-root <run-root>/packages --rdce-asset-wire <rdce-wire> --rdce-wire-sha256 20e44cb0eb2f5698e6d5f9029b63cf296ffbf4716edb999fed6743c8671bd795 --rcmr-method-lock configs/d106_rcmr_2v_method_lock_20260801.json --rcmr-method-lock-sha256 be452cc52da8e5c43d3addc73568580d63a83f146310ec3559bb5daa99076b0c --output-dir <run-root>/predictions
python code/scripts/run_d106_g1_sourceheld_one_shot.py score --prediction-root <run-root>/predictions --truth-json <run-root>/packages/scorer_only/truth.json --truth-input-seal-json <run-root>/packages/scorer_only/truth_input_seal.json --truth-open-event-json <run-root>/scores/truth_open_event.json --output-json <run-root>/scores/held_scores.json
```

## 5.健康、停止与完成条件

- 启动后核唯一PID、CWD、cmdline、日志；prepare、predict封存、score分别用短连接检查。
- predict必须先完整形成63行/252臂；score才可开启truth。
- 仅P0、错误SHA、覆盖风险或至少两个不同row相同确定性零预测异常指纹可停止精确run-owned树；性能差不能停。
- fresh-run retry不授权；技术失败保全并标记`NO_PERFORMANCE_RESULT`。
- 成功需三阶段退出0、21 packages、63 row、252 arm、truth-open时序、SHA/receipt全闭合。
- 预期artifact：package manifest、21 NPZ、prediction manifest、63 row JSON、truth-open event、held scores、完整log/PID/exit/SHA清单。

## 6.结果表（待回填）

|row|receiver|held class|K|arm|old BA|seen-new|H|old floor|all floor|correct/query|判定|
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
|待运行||||||||||||`NOT_RUN`|

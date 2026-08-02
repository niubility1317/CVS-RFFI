# D112-SEAM-qKNN source-held G1报告

状态：`PREREGISTERED / LOCAL_VERIFIED / NOT_RUN / NO_PERFORMANCE_RESULT`

## 身份与目标

|字段|值|
|---|---|
|run ID|`d112_g1_sourceheld_seam_20260802_r1`|
|日期|2026-08-02|
|operator|主agent：协议、集成、数据和结果分析；Terra Max：G1 surface与ground-head核心|
|目标|在已封存且此前独立于D112开发的新source-held split上，一次性判断D112联合方法是否相对M0产生old/new共同收益|
|矩阵|固定63行；`M0/M_HEAD_GROUND/M_JOINT_SEAM`三臂，共189个prediction单元|
|性能边界|source-held研发证据，非Target Phase2、非promotable|

## 方法与裁决

|arm|机制|
|---|---|
|`M0`|原Student-t qKNN|
|`M_HEAD_GROUND`|固定`alpha=0,v_h=0,a=g`的单位质量ground-anchor expert|
|`M_JOINT_SEAM`|完整球面LOO共享运动＋连续收缩＋unit-mass qKNN|

不设置`M_DA`，因为anchor motion脱离anchor expert没有prediction输出。报告`HEAD_GROUND_VS_M0`、`SEAM_MOTION_AT_HEAD`和`JOINT_VS_M0`的同row效应。主要裁决同时看K1登记42行的old BA、seen-new、H、old floor及negative tail；不以单一receiver、K或边际最大值晋级。若联合方法整体或negative tail明确恶化，则关闭D112，不调参、不复跑。

## 冻结输入、版本和验证

|项目|值|
|---|---|
|source-held archive SHA256|`f2ceae1b47f84027f21c561bd58f50cc9df5c511e4b8d110e04e8062db6bee41`|
|source-held manifest SHA256|`155d6ed4f75ec5f236da5169229d355a2cbfccadaec60c5ede61ed1e81235b94`|
|D106 tap SHA256|`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`|
|checkpoint SHA256|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|G1 surface commit|`d8805391`|
|三臂理论commit|`eb260250`|
|ground-head commit|`ca6db16d`|
|runner commit|`f43f0532`|
|本地验证|`ssr-gpu`；runner编译通过；D112 core＋runner共25项聚焦测试通过；`git diff --check`通过|

## 冻结运行面

- CWD：`E:\type10-7\code\snapshots\cdom_scxmap_d92_glf_r1_wt`。
- 环境：`ssr-gpu`；纯NumPy特征推断，不使用GPU，不使用N607。
- 输入根：`E:\type10-7\automation_reports\CV-SincNet\d110_g1_sourceheld_usqknn_20260802_040736_r1\input\d110_sourceheld_split`。
- 输出根：`E:\type10-7\automation_reports\CV-SincNet\d112_g1_sourceheld_seam_20260802_r1\artifacts`，所有输出不可覆盖。
- 顺序：`prepare`→`predict`→确认63行／189单元→`score`；预测入口没有truth参数。

精确命令使用`code/scripts/run_d112_g1_sourceheld_one_shot.py`：

1.`prepare --source-val-archive <input>/scorer_only/source_val/features.npz --source-val-manifest <input>/scorer_only/source_val/manifest.json --output-dir <artifacts>/packages`
2.`predict --package-root <artifacts>/packages --d106-tap-archive <D111-G0-input>/d106_ls_strict_tap.npz --d106-tap-receipt <D111-G0-input>/d106_ls_strict_tap.receipt.json --d106-tap-archive-sha256 48b92f...fa2f --checkpoint-sha256 2699ee...1c98 --run-id d112_g1_sourceheld_seam_20260802_r1 --output-dir <artifacts>/predictions`
3.`score --prediction-root <artifacts>/predictions --truth-json <artifacts>/packages/scorer_only/truth.json --truth-input-seal-json <artifacts>/packages/scorer_only/truth_input_seal.json --truth-open-event-json <artifacts>/truth_open_event.json --output-json <artifacts>/held_scores.json`

技术停止只针对输入／SHA／覆盖、非有限数、确定性异常或零prediction；禁止依据中间accuracy、H或floor停止。预期artifact为21个package、63个prediction row、prediction manifest、truth-open event和63行同row score。

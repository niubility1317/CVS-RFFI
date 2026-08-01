# D110-SCPM＋US-qKNN新source-held G1四臂报告

状态：`LOCAL_VERIFIED / RELEASE_FROZEN / NEW_SPLIT_SEALED_UNOPENED / NOT_LANDED / NO_PERFORMANCE_RESULT`

## 1.身份与目标

- run ID：`d110_g1_sourceheld_usqknn_20260802_040736_r1`
- 日期：2026-08-02
- operator：主agent冻结与分析；Terra Max分别实现DA×HEAD核心和新split；后续唯一Terra Max runner负责N607。
- objective：在未打开的新source-held split上完整运行`M0/M_DA/M_HEAD/M_JOINT`63行，分析DA、HEAD、交互、old/new平衡、floor和negative tail。
- 性能边界：新split完成全部预测封存前不得开启truth；历史D104 split只允许机械回归，不能用于D110晋级。

## 2.冻结四臂

|arm|距离|qKNN尺度|
|---|---|---|
|M0|identity平方距离|现有class-specific Student-t尺度|
|M_DA|D110 safe-relative SCPM Mahalanobis距离|同一class-specific公式|
|M_HEAD|identity平方距离|全注册类shared尺度|
|M_JOINT|同一SCPM距离|同一shared尺度公式|

- 四臂共用INT8 support decode、`nu/d_eff/gamma/h0/lambda/clip`、Student-t核与`logsumexp-minus-log-K`。
- shared K>1先求每类无序support pair均值，再做等类平均与原shrink/clip；K1严格取`h0`，所以`M_HEAD=M0`、`M_JOINT=M_DA`是预期可辨识边界。
- SCPM只使用`safe_relative_variances`；公共`1+1/K`不进入qKNN距离。
- query零fit、零update、零selection；无role、quota、truth或batch统计。

## 3.真实G0前置证据

`d110_scpm_g0_oneshot_20260802_033047_r1`已在真实588条tap上完成：K1/K5/K10的argmax变化分别为23/40/96，`zero_changed=[]`，资源预算未超，结论为`G0_PASS_PROCEED_G1 / NO_PERFORMANCE_RESULT`。

## 4.新source-held split预登记

- split ID：`d110_source_seed110813_v1`
- salt：`D110-SCPM-USQKNN|source-held|110813|v1`
- source pool：8400条既有单观测source feature；不重新forward。
- 排除：D103历史query 2478条、D104已打开held 2520条、D110 Phase1 L_s 588条；按physical ID取union。
- 容量：排除后168个receiver×TX×day cell最小合法容量为7，因此固定每cell 7条，共1176条；42个receiver×TX组各28条，四天等权。
- G1结构：7receiver×K1/K5/K10一般行21个，加7receiver×6 held-class×K1行42个，共63行×4臂=252预测单元。
- 每类query数：K1=27、K5=23、K10=18。

### 4.1冻结本地输入

|输入|SHA256|
|---|---|
|D103 8400 dual archive|`dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0`|
|D105 8400 strict pre-ReLU tap|`6626afbf5d5987b2944b53f9b4bddbb6c9397f4c577accb95cea5e0039b24578`|
|D103历史query manifest|`3fd07b7afcb53b12a08df1643efae80c52917c893cc7453104e68932dc1f5b26`|
|D104 no-truth package manifest|`55780d103e2cb19b4446ef44033446182beac992b42c59753b6910b0910e6710`|
|D104 L_s 588 ID source|`dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d`|

builder已证明D103/D105 physical ID同序、`ReLU(pre_relu)=z_id`逐值相同、`z_dom`逐值相同；只输出scorer-only archive、manifest和不含truth值的selection receipt，输出目录不可覆盖。

### 4.2真实split封存结果

状态：`D110_SOURCE_HELD_SPLIT_COMPLETE / UNOPENED_FOR_SCORING / NO_PERFORMANCE_RESULT`

|项目|结果|
|---|---|
|row/cell/group|1176／168／42；每cell 7、每receiver×TX 28|
|排除union|5335个physical ID|
|合法剩余容量|min=7、max=29|
|与D103历史query交集|0|
|与D104旧held交集|0|
|与D110 L_s交集|0|
|selection receipt内部root|`0dd909f39ede47c9ec43e57d67be05a74f54838898880bb16ef5c0264511d852`|
|selection receipt文件SHA|`a1d6c60150e068bd8342dfabf6e74b45f1d41769ea0fa84508e6213817ff3539`|
|scorer manifest SHA|`155d6ed4f75ec5f236da5169229d355a2cbfccadaec60c5ede61ed1e81235b94`|
|scorer archive SHA|`f2ceae1b47f84027f21c561bd58f50cc9df5c511e4b8d110e04e8062db6bee41`|
|selection receipt含truth值|false|
|performance computed|false|

本地封存目录：`input/d110_sourceheld_split/`。截至本节写入时未调用score、未读取新split性能、未生成任何预测。

## 5.本地实现与验证

|部分|commit|证据|
|---|---|---|
|D110理论与G0结论|`1e6b94d3`|真实G0三K非零|
|US-qKNN四臂core|`d9f8b62c`|M0 bit-match；D110专项14项通过|
|新split builder|`48a41e3b`|新旧split回归13项通过；真实split已封存且未评分|
|G1一次性入口|`ef8d3109`|7项目标测试、`py_compile`、`git diff --check`通过|

独立发布复审：`P0=0 / P1=0 / P2=0 / GO`。复审确认prediction manifest冻结truth seal SHA，score在创建truth-open event前校验prediction→package manifest→truth seal链；替换seal的负测在事件生成前fail-closed。该结论只表示发布就绪，不是性能结果。

## 6.不可变发布包

|artifact|SHA256|
|---|---|
|`release/d110_code_ef8d3109.zip`|`2ad3af6dea23c6c324a212dc7e7d19fcb900937bcdca82b0cfcd62c8edd2f28b`|
|`release/d110_sourceheld_split.zip`|`9083ddfd5abf73272c1ccd53dc7a4853321dbfb7e60a4dc166277f8a3c71ecff`|
|G1入口Python|`5f693a8438bac789b3759c2adc097aa95136256219851b7bc97d938c8c9beb71`|
|G1入口测试|`3dcd59a42dfdc2c68078252ad27b07028987edb263a61b64d4546c126fc796f7`|

## 7.N607预登记

- remote root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d110_g1_sourceheld_usqknn_20260802_040736_r1`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 运行性质：冻结NumPy特征推断，不训练；唯一runner仍需preflight和资源记录。
- 同步映射：两个本地`release/*.zip`同步到remote root，分别解压到`release_code/`与`input/`；不改动服务器已有D106 tap。
- 固定D106 tap：`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/strict_tap/d106_ls_strict_tap.npz`，SHA256=`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`。
- 唯一子进程依次执行`prepare`→`predict`→`score`；`predict`必须先封存63行、252个arm-row单元，`score`才能打开truth。
- 输出：`output/packages/package_manifest.json`、`output/predictions/prediction_manifest.json`、`output/truth_open_event.json`、`output/held_scores.json`、`logs/runner.log`、`launch.pid`、`launch.exit`。
- fresh-run retry：未授权；技术失败保留r1并返回本地修具体缺陷。
- 停止条件：仅P0协议/覆盖/错误SHA、确定性异常或零prediction；不得按性能停止。

后续只补：N607实际命令、PID/GPU/进程与日志证据、完整同row结果及推荐。


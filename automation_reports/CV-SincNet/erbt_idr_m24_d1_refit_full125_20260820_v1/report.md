# ERBT-IDR M2.4 D1-REFIT完整125实验报告

日期：2026-08-20

run ID：`erbt_idr_m24_d1_refit_full125_20260820_v1`

当前状态：`LOCAL_VERIFIED`

## 一、目标与方法

本实验按《M2.4 D1扩展实验全面复盘与问题定位》比较三个严格同row方法：

- R0：`M24-D0-HISTORICAL-F1`，即当前D92 E0主版本`P2-A1_NO_RF32`的历史拟合路径；
- R1：`M24-D1-COMPILE-PARITY`，从已经拟合的P2-A1头删除恒零RF32并编译为IF256，不重新估计support中心、协方差或残差；
- R2：`M24-D1-REFIT`，只从本row的IF256 support重新拟合，不读取历史P2-A1 coefficient、bias或已拟合head。

原`P2-FULL`仅作为历史含RF32对照，不再作为D92 E0默认基线。

## 二、完整125矩阵

|维度|取值|
|---|---|
|receiver|`20-1`、`3-19`、`7-14`、`7-7`、`8-8`|
|method seed|`7282101`、`7282102`、`7282103`、`7282104`、`7282105`|
|条件|`K1/new20`、`K2/new20`、`K5/new20`、`K10/new20`、`K10/new5`|
|每方法规模|5 receiver×5 seed×5条件=125行|
|总规模|3方法×125=375个方法行；每行3个`leo_*_weak`场景，共1125个场景单元|
|统计边界|同一`receiver×seed`下不同K可能复用query；按K分层25行或按`receiver×seed`聚类分析|

既有合法缓存覆盖3个seed、75个输入身份。为达到完整125，本run补建`7282104`和`7282105`对应的50个同协议输入身份；不重验`VALIDATED_ONCE`数据，不改变received IQ、协议schema或class draw。

## 三、协议与证据边界

- `protocol_schema=p2_min_v1`；
- `phase2_data_status=VALIDATED_ONCE`；
- 复用既有`capsule_id`／`split_id`规则；
- prediction阶段不读取query truth，不更新方法状态；
- 每个query独立在全部已注册类中argmax，不使用角色、类别配额、真实batch类数或全局重分配；
- 全部375个prediction闭合并写入matrix index后，独立scorer才连接truth；
- R1任一before／after预测与R0不一致属于直接技术失败；R2性能偏低不停止矩阵。

## 四、冻结实现与验证

|字段|值|
|---|---|
|分支|`work/m24-safe-residual`|
|实现提交|`4b2e42ffcbae48e4522e63359cb58b3b83f7c2ec`|
|本地环境|`ssr-gpu`|
|聚焦验证|M2.3/M2.4共53项测试通过；6个相关Python文件编译通过；controller两模式参数转发通过；`git diff --check`通过|
|Git发布|origin远端分支OID已独立回读并与本地HEAD一致|
|独立审查|首次发现2个P1：R2仍提前构造历史头、scorer未证明完整125；定点修复后仅针对原问题复审PASS|

变更文件及用途：

- `stage2_m24_safe_residual.py`：拆分D1编译等价路径与D1-REFIT身份；
- `stage2_m24_refit.py`：IF256 support独立重拟合；
- `stage2_m24_row_executor.py`：同family四状态、统一遗忘基线、truth-blind margin／中心角距侧车；
- `extend_m24_full125_inputs.py`：补建两个method seed；
- `run_m24_d1_refit_matrix.py`：完整125×3方法prediction；
- `score_m24_d1_refit_matrix.py`：truth-last同row评分、help/harm与`F_within`／`F_std`。

## 五、N607路径与资源

|字段|路径或设置|
|---|---|
|项目根|`/home/szu2070436088/2510044040/CV-SincNet`|
|release|`/home/szu2070436088/2510044040/CV-SincNet/releases/erbt_idr_m24_d1_refit_full125_20260820_v1`|
|既有feature root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features`|
|既有scoring root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_sidecar_t1_20260730_v3_47212437/artifacts/sidecars`|
|补充package root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_packages`|
|补充feature root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_features`|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m24_d1_refit_full125_20260820_v1`|
|log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/erbt_idr_m24_d1_refit_full125_20260820_v1`|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|prediction设备|`cpu`，最多2个worker|
|补充feature设备|按GPU实时空闲slot分配；每GPU总训练进程不超过2|

启动前N607只读盘点显示8张GPU均有其他Phase1训练，当前各1个compute进程。补充feature构建只占每GPU允许的第二slot；不停止、不修改其他任务。R0／R1／R2矩阵在CPU运行。

## 六、运行顺序与停止规则

1.发布单一release归档，并只做一次本地／远端归档SHA比较和一次远端编译。
2.使用既有受控controller补建`7282104`、`7282105`的30个package和50个feature身份。
3.先执行一个真实checkpoint无truth smoke；PASS后立即进入完整375行prediction。
4.启动后核对一次PID、CWD、cmdline、run root、GPU映射和日志增长。
5.只有协议／query泄漏、错误矩阵身份、输出碰撞、错误checkout、不能启动、无prediction闭合或至少两行出现相同确定性prediction前异常才停止；不得因中间性能停止。
6.375行闭合后truth-last评分，随后按总体、K、receiver、seed、scene、old/new、class、margin、中心角距及help/harm方向分析。

## 七、预期artifact

- `predictions/matrix_index.json`：125个配对输入身份、375个方法行；
- 每行`predictions.cvspred`、`row_execution_receipt.json`、`truth_blind_diagnostics.npz`；
- `scores/scored_matrix_index.json`及每行same-row、four-state、paired-vs-R0、标准化遗忘结果；
- 最终`results_summary.json`、完整报告和D92 E0总报告追加章节。

## 八、结果

### 8.1技术裁决

本run于2026-08-20产生375个`row_execution_receipt.json`后停止，状态为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。控制器没有生成`predictions/matrix_index.json`，独立scorer未连接truth，因此本run没有合法性能结果，也不能进入R2与R0的晋级比较。

停止原因是R1编译等价硬约束失败。唯一失败行是`rx20-1_m7282101_k10_new20__M24-D1-COMPILE-PARITY`：注册后预测与R0完全一致（0/1560不一致），但注册前预测出现1/1560不一致。差异位于`leo_rain_weak`的query`qid_29b3f575fa647024e1bf91c23e216fcdb002e7ce0bfb422c5893eed0852115a6`；R0注册前预测为`cls_75aa6d506081240f50cf3b79a0bd91714fa0084a635a472ca63194e57ec1dca2`，R1注册前预测为`cls_8b02d99905a8fe579368ac8e37eff51c505aaa89a646eba8892d5d800aa08416`。

### 8.2证据边界与处置

- 预测父进程PID`3738921`已退出；控制器日志记录`RuntimeError: D1 compile parity failed`。
- 375个row receipt及其sealed prediction、truth-blind diagnostics全部原地保留，不删除、不覆盖。
- `matrix_index.json`不存在；query truth保持关闭，未创建`scores/`。
- 不使用375个孤立receipt补造矩阵索引，不忽略1个注册前差异，不以局部或非闭合结果形成性能结论。
- 本轮未重启、未重复启动失败行。后续若修复编译等价实现，必须使用新的不可覆盖run ID重新取得完整125×3预测证据；本run永久保留为技术失败记录。

机器可读裁决见同目录`results_summary.json`。

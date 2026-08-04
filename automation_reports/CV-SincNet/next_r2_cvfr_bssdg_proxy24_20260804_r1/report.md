# NEXT-R2 CVFR-BSSDG Proxy24实验报告

## 登记

- 实验ID：`next_r2_cvfr_bssdg_proxy24_20260804_r1`
- 日期：`2026-08-04`
- 状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
- 责任：主agent负责方法、数据/结果分析和晋级；唯一Luna/max runner负责N607落地、启动、监控和artifact回收
- 目标：以最小source-held矩阵判断`CVFR-BSSDG/r1`是否产生非恒等且同row联合正收益；不是Target或正式Stage2-C性能结论

## 冻结矩阵与指标

矩阵为固定SHA256规则选择2个source-only receiver×6个held class×`K={1,5}`，共24个outer key。每个key必须产生`DA0_REG0、DA1_REG0、DA0_REG1、DA1_REG1`四态，共96份prediction。排除Target、125、K10、Cn20、超参扫描和失败后调参重跑。

|状态|定义|允许指标|
|---|---|---|
|`DA0_REG0`|域适应前/新类注册前|`A_retained、F_retained、total_correct`；`N/H=NA`|
|`DA1_REG0`|域适应后/新类注册前|`A_retained、F_retained、total_correct`；`N/H=NA`|
|`DA0_REG1`|域适应前/新类注册后|`A_retained、N_seen_new、H_retained_new、F_retained、total_correct`|
|`DA1_REG1`|域适应后/新类注册后|`A_retained、N_seen_new、H_retained_new、F_retained、total_correct`|

必须报告四差分`DA1_REG0-DA0_REG0`、`DA1_REG1-DA0_REG1`、`DA0_REG1-DA0_REG0`、`DA1_REG1-DA1_REG0`和retained共同指标DiD。

## 本地版本与验证

- 工作树：`E:\type10-7\code\snapshots\d92_lite125_20260804_wt`
- 发布commit：`6c254445`
- scorer初始commit：`0923a361`
- truth/split边界修复：`f1580d83`
- 集成commit：`8af10612`
- 核心修复：`d22b41a8、d2a0b418、41a8b79f、91cdf6b9`
- 设计/命名：`18b41511、06c245c6`
- 环境：`ssr-gpu`
- 验证：核心+集成+独立scorer共`67 passed`，`py_compile`与`git diff --check`通过；`predict --help`无`ls_join`参数
- 独立审查：`P0=0、P1=0、RELEASE_REVIEW_GO`
- 主要文件：`stage2_next_r2_cvfr.py、stage2_next_r2_bssdg.py、stage2_next_r2_matrix.py、stage2_next_r2_runtime.py、stage2_next_r2_real.py、stage2_next_r2_score.py、run_next_r2_proxy24.py、score_next_r2_proxy24.py`及对应测试

## 固定输入

|输入|N607路径|SHA256|
|---|---|---|
|checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|selected IQ|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/selected_ls_iq/d106_ls_received_iq.npz`|`e32708214eaedaf39af532c572e16045f173422d63110e4022778f3ad0252ede`|
|selected receipt|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/selected_ls_iq/d106_ls_received_iq.receipt.json`|`a18bd5d610c9874bd0d6b50d34e845d85229d5892453bce9ff5bfeaa8ee82d59`|
|L_s label join|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/input/d104_split/L_s/features.npz`|`dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d`|
|句柄|`capsule_id=d106_ls588_proxy_dd315295`；`split_id=d104_source_seed104713_v2`|固定source-held proxy|

## N607预登记

- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- CWD：`/home/szu2070436088/2510044040/CV-SincNet`
- 同步目标：仅将NEXT-R2模块与两个CLI按原相对路径同步到现有项目根；不覆盖其他文件；同步后逐文件SHA256和`py_compile`必须匹配
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/next_r2_cvfr_bssdg_proxy24_20260804_r1`
- capsule与builder receipt：`.../input/next_r2_proxy24.capsule.json`、`.../input/next_r2_proxy24.builder_receipt.json`
- prediction输出：`.../output`，启动前必须不存在
- 日志：`.../logs/build_capsule.log`与`.../logs/predict.log`
- GPU规则：预检后选择未满2个训练进程、显存占用最低、索引最小者；启动前回填精确GPU/PID
- 重试：无；技术失败保留artifact，以新run ID重新登记

builder与predict是两个独立进程。只有builder可读取`ls_join`；predict CLI仅可读取fixed IQ、capsule和checkpoint。正式命令由runner将上述绝对路径代入以下冻结入口：

```text
python code/scripts/run_next_r2_proxy24.py build-capsule --selected-iq <selected_iq> --selected-iq-sha256 e32708214eaedaf39af532c572e16045f173422d63110e4022778f3ad0252ede --selected-receipt <selected_receipt> --selected-receipt-sha256 a18bd5d610c9874bd0d6b50d34e845d85229d5892453bce9ff5bfeaa8ee82d59 --ls-join <ls_join> --ls-join-sha256 dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d --capsule-id d106_ls588_proxy_dd315295 --split-id d104_source_seed104713_v2 --capsule-output <capsule> --builder-receipt-output <builder_receipt>

CUDA_VISIBLE_DEVICES=<GPU> python code/scripts/run_next_r2_proxy24.py predict --run-id next_r2_cvfr_bssdg_proxy24_20260804_r1 --run-root <new_output_root> --checkpoint <checkpoint> --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 --selected-iq <selected_iq> --selected-iq-sha256 e32708214eaedaf39af532c572e16045f173422d63110e4022778f3ad0252ede --selected-receipt <selected_receipt> --selected-receipt-sha256 a18bd5d610c9874bd0d6b50d34e845d85229d5892453bce9ff5bfeaa8ee82d59 --capsule <capsule> --capsule-sha256 <capsule_file_sha256> --device cuda:0

python code/scripts/score_next_r2_proxy24.py --run-root <completed_output_root> --prediction-capsule <capsule> --prediction-capsule-sha256 <capsule_file_sha256> --ls-label-join <ls_join> --ls-label-join-sha256 dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d
```

## 健康、完成与晋级

- 启动后核验PID、CWD、cmdline、run root、GPU和日志增长。
- 完成条件：24/24 key、96/96 state、每态JSON/NPZ/BSSDG wire、DA1 CVFR wire、manifest/completion和独立score全部闭合。
- 只因P0协议/安全违规、错误hash/checkout、覆盖风险、launcher确定性故障，或两个不同key在prediction前出现同一异常指纹而停止；绝不因中间性能停止。
- 若全部DA1 identity或预测与DA0相同，记`NO_FUNCTION`并关闭。
- 否则K5 pooled的`DA1_REG1-DA0_REG1`须同时满足`ΔH>0、Δtotal_correct>0、ΔA_retained≥0、ΔN≥0、ΔF_retained≥0`才进入更完整source-held复核；失败即关闭，不调参、不重跑。

## 结果回填

|candidate|receiver/TX|K|DA状态|注册状态|A_retained|N_seen_new|H_retained_new|F_retained|total_correct|CVFR状态|资源|结论|
|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|---|
|`CVFR-BSSDG/r1`|待capsule冻结|1/5|`DA0/DA1`|`REG0/REG1`|待评分|REG0为NA|REG0为NA|待评分|待评分|待运行|待回填|待判定|

## 技术失败结案（2026-08-04）

- N607直连预检通过，8个NEXT-R2文件按正确包路径落地并通过远端SHA和`py_compile`核验；GPU0执行。
- builder retry1成功：capsule SHA=`070385a5acd07508c23d5a55aba0041610085e5a8a66a2ff1489a53a0a91fb03`，matrix SHA=`8f4a5d9c98e4f714883beaf3f9602b8cb0b12d90268dd9055f32200763d73425`。
- truth-free real smoke成功：receipt SHA=`e41a1a5809ff2f27360cd032d9b63258c710edbb961c4eac05c92aa893fed24f`；canonical重复精确、pre-ReLU160有限、query truth未进入prediction进程。
- predict PID=`1111152`自然退出前已形成96份JSON、96份NPZ、96份BSSDG wire和48份CVFR wire，但manifest因错误禁止重复内容SHA而失败；未生成manifest/completion，未打开truth，未运行score。
- 96份NPZ只形成28个内容SHA组；这是合法的内容相同、路径不同情形。该证据只能定位封存缺陷，不能解释为性能或`NO_FUNCTION`。
- 完整partial artifact已回收至根报告目录下`artifacts/technical_failure_partial_20260804/`；关键predict log SHA=`0a50b130477616651f3c5c7b34f5c86d3b0fd1a986b2180cf1294856219a2ba6`。
- 修复commit=`4aea94c7`仅把manifest唯一性约束从内容SHA改为run-root相对artifact路径；方法、矩阵、输入和评分规则均未变化。r1不得续跑、封存、重标或评分；后续必须使用新run ID。

根工作区同名报告保存完整landing、依赖闭包、smoke、PID、GPU、artifact计数和异常证据。

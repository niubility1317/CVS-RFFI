# DSSC-ZDOM-JG-qKNN-R4-BCRR/r1f完整125实验报告

## 运行身份与状态

- run ID：`dssc_zdom_jg_qknn_r4_bcrr_full125_r1f_849fa342_20260723_141937`
- 创建时间：`2026-07-23T14:19:37+08:00`
- candidate：`DSSC_ZDOM_JG_QKNN_R4_BCRR/design-r1f`
- 当前状态：`RUNNING / NO_PERFORMANCE_RESULT`
- 方法Git提交：`849fa342cd46cb8294b5d9b4f5358cea630d0643`
- 发布报告Git提交：以包含本文件的后续提交为准；不改变方法提交。
- sole launch owner：`/root/dssc_r1f_full125_runner`（`gpt-5.6-terra high`）；主agent未并发启动本run。
- retry：`false`。任何技术失败都不得覆盖本run root；需要修复时必须新revision或新run ID。

## 目标与假设

本run直接执行冻结完整125，不先发布单receiver、单seed、单K或单scene性能子集。假设ground约束的共享rank4模型adapter能在固定qKNN头上产生独立目标域净正确决策增益，BCRR解决不同的old/new尺度与floor误差；二者联合应严格优于两个单组件并得到正`I_syn`。

同row五臂固定为`M0/M_DA_NG/M_DA/M_OTHER/M_JOINT`：`M_DA_NG`与`M_DA`唯一差异是ground prior mask；BCRR是唯一OTHER。`I_syn=H(M_JOINT)-H(M_DA)-H(M_OTHER)+H(M0)`。

## 冻结矩阵

- receiver：`20-1,3-19,7-14,7-7,8-8`
- seed：`713102,713103,713104,713105,713106`
- registration slice：`(K10,new5),(K10,new10),(K10,new20),(K5,new20),(K1,new20)`
- 每job场景：`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`
- 总数：`125 jobs / 375 prediction slices / 1875 score rows`
- 每个query独立面对全部注册类；query truth只在五臂不可变prediction全部落盘后由scorer读取。
- 调度：LPT动态队列；preflight确认安全时使用GPU0–7各1个worker。部分GPU不安全时只减少并发、不缩窄矩阵。

## 本地冻结与独立复核

- `ssr-gpu`下4个DSSC文件`py_compile`通过。
- 专项、协议负例和artifact篡改测试：`21/21 passed`。
- 真实ADV3B02 checkpoint support-only无query smoke：`REAL_CHECKPOINT_SUPPORT_ONLY_NOQUERY_SMOKE_PASS`；ground/no-ground均只训练4个共享系数，S_B/S_C均非identity并merge，`query_packages_loaded=false`、`query_rows_used_for_fit=0`。
- 独立终审：`MERGE / P0=0 / P1=0`。
- 本地smoke receipt SHA：`0d84219d5c325a0695a73225d880295fdfe99334034971daab4ed57f16008cab`。

## 发布包与不可变输入

|artifact|SHA256|
|---|---|
|Git source ZIP|`d53bb1758ef77ef563d615325ceb999d578ddb9ffd3bd07ccc17ae392a7c7e1a`|
|DSSC method lock|`7663bbc4b7b199d98caa85b7736547a6927a2c7eb8e6a4de636967edca1e9c10`|
|DSSC ground bundle|`109724913cac4f82ff58359b927a7f1e7f7e7d233c0bfd0d05d323f94b1b12da`|
|SOMPH package lock|`0496594db4a82efbbf17ec3d67ebc3fb1f0c7ced41b542a5a0bde3482e704523`|
|GEOFF/r8 coverage receipt|`c6e25ebeaed32b577e3321e78cd569acff934a7c804d0cb621b26e68f26d0c17`|
|Phase1 checkpoint|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|sealed runtime|`f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a`|
|Phase1 archive|`dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0`|
|Phase1 archive manifest|`34213331d20594dceface61680ab0fea8ffc40ee72d7e13c844763c70fef26d4`|
|Phase1 parity receipt|`b93219c40b79be8ecdf8c0a51d77710d8119f8899331ae7e2518b77adfeac60b`|

科学文件SHA：方法模块=`670e1520dc5b2879c6101e174444b33602d2ac88a9d2502281b48159046c91ea`；bundle builder=`b1fa7e3cb33e0d82358d14d5d79dca875c504553afe6ecb7bac17f5665edd401`；完整125 runner=`ac687baf67be21cd930d490155687d7be211d0c3eea09b753ebf8b351080e335`；专项测试=`e291862479bccc3a0210c71b43c261af56a72d6bd7071356293850bc73897a1b`。

本run不生成或重验数据。archive、manifest、parity和coverage均复用既有`VALIDATED_ONCE`/GEOFF/r8资产；runner只验证固定SHA、schema和远端可见性。当前本地状态：archive=`PRESENT_REUSED`，coverage=`PRESENT_REUSED`，parity receipt=`PRESENT_REUSED`。

## N607路径与启动合同

- direct preflight：`powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`
- 远端Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 远端项目根：`/home/szu2070436088/2510044040/CV-SincNet`
- 远端run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/dssc_zdom_jg_qknn_r4_bcrr_full125_r1f_849fa342_20260723_141937`
- 解压源码：`<run>/source`
- 输入：`<run>/input`
- 主日志：`<run>/logs/full125.stdout.log`与`<run>/logs/full125.stderr.log`
- PID：`<run>/launcher.pid`
- 矩阵输出：`<run>/artifacts`，启动前必须不存在。
- cache根：`/home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/cache_matrix`
- authority根：`/home/szu2070436088/2510044040/CV-SincNet/runs/d81_comprehensive_125_v2auth_20260720/authority_controller/authority_final_retry1`
- checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- sealed runtime：`/home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt`

runner在同步后必须重算源码ZIP、4个input文件、checkpoint、runtime和所有远端固定输入SHA，完成解压源码`py_compile`，确认run/artifacts/log路径不可覆盖，再以detach方式执行以下等价命令；`--gpu-ids`由实时preflight安全slot决定，默认`0,1,2,3,4,5,6,7`：

```text
env PYTHONPATH=<run>/source/code OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python <run>/source/code/scripts/run_dssc_zdom_jg_qknn_r4_bcrr_125.py matrix --phase1-checkpoint <checkpoint> --sealed-runtime <sealed-runtime> --package-method-lock <run>/input/somph_method_lock.json --dssc-method-lock <run>/input/dssc_method_lock.json --ground-bundle <run>/input/phase1_dssc_zdom_jg_ground_bundle.npz --coverage-receipt <run>/input/coverage_receipt.json --cache-root <cache-root> --authority-root <authority-root> --run-root <run>/artifacts --gpu-ids 0,1,2,3,4,5,6,7
```

## 监控、artifact与停止条件

runner只使用短连接监控，不杀进程、不自动重启、不修改方法。必须回传：远端PID、自然exit、逐GPU分配、`matrix_manifest.json`、125份launcher receipt、125份row receipt、`matrix_exit.json`、`aggregate_index.json`、完整日志和artifact SHA清单。技术成功门为`job_count=125`、`prediction_slice_count=375`、`score_row_count=1875`且全部真实artifact/hash/row绑定复算通过。缺完整prediction只能记`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。

性能分析必须同row报告old-before、old-after、old adaptation gain、seen-new、H、BA、floor、min-old、min-new、forgetting、old→new/new→old、逐类、receiver、scene、K、seed、coverage、量化margin、MAC、时延、显存、state bytes及五臂完整比较。立即判负条件包括：M_DA净正确决策不为正或old/new任一净负；M_OTHER无独立正收益；M_JOINT.H不严格胜M_DA和M_OTHER；mean`I_syn<=0`；正协同少于188/375个scene slice或少于2/3个scene均值为正；联合臂损害任一保护指标或增加forgetting/双向混淆；协议、INT8或资源门失败。完成prediction但未过门记`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不得用125反向调参。

## Runner回填区

- landed：`PASS`。direct preflight、远端run-root不可覆盖检查、5项上传输入、checkpoint/runtime SHA和4文件`py_compile`均通过。
- remote PID：`742449`；启动5秒后主launcher存活。
- GPU：`0,1,2,3,4,5,6,7`；启动前每GPU既有compute=1，新增后仍满足每GPU不超过2个训练进程。
- remote exit：`RUNNING`；`matrix_manifest.json`已生成，`matrix_exit.json`尚未生成。
- parity receipt：`PRESENT_REUSED / b93219c40b79be8ecdf8c0a51d77710d8119f8899331ae7e2518b77adfeac60b`
- archive：`PRESENT_REUSED / dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0`；manifest=`34213331d20594dceface61680ab0fea8ffc40ee72d7e13c844763c70fef26d4`
- coverage：`PRESENT_REUSED / c6e25ebeaed32b577e3321e78cd569acff934a7c804d0cb621b26e68f26d0c17`
- prediction/score：运行中，尚未形成完整计数。
- 最终性能表与裁决：待artifact完整后回填

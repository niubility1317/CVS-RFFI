# DSSC-ZDOM-JG-qKNN-R4-BCRR/r1f-techfix2完整125实验报告

## 身份与状态

- run ID：`dssc_zdom_jg_qknn_r4_bcrr_full125_r1f_techfix2_b77cc6c4_20260723_155058`
- 创建时间：`2026-07-23T15:50:58+08:00`
- operator：主agent；sole launch owner：`/root/dssc_pkgfix1_independent_review`（`gpt-5.6-terra high`）
- candidate：`DSSC_ZDOM_JG_QKNN_R4_BCRR/design-r1f`；implementation tag=`techfix2`
- 状态：`RUNNING / NO_PERFORMANCE_RESULT`
- 科学方法提交：`849fa342cd46cb8294b5d9b4f5358cea630d0643`
- techfix2代码与source package提交：`b77cc6c463b3ee7be6c93392171e6c99cdc21432`
- 报告Git承载：本文件由独立report-only提交纳入版本库，不包含于上述source ZIP；准确提交以`git log -1 -- <本报告路径>`为准
- parent预启动失败run：`dssc_zdom_jg_qknn_r4_bcrr_full125_r1f_techfix1_pkgfix1_f02dd8b0_20260723_150624`
- retry=`false`；本run必须使用全新本地与远端root，不得复用或覆盖任何parent路径。

## 目标、假设与唯一技术delta

目标是在不改变候选、数据、五臂、矩阵或参数的前提下，闭合sealed TorchScript固定`cuda:0`与多物理GPU调度的设备冲突，并取得完整125的真实prediction与同row性能结果。

唯一技术delta是：父launcher为每个物理GPU worker设置独立`CUDA_VISIBLE_DEVICES=<physical_gpu>`，子row统一使用逻辑`cuda:0`。因此仍使用物理GPU0–7共8个LPT动态worker，而不是只使用物理GPU0。row从实际环境和PyTorch采集固定schema execution evidence，严格要求单卡命名空间、`device_count=1/current_device=0`，再与launcher的physical/visible/logical映射逐行交叉绑定。方法、loss、adapter、qKNN、BCRR、INT8、fallback、数据、全部锁定SHA及decision geometry均不变。

本地`ssr-gpu`验证：runner与专项测试`py_compile PASS`；专项、协议负例及设备映射篡改`29/29 passed`；`git diff --check PASS`。首轮独立review发现row receipt未回绑实际命名空间的P1，修复后复审裁决=`MERGE / P0=0 / P1=0`。pytest退出后的Windows临时符号链接清理告警不影响退出码0。

## 冻结完整125与裁决门

- receiver：`20-1,3-19,7-14,7-7,8-8`
- seed：`713102,713103,713104,713105,713106`
- slice：`(K10,new5),(K10,new10),(K10,new20),(K5,new20),(K1,new20)`
- scene：`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`
- 总数：`125 jobs / 375 prediction slices / 1875 score rows`
- 五臂：`M0/M_DA_NG/M_DA/M_OTHER/M_JOINT`
- 调度：预期GPU0–7各1个本run worker；若实时安全门禁止某卡，只能降低并发，不能缩窄125矩阵。
- `I_syn=H(M_JOINT)-H(M_DA)-H(M_OTHER)+H(M0)`。

技术完成必须有125份launcher receipt与row receipt、375份prediction slice、1875行score、完整`matrix_exit.json/aggregate_index.json`及全部artifact/hash/row闭合；否则严格标记`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不retry。

性能完成后必须同row报告old-before、old-after、old adaptation gain、seen-new、H、BA、floor、min-old、min-new、forgetting、old→new/new→old、逐类、receiver、scene、K、seed、new-count、coverage、量化margin、MAC、时延、显存和state bytes。M_DA或M_OTHER无独立正收益、JOINT不胜两者、mean`I_syn<=0`、正协同少于188/375或少于2/3个scene、任一保护指标退化、协议/INT8/资源门失败，均裁为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`；不得用125反向调参。

## 不可变发布包

|artifact|SHA256或闭合值|
|---|---|
|source ZIP|`8e1c145cb65e7a49a0d3e45b21804f0b3bb09e17a899c1f39e7401ea20cb7676`；32,995,113B；deflate9|
|source tree|`b77cc6c463b3ee7be6c93392171e6c99cdc21432`；3,958/3,958个regular safe entry与raw Git blob一致|
|ZIP runner entry|`ab29bb5c6ceb64e16f5cea3c5d91948d061ad123847f027f44abe7392316f55b`|
|ZIP test entry|`e5618e64db48d8a1c1ad658b22819c70e96bb8ad4fe2c5414f9ac43858748aba`|
|DSSC method lock|`7663bbc4b7b199d98caa85b7736547a6927a2c7eb8e6a4de636967edca1e9c10`|
|ground bundle|`109724913cac4f82ff58359b927a7f1e7f7e7d233c0bfd0d05d323f94b1b12da`|
|SOMPH lock|`0496594db4a82efbbf17ec3d67ebc3fb1f0c7ced41b542a5a0bde3482e704523`|
|coverage|`c6e25ebeaed32b577e3321e78cd569acff934a7c804d0cb621b26e68f26d0c17`|
|checkpoint|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|sealed runtime|`f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a`|
|Phase1 archive/manifest|`dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0` / `34213331d20594dceface61680ab0fea8ffc40ee72d7e13c844763c70fef26d4`|
|parity receipt|`b93219c40b79be8ecdf8c0a51d77710d8119f8899331ae7e2518b77adfeac60b`|

本run只同步source ZIP与4个小型input；checkpoint和sealed runtime使用远端既有冻结路径。GEOFF/r8 archive、manifest、parity和coverage均为`PRESENT_REUSED`，不得生成、修改或重复验证数据。

## N607发布合同

- preflight：`powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`；direct N607优先，只有direct网络路径失败且身份无歧义时才使用既定lab bridge。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- remote run：`/home/szu2070436088/2510044040/CV-SincNet/runs/dssc_zdom_jg_qknn_r4_bcrr_full125_r1f_techfix2_b77cc6c4_20260723_155058`
- source ZIP：`<run>/input/source_b77cc6c4_rawblob_deflated.zip`；解包到`<run>/source`
- cache：`/home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/cache_matrix`
- authority：`/home/szu2070436088/2510044040/CV-SincNet/runs/d81_comprehensive_125_v2auth_20260720/authority_controller/authority_final_retry1`
- checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- runtime：`/home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt`
- output：`<run>/artifacts`，启动前必须不存在；顶层日志=`<run>/logs/matrix.stdout.log`与`matrix.stderr.log`；PID=`<run>/launcher.pid`；exit=`<run>/launcher.exit`。

Runner必须先验证远端run root不存在，再同步5个文件，核验ZIP整体SHA、3,958个safe regular entry、runner/test条目SHA、4项input SHA、checkpoint/runtime SHA和`py_compile`。预启动GPU1 smoke必须在独立进程设置`CUDA_VISIBLE_DEVICES=1`，确认`device_count=1/current_device=0`，以逻辑`cuda:0`加载sealed runtime并完成零IQ`[2,2,256]→[2,160]`有限FP32前向，同时导入runner设备/NumPy2边界；不得读取target package、query或生成prediction。

实时GPU安全门通过后，只允许sole runner短连接detach启动一次：

```text
env PYTHONPATH=<run>/source/code OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python <run>/source/code/scripts/run_dssc_zdom_jg_qknn_r4_bcrr_125.py matrix --phase1-checkpoint <checkpoint> --sealed-runtime <runtime> --package-method-lock <run>/input/somph_method_lock.json --dssc-method-lock <run>/input/dssc_method_lock.json --ground-bundle <run>/input/phase1_dssc_zdom_jg_ground_bundle.npz --coverage-receipt <run>/input/coverage_receipt.json --cache-root <cache> --authority-root <authority> --run-root <run>/artifacts --gpu-ids 0,1,2,3,4,5,6,7
```

不得修改方法、参数、输入、GPU命名空间schema或矩阵；不得kill/restart/retry。超时后先做只读landed probe。监控只能使用短SSH并在每次结束后确认本地无残留SSH连接。完成后回收完整日志、PID/exit、matrix/aggregate、125 launcher与row receipt、prediction、score及SHA清单；不得回收数据、checkpoint或runtime。

## Runner与性能回填

|字段|当前值|
|---|---|
|direct/bridge preflight|`PASS / direct`；未使用bridge|
|远端root不可覆盖检查|先确认`ABSENT`后排他创建，`PASS`|
|source/input/checkpoint/runtime SHA|`PASS`；ZIP整体、3,958/3,958 raw Git blob、runner/test条目、4项input、checkpoint/runtime与`py_compile`闭合|
|GPU1隔离smoke|`PASS`；`CUDA_VISIBLE_DEVICES=1`、`device_count=1/current_device=0`、logical `cuda:0`、zeroIQ `[2,2,256]→[2,160]`有限FP32、NumPy2 byte-equal、target/query/prediction=0|
|GPU安全slot与实际`--gpu-ids`|GPU0–7启动前均无compute process；实际`0,1,2,3,4,5,6,7`|
|PID / launch exit|wrapper PID=`796973`，matrix PID=`796975`；唯一启动时间=`2026-07-23T08:16:38Z`；`launcher.exit`尚不存在|
|自然完成exit|待回填|
|launcher receipt / row receipt|启动后首个短连接为`32 / 0`；首批作业运行中|
|prediction slice / score row|0 / 0|
|archive / parity / coverage|`PRESENT_REUSED`；archive=`dd2a2b0…`、manifest=`34213331…`、parity=`b93219c4…`、coverage=`c6e25ebe…`，只做冻结SHA/receipt绑定，不重验内容|
|最终状态|`RUNNING / NO_PERFORMANCE_RESULT`|

启动后短连接核验：wrapper与matrix进程均存活，CWD均为冻结run的`source`目录，cmdline完整绑定本run matrix入口与冻结输入；8张GPU均出现本run Python worker，显存约592–596MiB、利用率18%–23%。顶层stdout/stderr当时均为0B且无错误。每次SSH结束后本地`ssh.exe=0`，到N607及bridge的`ESTABLISHED:22=0`。retry、kill、restart均为0；以上只证明技术进程已落地，不是prediction或性能结果。

完成后在本报告追加同row五臂性能表、逐receiver/scene/K/seed/new-count、逐类、transition、coverage、量化、资源、异常与最终`MERGE/REVISE/REJECT`或性能裁决。

# DSSC-ZDOM-JG-qKNN-R4-BCRR/r1f-techfix3-releasefix1完整125实验报告

## 身份与状态

- run ID：`dssc_zdom_jg_qknn_r4_bcrr_full125_r1f_techfix3_releasefix1_3bc31826_20260723_172109`
- 创建时间：`2026-07-23T17:21:09+08:00`
- operator：主agent；sole launch owner：待分配`gpt-5.6-terra high`runner
- candidate：`DSSC_ZDOM_JG_QKNN_R4_BCRR/design-r1f`；implementation=`techfix3`；release correction=`releasefix1`
- 状态：`PREREGISTERED / NO_PERFORMANCE_RESULT`
- 科学方法提交：`849fa342cd46cb8294b5d9b4f5358cea630d0643`
- 冻结source tree：`3bc318266006a040bb5957b297a4a6cd2345a0f2`
- parent：`dssc_zdom_jg_qknn_r4_bcrr_full125_r1f_techfix3_3bc31826_20260723_165358`，自然exit1，因预建空`artifacts`在matrix入口失败；无query/prediction/score。
- 本run使用全新本地与远端root；不得复用、覆盖、restart或retry任何parent路径。

## 唯一发布修正

方法代码、五臂、输入、矩阵、参数、loss、adapter、qKNN、BCRR、INT8、fallback、GPU命名空间和全部冻结SHA逐字不变。唯一修正是发布准备只创建远端run root、`input/`、`source/`和`logs/`；`<run>/artifacts`在detach前必须`ABSENT`，由matrix作为`--run-root`自行原子创建。不得增加新门、修改runner或用旧run续跑。

parent已经证明source/input/checkpoint/runtime SHA、3,959个Git blob、py_compile、GPU1 zeroIQ和非字典序registry无query state均可通过；本run仍按独立run合同重新做direct preflight、精确SHA和GPU1技术smoke，不重验数据内容。

## 完整125与裁决门

- receiver：`20-1,3-19,7-14,7-7,8-8`
- seed：`713102,713103,713104,713105,713106`
- slice：`(K10,new5),(K10,new10),(K10,new20),(K5,new20),(K1,new20)`
- scene：`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`
- 五臂：`M0/M_DA_NG/M_DA/M_OTHER/M_JOINT`
- 计数：`125 jobs/375 prediction slices/1875 score rows`
- 调度：GPU0–7各最多1个本run worker，父进程设置`CUDA_VISIBLE_DEVICES=<physical>`，子进程只见逻辑`cuda:0`。

技术完成要求125份launcher与row receipt、375份prediction、1875行score、`matrix_exit.json/aggregate_index.json`和全部SHA闭合。没有prediction只能`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。性能完成后必须同row报告old-before、old-after、old gain、seen-new、H、BA、floor、min-old、min-new、forgetting、双向混淆、逐类/receiver/scene/K/seed、coverage、量化、MAC、时延、显存和state；负结果也形成完整prediction和诊断。

## 冻结发布物

|artifact|SHA256或闭合值|
|---|---|
|source ZIP|`512e1f74a27f6bed4c6d8820bf448f23c8eadf4832c9f3243e81deeff5ff689d`；33,008,269B；3,959/3,959 raw Git blob|
|runner/method/test entry|`ab29bb5c6ceb64e16f5cea3c5d91948d061ad123847f027f44abe7392316f55b` / `d087030f78f730f0eb930ab9f298ae902d9709f02d896e1a1b6e122c9127bc1f` / `b95224bed811eec738ea140e7b43928432d811a9f55634166a620164fec4d0bb`|
|DSSC lock|`7663bbc4b7b199d98caa85b7736547a6927a2c7eb8e6a4de636967edca1e9c10`|
|ground bundle|`109724913cac4f82ff58359b927a7f1e7f7e7d233c0bfd0d05d323f94b1b12da`|
|SOMPH lock|`0496594db4a82efbbf17ec3d67ebc3fb1f0c7ced41b542a5a0bde3482e704523`|
|coverage|`c6e25ebeaed32b577e3321e78cd569acff934a7c804d0cb621b26e68f26d0c17`；`PRESENT_REUSED / NOT_GENERATED`|
|checkpoint/runtime|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98` / `f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a`|
|archive/manifest/parity|`dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0` / `34213331d20594dceface61680ab0fea8ffc40ee72d7e13c844763c70fef26d4` / `b93219c40b79be8ecdf8c0a51d77710d8119f8899331ae7e2518b77adfeac60b`；`PRESENT_REUSED / NOT_GENERATED`|

## N607合同

- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- remote run：`/home/szu2070436088/2510044040/CV-SincNet/runs/dssc_zdom_jg_qknn_r4_bcrr_full125_r1f_techfix3_releasefix1_3bc31826_20260723_172109`
- cache：`/home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/cache_matrix`
- authority：`/home/szu2070436088/2510044040/CV-SincNet/runs/d81_comprehensive_125_v2auth_20260720/authority_controller/authority_final_retry1`
- checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- runtime：`/home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt`
- output：`<run>/artifacts`；detach前必须不存在；日志=`<run>/logs/matrix.stdout.log`和`matrix.stderr.log`；PID/exit=`<run>/launcher.pid`/`launcher.exit`。

唯一正式命令与parent预注册命令相同，只替换全新run绝对路径。runner必须在detach前最后一次执行只读`test ! -e <run>/artifacts`并记录PASS；随后启动一次GPU0–7完整125。不得预建output、kill、restart、retry、修改方法或缩窄矩阵。完成后回收完整日志、PID/exit、artifact及SHA inventory。

## 回填

|字段|当前值|
|---|---|
|preflight/landing/SHA|`PENDING`|
|GPU1 smoke|`PENDING`|
|`artifacts` pre-detach|`PENDING_ABSENT`|
|PID/exit|`PENDING / PENDING`|
|launcher/row/prediction/score|`0/125 / 0/125 / 0/375 / 0/1875`|
|parity/archive/coverage generation|`NO / NO / NO`；只复用冻结artifact|
|最终状态|`PREREGISTERED / NO_PERFORMANCE_RESULT`|

## Runner执行记录

- `2026-07-23T17:24:57+08:00`：direct只读preflight通过；身份为`szu2070436088@dell-DSS8440`，项目根可见，GPU0-7均为`0%/10MiB`。
- `2026-07-23T17:25:xx+08:00`：落地前安全门：全新远端run根`ABSENT`、外部cache/authority/checkpoint/runtime均`PRESENT`、`/home`剩余约7.5T；未发现同一DSSC runner。根目录`E:\type10-7`不是Git仓库；本报告按授权仅更新非Git承载面，Git镜像由主agent负责。
- 下步：创建且仅创建`<run>`、`input/`、`source/`、`logs/`，精确同步冻结5文件并核验SHA/3,959 entries/`py_compile`，随后GPU1进行zeroIQ与非字典序registry技术smoke。

## Runner落地与技术smoke证据

- landing：远端仅创建`<run>/input`、`<run>/source`、`<run>/logs`。5个冻结输入SHA与预注册表一致；source ZIP为`3,959` entries；runner/method/test-entry SHA分别为`ab29bb5c...f55b`、`d087030f...7bc1`、`b95224be...d0bb`，远端`py_compile`通过。
- GPU1技术smoke：`PASS`。物理GPU1经`CUDA_VISIBLE_DEVICES=1`映射为逻辑`cuda:0`；冻结checkpoint的zeroIQ身份特征形状为`[1,160]`；纯合成support构造的非字典序sealed registry为`old_z,new_m,old_a`；`query_rows_used=0`。`<run>/artifacts`在smoke前后均不存在。
- smoke日志保留在`<run>/logs/gpu1_technical_smoke.log`。其中前两段是runner侧临时封装诊断（本地引号转义、退化零support被qKNN teacher gate拒绝）；均未读取真实query、未产生prediction/artifacts、未启动矩阵。最终最小技术断言通过后才允许进入正式启动门。

## 唯一正式启动命令（待实时门后detach）

```text
cd <run>/source/code
PYTHONPATH=<run>/source/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python scripts/run_dssc_zdom_jg_qknn_r4_bcrr_125.py matrix --phase1-checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --sealed-runtime /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt --package-method-lock <run>/input/somph_method_lock.json --dssc-method-lock <run>/input/dssc_method_lock.json --ground-bundle <run>/input/phase1_dssc_zdom_jg_ground_bundle.npz --coverage-receipt <run>/input/coverage_receipt.json --cache-root /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/cache_matrix --authority-root /home/szu2070436088/2510044040/CV-SincNet/runs/d81_comprehensive_125_v2auth_20260720/authority_controller/authority_final_retry1 --run-root <run>/artifacts --gpu-ids 0,1,2,3,4,5,6,7
```

该命令只会在最后一次`test ! -e <run>/artifacts`记录`PASS`后detach一次。预期`launcher.pid`、`launcher.exit`、`matrix.stdout.log`、`matrix.stderr.log`以及由matrix原子创建的`artifacts/`；launch不是性能结论。

## 启动回填

- `ARTIFACTS_PREDETACH_ABSENT=PASS`已写入`logs/pre_detach_artifacts_gate.log`，随后唯一detach完成，launcher PID=`837838`。
- 首次短连接健康检查：PID存活，`launcher.exit`尚不存在；`artifacts/`已由matrix创建；GPU0-7均出现约`18-22%`利用率和约`609-611MiB`显存占用。标准输出/错误输出当时尚无内容，未观察到traceback或性能指标。
- 当前状态：`RUNNING / NO_PERFORMANCE_RESULT`。继续以短连接监控自然exit；不得以启动或GPU占用宣称性能成功。

## 停止、回收与技术失败闭环

用户在运行期间明确要求立即止损。先只读核定主PID`837838`及其子bash、matrix Python和row Python的父子关系、CWD和命令行；仅对这11个可证明属于本run的PID发送`TERM`。5秒后本run进程数为0，GPU0-7均回到`0%/10MiB`，无需升级信号，未影响其他任务。

|字段|最终证据|
|---|---|
|运行终态|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；用户授权TERM停止，非自然exit|
|主PID|`837838`；已终止且复核不存在|
|启动/完成|28个job目录已创建；20/125份launcher receipt自然完成，均为`TECHNICAL_FAILURE`；其余已启动row在止损时中断|
|row/prediction/score|`0/125`row receipt；`0/375`prediction；`0/1875`score|
|matrix汇总|`matrix_manifest.json`存在，SHA=`d24ac53df79405e9f9820de6d5dc867635106e8d7b0c3eb059f791fbbffde94b`；`matrix_exit.json`、`aggregate_index.json`不存在|
|launcher exit|`launcher.pid`已回收；因TERM中断wrapper，`launcher.exit`不存在|
|parity/archive/coverage|均为冻结复用物；本run未生成parity/archive/coverage|
|回收|已拉回`logs/`、`launcher.pid`和部分`artifacts/`至`remote_artifacts/`：1,415个文件、807,299,279B；远端与本地规范化SHA inventory均为`8cefe68543c174b69dc30e966e3a937e5b3b77149943dec8a103ca0355134eae`|

### 首个可定位根因

首个可定位失败row为`dssc_r1f_rx_20-1_s_713102_k_10_n_20`的`before/leo_clear_weak`。固定包的IQ输入本身均非零；冻结原始模型的`raw_support`60条、`raw_query`120条均无零范数，ground模型的`ground_support`60条也无零范数。经S_B ground adapter部署后的`ground_query`120条中有2条零范数（最小范数`0.0`，首个索引`7`），随后在`predict_five_arms`的ground qKNN评分路径`_svrn_scores -> score_zid_student_t_logits -> normalize_zid_rows`触发`z_id rows contain a zero-norm vector`。该ground路径由`M_DA/M_JOINT`共享；该row尚未形成任何prediction或性能指标。

对同一固定包的归一化前`feat_joint`直接复核确认这不是归一化伪影：ground S_B部署模型的support 60条为`0`零向量、最小范数`0.01040600147`；query120条为`2`零向量，索引`[7,19]`、最小范数`0.0`。因此首个产生点是ground S_B adapter部署后模型的query输出端，而不是IQ输入或support。

因此本run只证明技术失败已被安全停止并完整保留部分证据，不提供任何性能结论、方法比较或promotable声明。

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

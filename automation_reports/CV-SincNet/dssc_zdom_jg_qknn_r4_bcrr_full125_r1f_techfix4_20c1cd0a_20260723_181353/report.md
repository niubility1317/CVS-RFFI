# DSSC/r1f-techfix4完整125发布报告

## 身份与状态

- run ID：`dssc_zdom_jg_qknn_r4_bcrr_full125_r1f_techfix4_20c1cd0a_20260723_181353`
- candidate：`DSSC_ZDOM_JG_QKNN_R4_BCRR/design-r1f`
- implementation revision：`r1f-techfix4`
- 创建时间：`2026-07-23T18:13:53+08:00`
- operator：主agent；N607唯一launch owner必须为单一`gpt-5.6-terra high`runner
- 状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
- Git commit：`20c1cd0a00c568fe2a23726d13d1b3a7ba3bd6ba`
- parent failure：`dssc_zdom_jg_qknn_r4_bcrr_full125_r1f_techfix3_releasefix1_3bc31826_20260723_172109`
- parent终态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；不得续跑或覆盖

## 目标、假设与比较

目标是在不修改Phase2数据、adapter、loss、qKNN、BCRR、五臂或125矩阵的前提下，使adapted `z_id`在精确零范数行上成为全定义函数，并直接取得完整125的真实prediction和同row性能结果。冻结规则为：no-ground/ground的support/query行范数`<=1e-12`时，仅替换为同IQ、同row的raw `z_id`；非零adapted行字节不变；raw必须finite FP32且每行范数`>1e-12`，否则fail-closed。规则不读truth/role、不更新state、不跨query、不设覆盖率阈值。

parent已证明首失败row的ground S_B部署模型在`before/leo_clear_weak`产生2/120条未归一化零向量，而ground support为0/60、raw support/query均无零向量。本revision的技术假设是：精确零行identity totalization可消除qKNN定义域错误；替换率只作为coverage和科学负证据，不构成性能收益。

matched五臂保持：

- `M0`：raw qKNN；
- `M_DA_NG`：target-only rank-4 adapter＋qKNN；
- `M_DA`：ground-initialized rank-4 adapter＋qKNN；
- `M_OTHER`：raw qKNN＋BCRR；
- `M_JOINT`：ground adapter＋qKNN＋BCRR。

## 冻结矩阵与协议

- `protocol_schema=p2_min_v1`，复用`VALIDATED_ONCE`数据、GEOFF/r8 archive/manifest/parity/coverage；不重复数据验证。
- 5 receivers：`20-1/3-19/7-14/7-7/8-8`。
- 5 seeds：`713102/713103/713104/713105/713106`。
- 5 registration slices：`(K10,new5)/(K10,new10)/(K10,new20)/(K5,new20)/(K1,new20)`。
- 每job覆盖`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`。
- 预期闭合：125 jobs、375 prediction slices、1875 score rows。
- query逐样本面对全部注册类；query truth只在五臂immutable prediction全部封存后进入独立scorer。
- GPU0–7动态LPT队列；不得缩窄receiver、seed、K、new-count或scene。

## 本地变更、验证与独立review

|文件|目的|SHA256|
|---|---|---|
|`code/cvsrffi/stage2_dssc_zdom_jg_qknn_r4_bcrr.py`|typed精确零范数totalization与receipt|`d5eddeb960b764647142570650e737e8e9ae08a1c8facd01feefe81426b3c539`|
|`code/scripts/run_dssc_zdom_jg_qknn_r4_bcrr_125.py`|在五臂state/predict前totalize四个adapted support/query面并封存receipt|`fdec029e70f2132d7de91e3cee594b3cce3610a0e705d6ea3602479c1e083df3`|
|`tests/test_stage2_dssc_zdom_jg_qknn_r4_bcrr.py`|零行、非零字节、raw失败、逐query不变、无零等价和五臂闭合负例|`2b42ed7af55826c6cbf5a7876e1444cf7d13632eed6be90ec5faf4983310772b`|

- `ssr-gpu`直接DSSC专项：`36/36 passed`，exit0。
- `git diff --check`：PASS。
- 独立Terra终审：`MERGE / P0=0 / P1=0`。
- 真实checkpoint support-only无query smoke：SHA=`2699eedc...`；只打开`20-1/leo_clear/K10`的60条enrollment support；S_B ground adapter 25步；raw/ground最小范数约1；替换`0/60`；6类正式qKNN state构建成功；`query_packages_loaded=false`、`query_rows_used_for_fit=0`。

## 本地发布输入

本地输入根：`E:\type10-7\automation_reports\CV-SincNet\dssc_zdom_jg_qknn_r4_bcrr_full125_r1f_techfix4_20c1cd0a_20260723_181353\input`

|输入|SHA256|
|---|---|
|`coverage_receipt.json`|`c6e25ebeaed32b577e3321e78cd569acff934a7c804d0cb621b26e68f26d0c17`|
|`dssc_method_lock.json`|`7663bbc4b7b199d98caa85b7736547a6927a2c7eb8e6a4de636967edca1e9c10`|
|`phase1_dssc_zdom_jg_ground_bundle.npz`|`109724913cac4f82ff58359b927a7f1e7f7e7d233c0bfd0d05d323f94b1b12da`|
|`somph_method_lock.json`|`0496594db4a82efbbf17ec3d67ebc3fb1f0c7ced41b542a5a0bde3482e704523`|
|`source_20c1cd0a_rawblob_deflated.zip`|`72c7d770e45c62b195407fa350450bef3eb874da343a929588fe6dd43871c306`|

源码包为3,962个safe regular entry，路径集合与HEAD完全一致，全量entry与raw Git blob比较为`byte_mismatch_count=0`。不得使用同目录中任何其他ZIP。

冻结外部SHA：

- checkpoint：`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`；
- sealed runtime：`f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a`；
- Phase1 archive：`dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0`；
- archive manifest：`34213331d20594dceface61680ab0fea8ffc40ee72d7e13c844763c70fef26d4`；
- parity receipt：`b93219c40b79be8ecdf8c0a51d77710d8119f8899331ae7e2518b77adfeac60b`；
- GEOFF/r8 coverage：`c6e25ebeaed32b577e3321e78cd569acff934a7c804d0cb621b26e68f26d0c17`。

## N607落地与命令

远端run根必须全新且不存在：

`/home/szu2070436088/2510044040/CV-SincNet/runs/dssc_zdom_jg_qknn_r4_bcrr_full125_r1f_techfix4_20c1cd0a_20260723_181353`

只创建`input/`、`source/`、`logs/`；detach前`<run>/artifacts`必须为ABSENT，由matrix原子创建。Conda/Python为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD为`<run>/source/code`。

唯一正式子命令：

```text
PYTHONPATH=<run>/source/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python scripts/run_dssc_zdom_jg_qknn_r4_bcrr_125.py matrix --phase1-checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --sealed-runtime /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt --package-method-lock <run>/input/somph_method_lock.json --dssc-method-lock <run>/input/dssc_method_lock.json --ground-bundle <run>/input/phase1_dssc_zdom_jg_ground_bundle.npz --coverage-receipt <run>/input/coverage_receipt.json --cache-root /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/cache_matrix --authority-root /home/szu2070436088/2510044040/CV-SincNet/runs/d81_comprehensive_125_v2auth_20260720/authority_controller/authority_final_retry1 --run-root <run>/artifacts --gpu-ids 0,1,2,3,4,5,6,7
```

预期launcher evidence：`launcher.pid`、`launcher.exit`、`matrix.stdout.log`、`matrix.stderr.log`、逐row日志/receipt、`matrix_exit.json`和`aggregate_index.json`。

## 实验健康检查与停止条件

正式detach后runner必须进行首波健康检查，不得只确认PID/GPU后等待自然exit：

1.读取PID/parent-child/CWD/cmdline、GPU0–7利用率/显存、row exit、异常指纹、row/prediction/score数量和artifact闭合；
2.不得读取准确率、H或其他性能值来早停；
3.若出现任一P0协议/安全错误，或至少2个不同row在无prediction时出现相同确定性异常指纹，立即停止继续派发，核对归属后TERM且仅TERM本run进程；
4.停止后确认GPU释放、local SSH/TCP22残留为0，回收partial日志并标记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；
5.若首波技术健康，继续完整125；重点核对先前失败row的receipt是否记录ground/query替换`2/120`且ground/support为`0/60`，该检查不读取truth；
6.不得kill/restart无关任务，不得对同一run重试或覆盖。任何重试必须由主agent以新run ID重新授权。

## 成功、证伪与待回填

技术成功要求125/125 row receipt、375/375 prediction、1875/1875 score及完整matrix汇总。性能报告必须同row给出old-before、old-after、old adaptation gain、seen-new、H、BA、floor、min-old、min-new、forgetting、old→new/new→old、逐类/receiver/scene/K/seed/new-count、totalization coverage、量化margin、MAC、时延、显存和state bytes。

|字段|当前值|
|---|---|
|remote PID|`875331`，已按健康门停止并确认run进程为0|
|launcher exit|`ABSENT`，TERM中断wrapper后未生成，不虚构exit code|
|parity receipt|`PRESENT_REUSED`|
|archive/manifest|`PRESENT_REUSED`|
|coverage|`PRESENT_REUSED`|
|row/prediction/score|`0/125 / 0/375 / 0/1875`|
|最终裁决|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`|

没有完整prediction只能是技术结果，不能形成性能或promotable声明。

## N607runner执行记录

- `2026-07-23T18:19:53+08:00`：direct只读预检通过；项目根可见，GPU0–7均为`0%/10MiB`，本地SSH/TCP22连接已确认清零。
- `2026-07-23T18:20:xx+08:00`：run根创建前确认为`ABSENT`；checkpoint=`2699eedc...`、sealed runtime=`f119e8cb...`精确匹配；cache/authority根存在且无其他训练进程。仅创建`input/`、`source/`、`logs/`；根`artifacts/`保持`ABSENT`。
- `2026-07-23T18:22:xx+08:00`：5个冻结输入已同步且远端SHA均匹配；source ZIP SHA=`72c7d770...`、3962项、压缩完整性及安全路径检查通过。解包后方法、runner、测试SHA分别为`d5eddeb...`、`fdec029...`、`2b42ed7...`；固定远端Python的`py_compile`通过；8卡zero-IQ合成smoke通过，`predictions=0`、`queries=0`。
- `2026-07-23T18:23:55+08:00`：满足唯一detach前技术门槛；下一步只按本报告冻结命令启动并执行首波健康检查，不按任何性能值干预。
- `2026-07-23T18:24:xx+08:00`：唯一detach已完成，launcher PID=`875331`、PPID=`1`、CWD=`<run>/source/code`，cmdline与冻结命令匹配；`artifacts/matrix_manifest.json`由matrix原子创建。首波8个worker已起；其`CUDA_VISIBLE_DEVICES`为`0,1,2,4,3,5,7,6`，各worker的进程内`cuda:0`因此分别映射至GPU0–7。
- `2026-07-23T18:25:xx+08:00`：首波持续运行约96秒，8卡均有约`859MiB`显存与17–25%利用率；当前`row receipt/prediction/score=0/125 / 0/375 / 0/1875`，无row exit、无stderr异常指纹。继续按首个row退出与首波完成门槛检查，不读取性能指标。
- `2026-07-23T18:32:xx+08:00`：首波健康检查触发预注册系统技术失败停止条件。不同row`20-1/seed713104`、`20-1/seed713105`、`20-1/seed713106`均在任何prediction前以同一确定性指纹失败：`DSSCStateError: typed real qKNN/BCRR build failed: qKNN INT8 teacher gate failed`。当时`row receipt/prediction/score=0/125 / 0/375 / 0/1875`；未读取准确率、H或其它性能值。
- `2026-07-23T18:32:xx+08:00`：已先逐一核对launcher PID=`875331`及8个活动子PID的CWD/cmdline均绑定`<run>/source/code`和本run根，再仅向该parent及其子PID发送`TERM`以停止继续派发。复核：run-owned live process=`0`，GPU0–7均`0%/10MiB`，本地SSH/TCP22残留=`0`；未需要升级信号。`launcher.exit`未生成，partial artifacts/logs保留并完成选择性回收。

## 终态

`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`

本run没有完整prediction、score或row receipt闭合，不能进行性能分析、比较或promotable声明。后续修复必须在本地完成、独立复审并以全新run ID重新发布；不得续跑、覆盖或重标此run。

## 失败取证与回收

三条失败均发生在第一个`leo_clear_weak/before`组合、prediction发布前：seed713104在`raw`branch失败，seed713105和713106在`ground`branch失败。完整调用链均为`stage2_svrn_bcr.py:873`→`stage2_dssc...py:1043/1044`→`run...py:476/892/1597/1603`；stderr SHA分别为`21400d583a1d3878e971981a9f1e89d0eec946727ac27cbe17160d91693e0300`、`694752e84586b30fe7766485d6e9841731f25f52d524b25ed91fdedca4f62662`、`694752e84586b30fe7766485d6e9841731f25f52d524b25ed91fdedca4f62662`。

触发条件可证实为`top1_agreement<0.995`或`margin_sign_flip_count!=0`，但失败路径在审计receipt写盘前抛出；partial artifacts中audit/INT8文件和`top1_agreement`字段均为0。因此不能从本run恢复具体审计值，后续必须用全新support-only不可变诊断取得，不能推断或回填。

远端完整库存为557 files，SHA=`334b8377be84194c23f42abf4952f117ab684a604ab439ace2715f986f7c62ad`；本地选择性回收41 files，SHA=`c840156609f0e774fdc2029f0f865255033a27d993b1cde1a5416a1c3a4c20aa`。回收包含matrix日志、25个launcher日志、matrix manifest及3个失败row的16个JSON，逐项SHA匹配；未复制重复runtime或checkpoint。

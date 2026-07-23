# ADV3B02/r2-affine-bcr2完整125发布报告

## 身份与当前状态

- run ID：`adv3b02_ts_drqknn_bcrr_r2_affine_bcr2_full125_166c1afc_20260723_215744`
- candidate：`ADV3B02-TS-DRQKNN-BCRR/r2-affine-bcr2`
- Git commit：`166c1afcf16afe404bc14c4914ca5e08976b729e`
- 创建时间：`2026-07-23T21:57:44+08:00`
- operator：主agent；N607唯一launch owner必须为单一`gpt-5.6-terra high`runner
- 状态：`PREREGISTERED / NOT_LAUNCHED / NO_PERFORMANCE_RESULT`
- parent技术失败：`dssc_zdom_jg_qknn_r4_bcrr_full125_r1f_techfix4_20c1cd0a_20260723_181353`
- parent终态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；不得续跑、复用或覆盖

## 目标、机制与matched四臂

目标是在同一完整125中验证TX抑制`z_dom`类内邻域条件化是否为`z_id`统一qKNN提供真实目标域收益，以及它与BCRR是否产生正协同。`r2-affine-bcr2`只将BCR权重部署格式改为固定两级INT8残差codec；DA、qKNN、BCRR、`omega`、K、fallback和四臂均不变。

|arm|冻结机制|
|---|---|
|`M0`|逐向量仿射INT8`z_id`Student-t qKNN|
|`M_DA`|TX抑制`z_dom`类内权重＋同一`z_id`qKNN|
|`M_OTHER`|基础`z_id`qKNN＋BCRR|
|`M_JOINT`|双qKNN＋BCRR|

主协同量为`I_syn=H(M_JOINT)-H(M_DA)-H(M_OTHER)+H(M0)`。只有完整prediction和同row scorer结果可以形成性能结论。

## 冻结协议与完整矩阵

- `protocol_schema=p2_min_v1`；复用既有`VALIDATED_ONCE`、GEOFF/r8 archive/manifest/parity/coverage，不重复数据验证。
- receivers：`20-1/3-19/7-14/7-7/8-8`。
- seeds：`713102/713103/713104/713105/713106`。
- slices：`(K10,new5)/(K10,new10)/(K10,new20)/(K5,new20)/(K1,new20)`。
- 每job覆盖`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`。
- 完整闭合：125 jobs、375 scene slices、1500 score rows、1000 arm-state prediction artifacts。
- GPU0–7动态队列；8张卡安全可用时每卡1个worker，本run每卡并发上限1，不干预无关任务。
- query逐样本面对全部注册类；query及truth不得进入fit、metric、temperature、fallback、state或健康早停。

## 本地实现、验证与独立review

|文件|用途|SHA256|
|---|---|---|
|`code/cvsrffi/stage2_adv3b02_ts_drqknn_bcrr.py`|双qKNN、BCR2、四臂state与receipt|`92b58bf8280c3f9de4f6ea5f9abd427be0f90e3cf84b9697765ce4ec57155bfe`|
|`code/scripts/run_adv3b02_ts_drqknn_bcrr_125.py`|authority-backed完整125、8GPU动态队列与健康退出|`59323ddc202c5f1f8c3bd43698907874b67fcbb0ef8b083ab43987a3bb65caa7`|
|`tests/test_stage2_adv3b02_ts_drqknn_bcrr.py`|协议、量化、artifact、进程树和健康负例|`e8e736085c72b64b994c823d2307fde93d4ac70171ec8f66d82a0e85455d5fd4`|

- `ssr-gpu`下`py_compile`：PASS。
- 候选＋相邻DSSC回归：`91 passed,1 skipped`；唯一skip为Windows上的POSIX root-grandchild-sentinel。
- `git diff --check`：PASS。
- 最终独立Terra review：`MERGE / P0=0 / P1=0`。
- 真实checkpoint support-only无query smoke：54/54 state PASS；receipt SHA=`8789320f05c29141f1e6f1f0021cd1cf373e6153864b26977b4183d7f825d6e1`。
- smoke中BCR最低top1=`1.0`、any/large flip总数=`0/0`；qKNN最低top1=`0.9961538462`、large flip=`0`；C26 BCR wire=`8424B`；完整state最大=`116755B`。
- smoke只打开support：`query_file_count=0`、`truth_file_count=0`、`query_packages_loaded=false`、`query_rows_used_for_fit=0`。

## 本地发布输入

本地输入根：`E:\type10-7\automation_reports\CV-SincNet\adv3b02_ts_drqknn_bcrr_r2_affine_bcr2_full125_166c1afc_20260723_215744\input`

|输入|SHA256|说明|
|---|---|---|
|`source_166c1afc_rawblob_deflated.zip`|`d01b8e8c7ac01d0e02f639a726cc0845f22fa33dd75c999688170264ba3b01f3`|32,398,039B；3968个safe regular Git blob；路径集合与HEAD精确一致；全量字节不匹配0|
|`somph_method_lock.json`|`0496594db4a82efbbf17ec3d67ebc3fb1f0c7ced41b542a5a0bde3482e704523`|既有冻结package lock|

冻结外部输入：

- checkpoint：`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`；
- sealed runtime：`f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a`；
- Phase1 archive：`dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0`；
- archive manifest：`34213331d20594dceface61680ab0fea8ffc40ee72d7e13c844763c70fef26d4`；
- parity receipt：`b93219c40b79be8ecdf8c0a51d77710d8119f8899331ae7e2518b77adfeac60b`；
- GEOFF/r8 coverage：`c6e25ebeaed32b577e3321e78cd569acff934a7c804d0cb621b26e68f26d0c17`。

## N607冻结路径与正式命令

远端run根必须在创建前为ABSENT：

`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_ts_drqknn_bcrr_r2_affine_bcr2_full125_166c1afc_20260723_215744`

远端只创建`input/`、`source/`和`logs/`；正式detach前`<run>/artifacts`必须为ABSENT，由matrix原子创建。Python为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD为`<run>/source/code`。

正式child command：

```text
PYTHONPATH=<run>/source/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python scripts/run_adv3b02_ts_drqknn_bcrr_125.py matrix --phase1-checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --sealed-runtime /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt --package-method-lock <run>/input/somph_method_lock.json --cache-root /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/cache_matrix --authority-root /home/szu2070436088/2510044040/CV-SincNet/runs/d81_comprehensive_125_v2auth_20260720/authority_controller/authority_final_retry1 --run-root <run>/artifacts --gpu-ids 0,1,2,3,4,5,6,7
```

detach wrapper必须把PID写入`logs/launcher.pid`、child stdout/stderr写入`logs/matrix.stdout.log`和`logs/matrix.stderr.log`，并在自然退出时把真实exit code写入`logs/launcher.exit`。runner必须在报告中回填实际完整wrapper命令。

## 启动前门与健康检查

1.先运行本地只读`tools\n607_ssh_preflight.ps1`，direct N607优先；失败才使用已验证lab bridge。
2.记录GPU0–7现有进程、显存和安全slot；不kill、暂停、迁移或覆盖无关任务。
3.同步恰好两个input，远端SHA逐项匹配；安全解包后验证3968个成员，method/runner/test SHA与本报告一致。
4.远端固定Python执行3文件`py_compile`。
5.正式launch前必须实跑并通过：`python -m pytest tests/test_stage2_adv3b02_ts_drqknn_bcrr.py::test_posix_root_exit_still_cleans_grandchild_and_preserves_unrelated_sentinel -q`。失败即`BLOCKED_PRE_LAUNCH / NO_PERFORMANCE_RESULT`。
6.确认`<run>/artifacts`为ABSENT后只detach一次，不retry、不复用run ID。
7.detach后立即进行健康检查，并以不超过60秒的短连接周期检查PID/PPID/CWD/cmdline/PGID、GPU、row exit、结构化P0、异常指纹及row/prediction/score计数。
8.任一P0协议或安全错误立即停派；至少2个不同row在无prediction时出现同一确定性异常指纹立即停派。只终止核验为本run的进程树并确认清理；不得按准确率、H或其它性能值早停。
9.runner运行期间主agent继续下一模型DA候选的只读研究，不线性占用N607 runner lane。

## 成功、证伪与待回填

技术完成要求125/125 row receipt、375/375 scene slice、1500/1500 score row、1000/1000 prediction artifact及完整matrix汇总。完成后必须报告同row的old-before、old-after、old adaptation gain、seen-new、H、BA、floor、min-old、min-new、forgetting、old→new/new→old、逐类/receiver/scene/K/seed/new-count、DA净正确变化、coverage、INT8、MAC、时延、显存、state bytes和`I_syn`。

|字段|当前值|
|---|---|
|remote PID|`PENDING`|
|launcher exit|`PENDING`|
|parity receipt|`PRESENT_REUSED / NOT_GENERATED`|
|archive/manifest|`PRESENT_REUSED / NOT_GENERATED`|
|coverage|`PRESENT_REUSED / NOT_GENERATED`|
|row/prediction/score|`0/125 / 0/1000 / 0/1500`|
|最终裁决|`PENDING`|

没有完整prediction只能标记`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；完整prediction但性能未达门标记`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。不得把preflight、PID、测试、support fit或量化审计当作性能成功。

## N607 runner执行记录

- `PENDING`。

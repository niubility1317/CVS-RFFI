# ADV3B02/r2-affine-bcr2完整125发布报告（posixfix1）

## 身份与当前状态

- run ID：`adv3b02_ts_drqknn_bcrr_r2_affine_bcr2_full125_posixfix1_ab04f624_20260723_223533`
- candidate：`ADV3B02-TS-DRQKNN-BCRR/r2-affine-bcr2`；科学机制未变，本次仅为发布技术修复`posixfix1`
- Git commit：`ab04f62483826d84c713dda79b7a337c999e9d38`
- 创建时间：`2026-07-23T22:35:33+08:00`
- operator：主agent；N607唯一launch owner为单一`gpt-5.6-terra high`runner
- 状态：`PREREGISTERED / NOT_LAUNCHED / NO_PERFORMANCE_RESULT`
- parent run：`adv3b02_ts_drqknn_bcrr_r2_affine_bcr2_full125_166c1afc_20260723_215744`
- parent终态：`BLOCKED_PRE_LAUNCH / NO_PERFORMANCE_RESULT`；冻结Python无`pytest`，未detach、无PID、无prediction；本run不得复用或覆盖parent

## 目标、假设与matched四臂

目标是在同一完整125中验证TX抑制`z_dom`类内邻域条件化是否为`z_id`统一qKNN提供真实目标域收益，以及它与BCRR是否产生正协同。`posixfix1`只把启动前进程组安全验证改为冻结Python可直接执行的标准库子命令；DA、qKNN、BCRR、BCR2、四臂、K、矩阵和停止条件均未改变。

|arm|冻结机制|
|---|---|
|`M0`|逐向量仿射INT8`z_id`Student-t qKNN|
|`M_DA`|TX抑制`z_dom`类内权重＋同一`z_id`qKNN|
|`M_OTHER`|基础`z_id`qKNN＋BCRR|
|`M_JOINT`|双qKNN＋BCRR|

主协同量为`I_syn=H(M_JOINT)-H(M_DA)-H(M_OTHER)+H(M0)`。只有完整prediction和同row scorer结果可以形成性能结论。

## 冻结协议与完整矩阵

- `protocol_schema=p2_min_v1`；复用既有`VALIDATED_ONCE`、GEOFF/r8 archive/manifest/parity/coverage，不重复数据验证。
- receivers：`20-1/3-19/7-14/7-7/8-8`；seeds：`713102/713103/713104/713105/713106`。
- slices：`(K10,new5)/(K10,new10)/(K10,new20)/(K5,new20)/(K1,new20)`；每job覆盖3个`leo_*_weak`场景。
- 完整闭合：125 jobs、375 scene slices、1500 score rows、1000 arm-state prediction artifacts。
- GPU0–7动态队列；8张卡安全可用时每卡1个worker，本run每卡并发上限1，不干预无关任务。
- query逐样本面对全部注册类；query及truth不得进入fit、metric、temperature、fallback、state或健康早停。

## 本地技术修复、验证与独立review

|文件|用途|Git blob SHA256|
|---|---|---|
|`code/cvsrffi/stage2_adv3b02_ts_drqknn_bcrr.py`|冻结科学方法，未修改|`92b58bf8280c3f9de4f6ea5f9abd427be0f90e3cf84b9697765ce4ec57155bfe`|
|`code/scripts/run_adv3b02_ts_drqknn_bcrr_125.py`|新增无`pytest`的`posix-sentinel`，异常路径封存PGID并隔离清理|`04f4d640ddc827df8ab4e2ffece3a346e4f4851efd5c6efac5109c00e4313c7d`|
|`tests/test_stage2_adv3b02_ts_drqknn_bcrr.py`|同实现测试、捕获失败和清理异常注入负例|`05c3f8f54ae8507748a81ea742e1c5ee02e3658a28a9fa84a32869602796c1c6`|

- `ssr-gpu`下`py_compile`：PASS。
- 候选＋相邻DSSC完整相关回归：`92 passed,3 skipped`；3个skip均为Windows上的真实POSIX专项。
- Windows直接调用`posix-sentinel`：明确非零并报告`requires a POSIX host`。
- `git diff --check`：PASS。
- 首轮独立review：`REVISE / P0=0 / P1=1`，发现`/proc`捕获前异常可能遗漏grandchild PGID且单项清理错误可能中断后续收尾。
- 修复后独立复审：`MERGE / P0=0 / P1=0 / P2=0`；未放宽cmdline、CWD、output-root、PGID或unrelated-process边界。
- 科学方法未变，继续复用commit`166c1afc`的54/54真实checkpoint support-only无query smoke；receipt SHA=`8789320f05c29141f1e6f1f0021cd1cf373e6153864b26977b4183d7f825d6e1`。

## 本地发布输入

本地输入根：`E:\type10-7\automation_reports\CV-SincNet\adv3b02_ts_drqknn_bcrr_r2_affine_bcr2_full125_posixfix1_ab04f624_20260723_223533\input`

|输入|SHA256|说明|
|---|---|---|
|`source_ab04f624_rawblob_deflated.zip`|`4603cca6c8153173615bf68087bda45c5b34cb7bc8a0146be9d01439ff2f1e47`|33,109,259B；3969个safe regular Git blob；path-set SHA=`44e7520c962b161c23f82c963ec850963bd7741af92329508a643506bf186ee0`；全量raw blob复核不匹配0|
|`somph_method_lock.json`|`0496594db4a82efbbf17ec3d67ebc3fb1f0c7ced41b542a5a0bde3482e704523`|既有冻结package lock|

冻结外部输入：checkpoint=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`；sealed runtime=`f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a`；Phase1 archive=`dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0`；manifest=`34213331d20594dceface61680ab0fea8ffc40ee72d7e13c844763c70fef26d4`；parity=`b93219c40b79be8ecdf8c0a51d77710d8119f8899331ae7e2518b77adfeac60b`；coverage=`c6e25ebeaed32b577e3321e78cd569acff934a7c804d0cb621b26e68f26d0c17`。

## N607冻结路径、命令与健康检查

远端run根必须在创建前为ABSENT：

`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_ts_drqknn_bcrr_r2_affine_bcr2_full125_posixfix1_ab04f624_20260723_223533`

Python为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD为`<run>/source/code`。正式detach前`<run>/artifacts`必须为ABSENT。

启动前真实POSIX门：

```text
PYTHONPATH=<run>/source/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python scripts/run_adv3b02_ts_drqknn_bcrr_125.py posix-sentinel
```

正式child command：

```text
PYTHONPATH=<run>/source/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python scripts/run_adv3b02_ts_drqknn_bcrr_125.py matrix --phase1-checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --sealed-runtime /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt --package-method-lock <run>/input/somph_method_lock.json --cache-root /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/cache_matrix --authority-root /home/szu2070436088/2510044040/CV-SincNet/runs/d81_comprehensive_125_v2auth_20260720/authority_controller/authority_final_retry1 --run-root <run>/artifacts --gpu-ids 0,1,2,3,4,5,6,7
```

- direct N607 preflight优先；失败才使用已验证lab bridge。
- 同步恰好两份input；安全解包后验证3969个成员、path set、三个冻结文件SHA和远端`py_compile`。
- POSIX门失败即`BLOCKED_PRE_LAUNCH / NO_PERFORMANCE_RESULT`；通过后确认`artifacts`ABSENT并只detach一次。
- detach后立即健康检查；任一P0立即停派，或至少2个不同row在无prediction时出现同一确定性异常指纹立即停派。只终止已核验为本run的进程树；不得按性能值早停。
- detach wrapper必须记录PID、真实exit、CWD、cmdline和PGID；监控使用不超过60秒的短连接并主动断开。

## 成功标准与待回填

技术完成要求125/125 row receipt、375/375 scene slice、1500/1500 score row和1000/1000 prediction artifact。完成后报告同row的old-before、old-after、old adaptation gain、seen-new、H、BA、floor、min-old、min-new、forgetting、双向混淆、逐类/receiver/scene/K/seed/new-count、DA净正确变化、coverage、INT8、MAC、时延、显存、state bytes及`I_syn`。

|字段|当前值|
|---|---|
|remote PID|`PENDING`|
|launcher exit|`PENDING`|
|parity receipt|`PRESENT_REUSED / NOT_GENERATED`|
|archive/manifest|`PRESENT_REUSED / NOT_GENERATED`|
|coverage|`PRESENT_REUSED / NOT_GENERATED`|
|row/prediction/score|`0/125 / 0/1000 / 0/1500`|
|最终裁决|`PENDING`|

没有完整prediction只能标记`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；完整prediction但性能未达门标记`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。

## N607 runner执行记录

- `PENDING`。

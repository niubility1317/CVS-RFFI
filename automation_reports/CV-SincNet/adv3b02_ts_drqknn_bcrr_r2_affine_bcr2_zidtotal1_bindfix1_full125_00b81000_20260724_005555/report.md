# ADV3B02/r2-affine-bcr2-zidtotal1-bindfix1完整125发布报告

## 身份与当前状态

- run ID：`adv3b02_ts_drqknn_bcrr_r2_affine_bcr2_zidtotal1_bindfix1_full125_00b81000_20260724_005555`
- candidate：`ADV3B02-TS-DRQKNN-BCRR/r2-affine-bcr2-zidtotal1-bindfix1`
- scientific Git commit：`00b810006af0d48d457a1afe2a37d6b10d24a4b9`
- 创建时间：`2026-07-24T00:55:55+08:00`
- operator：主agent；N607唯一launch owner为单一`gpt-5.6-terra high`runner
- 状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
- parent run：`adv3b02_ts_drqknn_bcrr_r2_affine_bcr2_zidtotal1_full125_21ffdabf_20260723_234716`
- parent终态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；本run不得复用、续跑或覆盖parent

## 唯一runner执行记录

- direct preflight通过，GPU0–7空闲；远端run根先确认为`ABSENT`，输入、安全解包、关键源码、`py_compile`和POSIX sentinel均通过。
- 唯一detach：wrapper PID/PGID=`1298387/1298387`，matrix PID=`1298389`；首次health为launched=`8`、failed=`0`、异常指纹=`none`。
- 首波8个row成功，方法越过parent的teacher-binding故障点；随后两个Stage2-C row在prediction前触发同一actual-branch audit故障，协调器健康止损。

## 目标、机制与matched四臂

目标是验证TX抑制`z_dom`类内条件化能否为`z_id`统一qKNN产生真实目标域收益，以及该DA与BCRR是否形成正协同。相对parent的唯一技术delta是修复repair receipt绑定`N(raw)`而actual branch错误绑定`N(N(raw))`的FP32非幂等故障；DA、qKNN、BCRR、INT8 bank、Stage2-C、K、四臂和性能门均不变。

|arm|冻结机制|
|---|---|
|`M0`|逐向量仿射INT8 `z_id` Student-t qKNN|
|`M_DA`|TX抑制`z_dom`类内权重＋同一`z_id`qKNN|
|`M_OTHER`|基础`z_id`qKNN＋BCRR|
|`M_JOINT`|双qKNN＋BCRR|

主协同量为`I_syn=H(M_JOINT)-H(M_DA)-H(M_OTHER)+H(M0)`。只有完整125的prediction和同row scorer结果可以形成性能结论。

## 协议、矩阵与资源

- `DATA_PROTOCOL=PRESENT_REUSED / NOT_REVALIDATED`；继续复用`p2_min_v1`、`VALIDATED_ONCE`和GEOFF/r8既有证据。
- receivers：`20-1/3-19/7-14/7-7/8-8`；seeds：`713102/713103/713104/713105/713106`。
- slices：`(K10,new5)/(K10,new10)/(K10,new20)/(K5,new20)/(K1,new20)`；每job覆盖3个`leo_*_weak`场景。
- 完整基数：125 jobs、375 scene slices、1500 score rows、1000 arm-state prediction artifacts。
- GPU0–7动态队列；每卡本run并发上限1，不干预无关任务。
- query逐样本面对全部注册类；query及truth不得进入fit、metric、temperature、fallback、state或健康早停。
- 125用于本次完整性能验证，不用于调参、选择rank、阈值、量化格式或fallback。

## 本地闭合与独立review

|文件|用途|SHA256|
|---|---|---|
|`code/cvsrffi/stage2_adv3b02_ts_drqknn_bcrr.py`|raw/unit teacher生命周期修复|`e2b6c958be1c9be1b960e5f775aaf95e5b5cafc4bdfdd8e24dc99b28bda625e6`|
|`code/scripts/run_adv3b02_ts_drqknn_bcrr_125.py`|仅launcher revision schema|`e72ad7bb8a3e8230804424a07f010f010fe980aaf9e7cdec9c0d5f826bc0386f`|
|`tests/test_stage2_adv3b02_ts_drqknn_bcrr.py`|非幂等、tamper、决策不变量和Stage2-C|`aa1c3537ba062cc5ae5971a8543ef53259295585c8612745a9c9be4e7b85cfaa`|
|`docs/ADV3B02_TS_DRQKNN_BCRR_R2_AFFINE_BCR2_ZIDTOTAL1_BINDFIX1_DESIGN_FROZEN.md`|冻结合同|`395487b10292db7a9037a00d164cdbff6dc63a95df2f20875440a9209d93632c`|

- `ssr-gpu`下目标测试`72 passed、3 Windows POSIX skipped、0 failed`；相邻DSSC`36 passed、0 failed`；3文件`py_compile`和`git diff --check`均通过。
- K5/K10×无零/单零的bank wire、codes/scales/offsets、两级BCR部署权重和固定query logits与parent逐字节一致。
- 真实checkpoint support-only smoke覆盖两个parent触发row×3 scene，6/6 state与binding通过；query/truth/apply打开0。receipt SHA=`b5e232d48fc07dbb1c744133204265e4b0d6634ef1ff299142bb23180c051474`。
- 独立Terra终审：`MERGE / P0=0 / P1=0 / P2=2`。revision header增加9B且后续可补golden均不阻塞；数组主体、MAC和256KiB门不变。

## 本地发布输入

本地输入根：`E:\type10-7\automation_reports\CV-SincNet\adv3b02_ts_drqknn_bcrr_r2_affine_bcr2_zidtotal1_bindfix1_full125_00b81000_20260724_005555\input`

|输入|SHA256|说明|
|---|---|---|
|`source_00b81000_rawblob_deflated.zip`|`c64da0582288b39f4f4479cbac078a4d3370fdfe73ea4abe572060cfd5c4d9de`|33,141,649B；3975个safe regular Git blob；path-set SHA=`05a30d6eec75b8faf0972ce3cb36d1e19ef35b40c8b2281275cea0ea464e7fd7`；blob manifest SHA=`fb998cd019cc5b9bea8315caaa5e89c2aceb570d12305151c208a2644a524b7e`；raw byte mismatch=0；directory entry=0|
|`somph_method_lock.json`|`0496594db4a82efbbf17ec3d67ebc3fb1f0c7ced41b542a5a0bde3482e704523`|复用冻结package lock|

冻结外部输入：checkpoint=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`；sealed runtime=`f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a`；Phase1 archive=`dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0`；manifest=`34213331d20594dceface61680ab0fea8ffc40ee72d7e13c844763c70fef26d4`；parity=`b93219c40b79be8ecdf8c0a51d77710d8119f8899331ae7e2518b77adfeac60b`；coverage=`c6e25ebeaed32b577e3321e78cd569acff934a7c804d0cb621b26e68f26d0c17`。

## N607冻结路径与命令

远端run根必须在创建前为ABSENT：

`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_ts_drqknn_bcrr_r2_affine_bcr2_zidtotal1_bindfix1_full125_00b81000_20260724_005555`

Python为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD为`<run>/source/code`。只允许创建`input/source/logs`；正式detach前`<run>/artifacts`必须为ABSENT。

启动前POSIX门：

```text
PYTHONPATH=<run>/source/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python scripts/run_adv3b02_ts_drqknn_bcrr_125.py posix-sentinel
```

正式child command：

```text
PYTHONPATH=<run>/source/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python scripts/run_adv3b02_ts_drqknn_bcrr_125.py matrix --phase1-checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --sealed-runtime /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt --package-method-lock <run>/input/somph_method_lock.json --cache-root /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/cache_matrix --authority-root /home/szu2070436088/2510044040/CV-SincNet/runs/d81_comprehensive_125_v2auth_20260720/authority_controller/authority_final_retry1 --run-root <run>/artifacts --gpu-ids 0,1,2,3,4,5,6,7
```

## 唯一runner职责与健康检查

- direct N607 preflight优先；失败才使用已验证lab bridge。先核对GPU、进程、磁盘和远端run根ABSENT。
- 同步恰好两份input；安全解包后验证3975个成员、path set、逐blob、关键源码SHA、外部输入SHA、远端`py_compile`和POSIX sentinel。
- 再次确认`artifacts`ABSENT后只detach一次，记录PID、PGID、CWD、cmdline、GPU映射和真实exit。
- detach后立即健康检查，随后仅用短连接监控。任一P0协议/安全错误，或至少2个不同row在无prediction时出现同一确定性异常指纹，立即停止继续派发并只终止已核验归属本run的进程树；不得自然等待剩余row重复失败。
- 不得按性能值早停、retry、调参、改方法、覆盖输出或重复数据验证。技术完整后回收全部receipt、prediction、score、日志和inventory。

## 预注册性能输出与裁决

技术完整要求125/125 row receipt、375/375 scene slice、1500/1500 score row和1000/1000 prediction artifact。完整后必须同row报告old-before、old-after、old adaptation gain、seen-new、H、BA、floor、min-old、min-new、forgetting、old→new/new→old、逐类/receiver/scene/K/seed/new-count、DA净正确变化、coverage、量化margin、MAC、时延、显存、state bytes和`I_syn`。

立即证伪保持冻结：`M_DA`无净正确增益、`M_OTHER`无独立正收益、`M_JOINT.H<=max(M_DA.H,M_OTHER.H)`、mean `I_syn<=0`、联合臂损害old/seen-new/floor/min-old/min-new或增加forgetting、正协同不足188/375个scene slice或不足2/3个scene均值为正，均不得推广。完成prediction但不达门标记`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`；无完整prediction只能标记`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。

## 执行记录

|字段|当前值|
|---|---|
|Git commit|`00b810006af0d48d457a1afe2a37d6b10d24a4b9`|
|remote PID/PGID|`1298387/1298387`；matrix PID=`1298389`|
|launcher/matrix exit|launcher exit=`1`；wrapper/matrix均已退出|
|parity receipt|`PRESENT_REUSED / NOT_GENERATED`；SHA=`b93219c40b79be8ecdf8c0a51d77710d8119f8899331ae7e2518b77adfeac60b`|
|archive/manifest|`PRESENT_REUSED / NOT_GENERATED`；SHA=`dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0`/`34213331d20594dceface61680ab0fea8ffc40ee72d7e13c844763c70fef26d4`|
|coverage|`PRESENT_REUSED / NOT_GENERATED`；SHA=`c6e25ebeaed32b577e3321e78cd569acff934a7c804d0cb621b26e68f26d0c17`|
|row/prediction/score|submitted/completed=`38/38`，success=`29`，failed/terminated=`9`，never submitted=`87`；partial prediction=`232`，partial score=`348`|
|最终裁决|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`|

## 终态技术故障闭环

|触发row|条件|prediction/query fit|首源|
|---|---|---|---|
|`adv3b02_r2_affine_bcr2_rx_8-8_s_713105_k_10_n_20`|receiver=`8-8`，seed=`713105`，K10/new20|`0/0`|`append_stage2_c -> _append_bank -> _make_actual_branch -> ActualBankBranchState.__post_init__:550`|
|`adv3b02_r2_affine_bcr2_rx_7-14_s_713102_k_10_n_10`|receiver=`7-14`，seed=`713102`，K10/new10|`0/0`|同上|

- 本run进程、GPU compute、本地SSH/TCP22均为0；38条termination receipt均验证归属和进程树退出。
- partial prediction/score严禁读取或形成性能结论。`DATA_PROTOCOL=PRESENT_REUSED / NOT_REVALIDATED`。
- enrollment-only诊断包只含两触发row的before/after三场景support，query/truth/apply/prediction/score条目为0。

## 主agent首源定位与下一revision

真实checkpoint support-only复算确定：一平面affine qKNN在K10/new10的160条after support上有1个小margin翻转，top1=`0.99375`、large flip=`0`；BCR top1=`1.0`、flip=`0`。单独增加residual平面仍失败。固定affine INT8主平面＋INT8 residual平面＋双FP16 class bandwidth后，两个触发row×3场景的qKNN top1=`1.0`、large flip=`0`，BCR top1=`1.0`、flip=`0`，最大logit误差`0.00108–0.00181`；未读取query/truth。

独立监督否决FP32 target sidecar后，对双FP16补偿终裁=`MERGE / P0=0 / P1=0`。下一revision=`ADV3B02-TS-DRQKNN-BCRR/r3-q2f32-bcr2-zidtotal1`，只修改qKNN部署codec；本run不得retry或覆盖。

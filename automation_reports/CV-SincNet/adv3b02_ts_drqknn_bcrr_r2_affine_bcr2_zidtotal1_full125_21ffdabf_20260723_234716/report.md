# ADV3B02/r2-affine-bcr2-zidtotal1完整125发布报告

## 身份与当前状态

- run ID：`adv3b02_ts_drqknn_bcrr_r2_affine_bcr2_zidtotal1_full125_21ffdabf_20260723_234716`
- candidate：`ADV3B02-TS-DRQKNN-BCRR/r2-affine-bcr2-zidtotal1`
- scientific Git commit：`21ffdabf01af01b6cd2cfaf9db96e8b021812a26`
- 创建时间：`2026-07-23T23:47:16+08:00`
- operator：主agent；N607唯一launch owner为单一`gpt-5.6-terra high`runner
- 状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
- parent run：`adv3b02_ts_drqknn_bcrr_r2_affine_bcr2_full125_posixfix1_ab04f624_20260723_223533`
- parent终态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_COMPLETE_PERFORMANCE_RESULT`；本run不得复用、续跑或覆盖parent

## 目标、机制与matched四臂

本run验证TX抑制`z_dom`类内邻域条件化能否为`z_id`统一qKNN产生真实目标域收益，以及该DA与BCRR是否形成正协同。相对parent的唯一delta是冻结规则`finite_exact_zero_singleton_class_medoid_v1`及真实binding闭合；DA、qKNN、BCRR、INT8 codec、K、四臂和完整125均不变。

|arm|冻结机制|
|---|---|
|`M0`|逐向量仿射INT8 `z_id` Student-t qKNN|
|`M_DA`|TX抑制`z_dom`类内权重＋同一`z_id`qKNN|
|`M_OTHER`|基础`z_id`qKNN＋BCRR|
|`M_JOINT`|双qKNN＋BCRR|

主协同量为`I_syn=H(M_JOINT)-H(M_DA)-H(M_OTHER)+H(M0)`。只有完整prediction和同row scorer结果可以形成性能结论。

## 协议、矩阵与输入

- `protocol_schema=p2_min_v1`；复用既有`VALIDATED_ONCE`和GEOFF/r8 archive/manifest/parity/coverage，不重复数据验证。
- receivers：`20-1/3-19/7-14/7-7/8-8`；seeds：`713102/713103/713104/713105/713106`。
- slices：`(K10,new5)/(K10,new10)/(K10,new20)/(K5,new20)/(K1,new20)`；每job覆盖3个`leo_*_weak`场景。
- 完整闭合：125 jobs、375 scene slices、1500 score rows、1000 arm-state prediction artifacts。
- GPU0–7动态队列；每卡本run并发上限1，不干预无关任务。
- query逐样本面对全部注册类；query及truth不得进入fit、metric、temperature、fallback、state或健康早停。

## 本地闭合、Git和独立review

|文件|用途|SHA256|
|---|---|---|
|`code/cvsrffi/stage2_adv3b02_ts_drqknn_bcrr.py`|z_id总化、receipt和runtime绑定|`0bba16cda04974e9724c9f6c3d4d85ae9dfe67679db7787f596a44fa8fe38658`|
|`code/scripts/run_adv3b02_ts_drqknn_bcrr_125.py`|每scene/state一次生成、全链复用和完整125校验|`47af326397b7731695c1eb5cf177e7ce24784dd7f2f34c37b47d7e838f722026`|
|`tests/test_stage2_adv3b02_ts_drqknn_bcrr.py`|规则、绑定、顺序和负例|`5926b8f5dcb72b1d716689cd8e314899d27d74bfe731902dc55a366794507cc8`|
|`docs/ADV3B02_TS_DRQKNN_BCRR_R2_AFFINE_BCR2_ZIDTOTAL1_DESIGN_FROZEN.md`|冻结合同|`f06db0a4fcd7f7bd83abf4a621d299d2f9c13dc56fe590bcfc0ba5b23418c9cd`|

- `ssr-gpu`下候选＋相邻DSSC回归：`103 passed,3 skipped`，主体exit0；`py_compile`和`git diff --check`均PASS。
- 真实checkpoint support-only无query smoke：repair count=`0/1/0`，正常行bitwise不变，teacher/actual-bank/append binding及qKNN/BCR门闭合；receipt SHA=`a2bd0ed6a4c5dc57c906c6a5439fb5b0b118893d00e35f09fb5f33dd8a609cad`。
- 独立Terra终审：`MERGE / P0=0 / P1=0 / P2=1`。唯一P2仅影响非formal默认API路径，不影响正式runner，不得延迟发布。

## 本地发布输入

本地输入根：`E:\type10-7\automation_reports\CV-SincNet\adv3b02_ts_drqknn_bcrr_r2_affine_bcr2_zidtotal1_full125_21ffdabf_20260723_234716\input`

|输入|SHA256|说明|
|---|---|---|
|`source_21ffdabf_rawblob_deflated.zip`|`152a9344f7ba2d13b3e94d613247ae18d9b9243be17db9a02667f8110fa923f4`|32,423,160B；3972个safe regular Git blob；path-set SHA=`99294479ff2c6d2e37daed137ec630af387b53bd2111a934e5b1eebf7185209b`；全量raw blob复核不匹配0|
|`somph_method_lock.json`|`0496594db4a82efbbf17ec3d67ebc3fb1f0c7ced41b542a5a0bde3482e704523`|复用既有冻结package lock|

冻结外部输入：checkpoint=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`；sealed runtime=`f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a`；Phase1 archive=`dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0`；manifest=`34213331d20594dceface61680ab0fea8ffc40ee72d7e13c844763c70fef26d4`；parity=`b93219c40b79be8ecdf8c0a51d77710d8119f8899331ae7e2518b77adfeac60b`；coverage=`c6e25ebeaed32b577e3321e78cd569acff934a7c804d0cb621b26e68f26d0c17`。

## N607冻结路径与命令

远端run根必须在创建前为ABSENT：

`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_ts_drqknn_bcrr_r2_affine_bcr2_zidtotal1_full125_21ffdabf_20260723_234716`

Python为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD为`<run>/source/code`。只允许创建`input/source/logs`；正式detach前`<run>/artifacts`必须为ABSENT。

启动前真实POSIX门：

```text
PYTHONPATH=<run>/source/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python scripts/run_adv3b02_ts_drqknn_bcrr_125.py posix-sentinel
```

正式child command：

```text
PYTHONPATH=<run>/source/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python scripts/run_adv3b02_ts_drqknn_bcrr_125.py matrix --phase1-checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --sealed-runtime /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt --package-method-lock <run>/input/somph_method_lock.json --cache-root /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/cache_matrix --authority-root /home/szu2070436088/2510044040/CV-SincNet/runs/d81_comprehensive_125_v2auth_20260720/authority_controller/authority_final_retry1 --run-root <run>/artifacts --gpu-ids 0,1,2,3,4,5,6,7
```

## 唯一runner职责与健康停止

- direct N607 preflight优先；失败才使用已验证lab bridge。核对GPU/进程/磁盘和run根ABSENT。
- 同步恰好两份input；安全解包后验证3972个成员、path set、关键源码SHA、外部输入SHA和远端`py_compile`。
- 冻结Python执行POSIX sentinel；随后再次确认`artifacts`ABSENT并只detach一次，记录PID、PGID、CWD、cmdline、GPU映射和真实exit。
- 启动后立即健康检查。任一P0协议/安全错误，或至少2个不同row在无prediction时出现相同确定性异常指纹，立即停止继续派发并只终止核验归属本run的进程树；不得自然等待剩余row重复失败，也不得按性能值早停。
- 监控使用短连接并主动断开。未经主agent新revision/新commit/新run授权不得retry、调参、改方法或覆盖输出。
- 技术完整后回收全部receipt/prediction/score/log/inventory；任何缺口都不得进入性能分析。

## 预注册成功标准与终态

技术完成要求125/125 row receipt、375/375 scene slice、1500/1500 score row和1000/1000 prediction artifact。完整后才允许报告同row性能。本run因系统性技术故障未达到完整性要求；六个局部完整row及其score只用于技术诊断，禁止形成性能或协同结论。

|字段|当前值|
|---|---|
|remote PID/PGID|`1255272/1255272`；matrix PID=`1255274`|
|launcher/matrix exit|`1`；matrix completion=`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE`|
|parity receipt|`PRESENT_REUSED / NOT_GENERATED`；SHA=`b93219c40b79be8ecdf8c0a51d77710d8119f8899331ae7e2518b77adfeac60b`|
|archive/manifest|`PRESENT_REUSED / NOT_GENERATED`；SHA=`dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0`/`34213331d20594dceface61680ab0fea8ffc40ee72d7e13c844763c70fef26d4`|
|coverage|`PRESENT_REUSED / NOT_GENERATED`；SHA=`c6e25ebeaed32b577e3321e78cd569acff934a7c804d0cb621b26e68f26d0c17`|
|row/prediction/score|完成`15/125`：成功6、失败9、未提交110；prediction=`48/1000`；score row=`72/1500`|
|最终裁决|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`|

没有完整prediction只能标记`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；完整prediction但性能未达门标记`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。

## N607终态、健康止损与证据回收

- direct preflight、两input远端SHA、安全解包3972个Git blob、path-set、关键源码SHA、checkpoint/runtime SHA、`py_compile`和冻结Python POSIX sentinel全部通过；GPU0–7首波均使用，唯一detach未retry。
- launcher PID/PGID=`1255272/1255272`，matrix PID=`1255274`，真实exit=`1`。两个不同row在0 prediction处出现同一确定性异常后，内置健康门立即停止继续派发并只终止本run。
- 触发row为`rx_20-1/seed713103/K10/new20`和`rx_3-19/seed713105/K10/new20`；均`rc=1`、`query_rows_used_for_fit=0`。首源为`_validate_repaired_support_for_state`抛出`z_id repair/state teacher binding drift`，调用链为`_make_actual_branch -> build_int8_qknn_state -> build_stage2_b_state -> build_four_arm_states -> run_row`。
- submitted/launched/completed=`15/15/15`，成功6、失败9，其中2个触发失败、7个因健康止损为`rc=-15`，未提交110。partial仅有18个scene slice、48份prediction和72个score row，不得做性能分析。
- 15份termination receipt均`ownership_verified=true`且`tree_exit_confirmed=true`，canonical SHA=`81fac374d6a4f2df862cd9fdf00cf17c5f986107d8ccb5afb887c29bccd8127c`；终态无本run进程、GPU compute app为空、本地SSH/TCP22残留为0。
- 最小回收inventory SHA=`f0b6e9d35251b9e7ec563e4b68ad13e05de9481c9251f603c61c62520cb043af`，101个证据文件、3,503,360B；bundle SHA=`aac5db16cc40986b3b6556125f9902f9f468ad0652d20f9f71af382f7f21b3cd`。逐文件SHA/size、ZIP CRC和路径安全全部通过。
- archive、manifest、parity和coverage均复用GEOFF/r8原件，本run未生成新副本、未重复数据验证。根目录权威报告封口SHA=`37ff6b573dd60b50e4ca1633886469738e14a35985d2a0a3d121a29fa27de49c`。
- 报告封口后又以同一唯一runner只读回收两个触发row的before enrollment support-only包：10个payload、1,062,814B，inventory SHA=`e7ac7e500661f9bb485759ac9f5e76bae1bb37253a0cafe46ad6bd52eb1ddf9b`；query/truth/apply/after回收数均为0。该证据只用于本地binding复现，不是数据重验或性能结果。

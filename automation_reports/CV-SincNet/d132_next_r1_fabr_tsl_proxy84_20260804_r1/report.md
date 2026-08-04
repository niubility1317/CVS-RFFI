# D132 NEXT-R1 FABR-TSL proxy84实验报告

## 当前状态

|字段|值|
|---|---|
|experiment ID|`d132_next_r1_fabr_tsl_proxy84_20260804_r1`|
|日期|2026-08-04|
|operator|主agent（Sol/high）；科学研发与复核为Terra/max；冻结后唯一runner为Luna/max|
|状态|`LOCAL_VERIFIED / CODE_REVIEW_PASS_P0_0_P1_0 / NOT_RELEASED / NO_NEW_PERFORMANCE_RESULT`|
|目标|一次必要的84行六臂source-held筛选，同时验证FABR域适应、TSL相对历史D92机制和联合替换|

## 已有真实性能与本轮假设

D130完整proxy84表明CSPAR-2的K5 DA为`ΔH=-0.556pp、总正确数-9`，SRDH-2为零效应；Lite160虽`ΔH=+0.164pp、总正确数+14`，但`A_held_proxy=-0.529pp、F_retained=-1.270pp`，因此没有可晋级正收益。D131只完成393条partial prediction，因5个K1精确并列和2个primary160零行技术退出，没有truth、score或性能结果。

本轮唯一假设是：Phase1-only Fisher锚定的单block rank2功能残差能在不写checkpoint的前提下改善support/query公共表示；同一signed-pre-ReLU160表示上的TSL类对称对角EB头能删除full covariance与role分裂，同时不牺牲retained、held-proxy和最低类。full288/selector、D131补丁链、D62 row splice、CSPAR、SRDH、RDCE及重复125均不进入本轮。

## 冻结前矩阵与比较

|维度|冻结值|
|---|---|
|fold|7个receiver×6个seen-class LOCO=42|
|K|`{1,5}`，K1为K5 support物理样本前缀|
|原子row|84|
|逻辑臂|`R0Q/R0F/R0L/R1Q/R1F/R1L`|
|表示|同一真实forward的signed-pre-ReLU160；不得读取full288/aux|
|K1|F/L逐logit alias Q；任一top tie为`TIE_UNRESOLVED/NO_PERFORMANCE_RESULT`|
|K5主比较|`R1Q-R0Q`、`R0L-R0F`、`R1L-R1F`|
|通过条件|每个主比较均`ΔH_proxy>0`且总正确数增加，并且`ΔA_retained>=0、ΔA_held_proxy>=0、ΔF_retained>=0`|
|失败处置|完整负结果立即关闭，不调参数、不重复矩阵|

## 版本、文件与本地验证

当前Git分支为`codex/next-r1-fabr-tsl-20260804`，设计起点commit为`b989cd0c`。本次先更新：

- `docs/STAGE2_RD_GOAL_20260731.md`：吸收D131复盘及独立复核的2个P0、3个P1；
- `analysis/next_r1_fabr_tsl_traceability_20260803.md`：增加表示、并列、FABR曲率、顺序/量化、TSL API和资源追踪；
- 本报告及根目录同路径镜像。

科学代码、真实D105桥、冻结qKNN/D92-Full160回调、六臂runtime和独立score均已实现。`ssr-gpu`下联合53项聚焦测试通过，`py_compile`和`git diff --check`通过；最终独立复核为`P0=0、P1=0、CODE_REVIEW_PASS`。真实smoke必须固定checkpoint文件SHA并执行两次逐元素一致的真实forward；FABR使用`torch.func.functional_call`，Phase1方向验证精确并列失败关闭；score验证冻结plan、manifest、84个row seal、逐行NPZ及completion四SHA。尚未同步N607，因此仍无性能结果。

|本地文件|SHA256|
|---|---|
|`code/cvsrffi/stage2_next_r1_fabr.py`|`eec1cb95ee7f443fc33220719cb4f43043923c31b7145845dc2fb83065a724fa`|
|`code/cvsrffi/stage2_next_r1_tsl.py`|`965328d68077b8175339c456c34fbc05ddcfbe941f599fab68636b468f2dbc9b`|
|`code/cvsrffi/stage2_next_r1_matrix.py`|`138d34dd7910be827c0305eee56ca41e2d22556101ca6c9917e415991e203805`|
|`code/cvsrffi/stage2_next_r1_assets.py`|`5b7286a1c58abf60fdbb43cad083f663f8e93cdafbcddcad3f113b5341e05389`|
|`code/cvsrffi/stage2_next_r1_runtime.py`|`fd4797de6f4afec195f68fbff3486192e373ab29fc5625b37588885446e3d2aa`|
|`code/cvsrffi/stage2_next_r1_real.py`|`42ec3e87ccfba6d6195d6505d71abac2f0711c34cdf701b8db952ad4cde99216`|
|`code/scripts/run_next_r1_proxy84.py`|`c1724e42e80d66ab6125cf2d11a836f1328a430dce3dfc262431ac8f3dde29e0`|

## 预注册资源与协议检查

- `p2_min_v1`；复用匹配`VALIDATED_ONCE`的既有received-IQ，不重复数据验证；
- query零fit、零update、零selection；不得读取clean/source、query truth、role、quota或global reassignment；
- FABR只封存Phase1-only的INT8 rank2基、FP16 scale、2×2 Fisher几何和冻结常数；Phase2仅support闭式拟合2个系数；
- TSL只接收`support_z160/support_labels/registered_classes`；Phase1先验封存量化误差、正方差、hash和margin-slack；
- 分开保存FABR support/query forward成本与head拟合、wire字节、query MAC、墙钟和瞬时工作集receipt。

## 发布前仅保留的硬门

1.独立差分复核已确认`P0=0、P1=0、DESIGN_FROZEN`；
2.实际Git方法入口与聚焦协议负测通过；
3.真实checkpoint-derived received-IQ no-query smoke通过，并完成一次K1 no-truth liveness scan；
4.不可覆盖run/output路径、本地commit、N607预检和资源记录。

不要求重复数据验证、通用发布平台、额外签名层、论文叙事、D62/D92/SVRN复跑或125矩阵。

## N607发布字段

|字段|值|
|---|---|
|本地文件与hash|见上表；实现提交`1e2a31a4`|
|同步目的地|`/home/szu2070436088/2510044040/CV-SincNet/runs/d132_next_r1_fabr_tsl_proxy84_20260804_r1/source`|
|server command|见下方冻结命令|
|Conda/Python环境|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|CWD|`/home/szu2070436088/2510044040/CV-SincNet/runs/d132_next_r1_fabr_tsl_proxy84_20260804_r1/source`|
|log/output|`logs/predict.log`；`output`必须ABSENT，由runner创建且不可覆盖|
|PID/GPU|待runner落地后记录；每GPU不超过2个训练实验|
|expected artifacts|84行prediction、完整manifest、表示/并列/量化/resource receipt、独立truth score|
|技术停止规则|P0协议/安全错误，或至少2个不同row在prediction前出现相同确定性异常指纹；不得按性能早停|
|fresh retry authority|默认无；若系统性技术失败，保留artifact并由主agent重新冻结新run ID|

冻结输入：checkpoint=`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`，SHA=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`；selected IQ=`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/selected_ls_iq/d106_ls_received_iq.npz`，SHA=`e32708214eaedaf39af532c572e16045f173422d63110e4022778f3ad0252ede`；receipt同目录`d106_ls_received_iq.receipt.json`，SHA=`a18bd5d610c9874bd0d6b50d34e845d85229d5892453bce9ff5bfeaa8ee82d59`；L_s join=`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/input/d104_split/L_s/features.npz`，SHA=`dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d`。

~~~text
CUDA_VISIBLE_DEVICES=<preflight-selected-GPU> PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_next_r1_proxy84.py predict --run-id d132_next_r1_fabr_tsl_proxy84_20260804_r1 --run-root /home/szu2070436088/2510044040/CV-SincNet/runs/d132_next_r1_fabr_tsl_proxy84_20260804_r1/output --checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 --selected-iq /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/selected_ls_iq/d106_ls_received_iq.npz --selected-iq-sha256 e32708214eaedaf39af532c572e16045f173422d63110e4022778f3ad0252ede --selected-receipt /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/selected_ls_iq/d106_ls_received_iq.receipt.json --selected-receipt-sha256 a18bd5d610c9874bd0d6b50d34e845d85229d5892453bce9ff5bfeaa8ee82d59 --ls-join /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/input/d104_split/L_s/features.npz --ls-join-sha256 dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d --device cuda:0 --microbatch-size 8

/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_next_r1_proxy84.py score --run-root /home/szu2070436088/2510044040/CV-SincNet/runs/d132_next_r1_fabr_tsl_proxy84_20260804_r1/output
~~~

predict必须detached启动并立即核验PID/CWD/cmdline/run-root/GPU/log增长；只有`completion.json`为`ARTIFACTS_COMPLETE_NOT_SCORED`且84行manifest、逐行artifact和四SHA全部闭合后，才启动独立score进程。

## 完成后结果表

|candidate|机制|receiver/class/K|A_retained|A_held_proxy|H_proxy|F_retained|总正确数|state/fit/query资源|结论|
|---|---|---|---:|---:|---:|---:|---:|---|---|
|`NEXT-R1 FABR-TSL/r1`|单block rank2 FABR＋类对称TSL|待84行完成|—|—|—|—|—|—|`NO_NEW_PERFORMANCE_RESULT`|

当前推荐：完成唯一候选的设计差分复核后立即实现并发布84行必要矩阵；不启动D92 Lite125。

## N607 runner终态（2026-08-04）

状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / ARTIFACTS_INCOMPLETE / NO_PERFORMANCE_RESULT`。源码由提交`945183e589802cac8de11eb9c4b3a384c376230c`精确归档并落地；archive为52,735,024 bytes、SHA256=`5971ff70c0e74e6f32ed77b678dd1e67da6e6cb7955e49ddbc8f85431f74f1c0`。远端archive SHA、commit marker、5235成员单根`source/`、7个关键源码SHA和`py_compile` 7/7均通过。

唯一detached启动使用物理GPU0，主PID=`995120`，CWD为`run_root/source`，冻结84行命令已写入`control/command.txt`。首个fold在prediction前报确定性`TypeError: expected np.ndarray (got numpy.ndarray)`（`stage2_next_r1_real.py:357 -> 296`），PID随后退出；rows预测artifact为0。`plan.json`和`preregistration.json`已生成；`completion.json`、`manifest.json`、`predictions.npz`、`truth_side.npz`和独立score均未生成，故未启动score。GPU0—7均恢复0%/1MiB，SSH/TCP22清理为0，retry authority=false。

同一远端解释器只读诊断：`torch=2.1.0+cu121`、`numpy=2.2.5`；`torch.from_numpy(np.zeros(...))`复现同一TypeError，`torch.frombuffer(bytearray(16),dtype=torch.float32)`通过。完整回收证据见`artifacts/remote_r1/runner_handoff.md`及同目录日志/控制文件；无性能或晋级结论。

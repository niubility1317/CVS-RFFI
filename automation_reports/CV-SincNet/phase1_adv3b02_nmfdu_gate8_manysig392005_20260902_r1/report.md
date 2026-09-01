# ADV3B02-NMFDU ManySig392005八实验最小预登记

- run ID：`phase1_adv3b02_nmfdu_gate8_manysig392005_20260902_r1`
- 候选矩阵：E1=五分支等权；E2=`I`；E3=`I+D`；E4=`I+D+S`；E5=固定单位系数`I+D+S+U`；E6=可学习全局正系数`I+D+S+U`且无样本校正；E7=完整物理证据+有界样本校正但无null；E8=完整NMFDU。按用户要求不包含ADV3B02对比基线。
- Git分支：`work/adv3b02-nmfdu-gate-v1`
- 实现提交：`df3b350b5e5c6b0ae185eab6c6d62f81de419f39`，已push并独立核对远端分支OID一致
- 命令：`bash code/scripts/launch_phase1_adv3b02_nmfdu_gate8_manysig392005_20260902.sh`
- 环境/CWD：N607普通账户；`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；新release根
- 输入：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`，equalized=`1`
- 数据范围：源RX=`1,3,4,6,8`、day=`1,2,3`；目标RX=`0,2,5,7,9,10,11`、day=`0,1,2,3`；TX=`0–5`；split seed=`392005`
- 源划分：pool=`90000`，`L_s/U_s/V=6300/56700/27000`。用户给出的两个13500验证数量不具有不同方法权限，遵循`项目.md`的单一`V=0.30`协议。
- 目标测试：clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`，每场景168000条；仅测试，不参与训练或checkpoint选择
- 固定训练：train seed=`392005`，epochs=`200`，`lambda_sat_cls=0.68`，`lambda_sat_cons=0`，`best_metric=source_val_sat_hmean`，`checkpoint_selection=final_only`
- 输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_nmfdu_gate8_manysig392005_20260902_r1/{E1,E2,E3,E4,E5,E6,E7,E8}`
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_nmfdu_gate8_manysig392005_20260902_r1/{E1,E2,E3,E4,E5,E6,E7,E8}.out`
- GPU：E1–E8预登记依次使用GPU0–7；启动前检查资源，每GPU训练进程不超过2个
- 停止规则：仅在数据/query越权、错误split/RX/day/seed/场景、输出覆盖、错误checkout、进程归属不清、无prediction闭合或同一确定性系统异常导致合法产物无法产生时停止；低性能不停止
- 预期artifact：八行各自最终checkpoint、训练日志、clean及三种LEO弱场景评估、门控诊断与prediction；prediction完整后由独立scorer连接truth并做同row分析
- 本地验证：关键Python模块`py_compile`通过；NMFDU、数据作用域、launcher、checkpoint兼容等聚焦套件`94 passed`
- 独立P0/P1审查：初审发现E3/E4提前引入可学习系数的P1；已改为E2–E5固定单位系数逐项累加，定点复审`FIXED`，无其他P0/P1
- N607只读preflight：`VERIFIED`；普通账户、项目根和8张RTX3090均可见，未改变远端状态
- 当前状态：`LOCAL_VERIFIED / RELEASE_PREPARING / NOT_LAUNCHED`

## release与N607发布状态

- release提交：`672797082d523cf1b441928fe6632b47ed0aa49b`
- release归档：`E:\type10-7\local_artifacts\releases\adv3b02_nmfdu_gate8_manysig392005_67279708.tar.gz`→`/home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_nmfdu_gate8_manysig392005_67279708.tar.gz`
- release根：`/home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_nmfdu_gate8_manysig392005_67279708`
- 归档SHA对照：本地与远端均为`165093278f1f3463a1c1adad706ab600d2a7d5c2618e98fae6bc888cf94194e9`
- N607验证：原生`bash -n`、远端Python编译和E8完整dry-run通过；dry-run读回E1–E8矩阵、精确RX/day、single V、seed=`392005`及源域选择指标
- 真实checkpoint无query smoke：`PASS`；严格加载、52个NMFDU新state、23组非零梯度，source RX/day/equalized/split seed与预登记一致，query/Phase2访问均为`false`
- 正式启动命令：`env ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_nmfdu_gate8_manysig392005_67279708 WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl RUNS_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_nmfdu_gate8_manysig392005_20260902_r1 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_nmfdu_gate8_manysig392005_20260902_r1 bash /home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_nmfdu_gate8_manysig392005_67279708/code/scripts/launch_phase1_adv3b02_nmfdu_gate8_manysig392005_20260902.sh`
- 资源读回：GPU0–7现有计算进程数依次为`2/3/3/3/2/3/3/2`，没有任何GPU低于启动上限2；未启动、不排队、不干预现有任务
- 当前状态：`RELEASED_READY / WAITING_FOR_GPU_SLOTS / NOT_LAUNCHED`

## 用户授权与启动复核（2026-09-02）

- 用户已明确授权发布实验，并明确要求不设置ADV3B02基线对比。
- 冻结矩阵复核：E1=`equal`、E2=`i_only`、E3=`i_d`、E4=`i_d_s`、E5=`physical_fixed`、E6=`physical_full`、E7=`full_no_null`、E8=`full`；八行均为`physical_gate_variant=nmfdu_v1`，不存在ADV3B02基线行。
- 启动前进程归属复核表明，GPU0–7当前分别承载`2/3/3/3/2/3/3/2`个独立训练实验；现有任务属于MARC-OT与DAOT实验族，并非同一实验的重复子进程。
- 按“每GPU最多2个并发训练实验”的硬约束，本轮未启动任何E1–E8行，未创建run/log根目录，也未停止、重启或修改任何现有任务。
- 当前状态保持：`RELEASED_READY / WAITING_FOR_GPU_SLOTS / NOT_LAUNCHED`。

## 首次启动与系统技术失败（2026-09-02）

- 用户明确授权突破默认并发限制后，本次启动仅将`MAX_ACTIVE_PER_GPU`临时提高为`4`，E1–E8分别绑定GPU0–7；未修改冻结矩阵、数据、seed或训练预算。
- 启动时间：N607服务器时间`2026-09-02 01:02:56 CST`；主PID依次为E1=`3901375`、E2=`3901378`、E3=`3901369`、E4=`3901387`、E5=`3901381`、E6=`3901390`、E7=`3901372`、E8=`3901384`。
- PID、run root、命令行、GPU0–7映射和初始日志均已落地，但8行在正式训练前均出现同一确定性异常：`RuntimeError: "lu_factor_cublas" not implemented for 'Half'`，栈顶定位到`code/cvsrffi/identifiability_stats.py::phase_residual_stats`的`torch.linalg.solve`。
- 该故障满足launcher级系统技术失败停止条件。所有run进程与dispatcher已自然退出，无需终止进程；未触碰其他MARC-OT或DAOT任务，r1目录及日志完整保留。
- r1没有形成prediction或性能结果，不得用于候选比较或性能判断。
- 当前状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；后续仅允许在本地修复、验证并以新release和新run ID重新发布相同冻结实验。

### 本地复现与修复证据

- 本地`ssr-gpu`环境（PyTorch`2.10.0+cu128`、RTX5070Ti）稳定复现同类异常：CUDA autocast将正规方程矩阵乘法降为FP16，`torch.linalg.solve`随后收到Half矩阵。
- 回归测试先分别在`phase_residual_stats`和`effective_fisher_summary`上以相同Half求解指纹失败，再将两处小型线性代数区限定为autocast关闭的FP32计算；门控公式、输入、输出语义和训练矩阵均未改变。
- 修复后两个CUDA autocast回归测试通过；完整NMFDU聚焦套件`96 passed`，关键模块`py_compile`通过，完整小型NMFDU模型的CUDA autocast前向返回有限FP32门控证据。
- 一次独立P0/P1定点审查结论：`PASS`，无P0/P1；审查范围仅限本次两处精度修复及其回归测试。

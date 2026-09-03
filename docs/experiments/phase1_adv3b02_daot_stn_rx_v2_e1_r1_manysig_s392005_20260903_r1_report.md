# ADV3B02-DAOT-STN-RX-V2 E1/R1实验报告

## 1.状态与预登记

- run ID：`phase1_adv3b02_daot_stn_rx_v2_e1_r1_manysig_s392005_20260903_r1`
- 当前状态：`LOCAL_VERIFIED`
- 数据：ManySig equalized，`split_mode=tx_rx_day_1_7_2`，seed=`392005`
- source：RX=`[1,3,4,6,8]`，day=`[1,2,3]`，`L_s/U_s/V=6300/56700/27000`
- target test：RX=`[0,2,5,7,9,10,11]`，day=`[0,1,2,3]`
- 最终评估：`clean`、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`
- 边界：不发布ADV3B02基线，不执行非LEO_WEAK场景，不修改或重启正在运行的P1～P5。

|row|GPU|相对P5的变化|冻结参数|
|---|---:|---|---|
|V2-E1|3|P5+Tensor Temporal Memory+批量Teacher identity-only forward|`λ_subspace=0`，`efficiency_mode=e1`|
|V2-R1|5|E1+选择性nuisance子空间|`λ_subspace=0.05`，rank=8，update interval=5|

基础P5权重固定为：`λ_tangent=0.035`、`λ_route=0.05`、`λ_RX=0.075`、`λ_tail=0.10`、旧`λ_nuisance=λ_fingerprint=0`。

停止规则：仅在数据/query边界错误、错误checkout/run root、输出覆盖、launcher-wide故障、两行出现相同确定性异常、无prediction闭合或scorer连接错误时停止；不得因中间性能低而停止。

预期产物：每行生成`final_ssdg.pth`、`metrics_joint.json`、clean与三个LEO_WEAK逐场景结果、per-RX结果、训练日志、状态文件和阶段2原型导出。

## 2.设计追踪

|ID|来源章节|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|E1-01|实施计划Task10、矩阵V2-E1|Teacher附加视图使用identity-only forward并批量拼接|`code/model_dual_cvsincnet.py`、`code/SSDG/train_ssdg.py`|verified|聚焦测试95项通过|不改变默认P1～P5路径|
|E1-02|实施计划Task10、原报告16/17|使用Tensor Temporal Memory|`code/cvsrffi/orbit_teacher.py`、训练入口|verified|既有单测与P1～P5运行入口|E1复用现有Tensor Bank|
|E1-03|实施计划4.1|E1作为P5的效率增量独立可选|CLI、worker、E1/R1 launcher|implemented|本地发布映射测试通过，待远端dry-run|必须有真实开关，不能只改候选名|
|R1-01|实施计划Task11、矩阵V2-R1|E1基础上启用选择性nuisance子空间|训练入口、E1/R1 launcher|implemented|参数映射测试通过，待真实checkpoint smoke|`λ_subspace=0.05`、rank=8、interval=5|
|R1-02|实施计划Task11|非有限、秩不足时保留有效basis并安全跳过|`code/cvsrffi/selective_nuisance_subspace.py`|verified|非有限输入回归测试通过|不得因高风险模块中断训练|
|PUB-01|最小实验流程|不可覆盖run root、release归档、远端编译/dry-run/启动核验|本报告、release、N607|pending|待发布|只校验一次release归档SHA|

独立P0/P1审查：首次发现R1对秩不足扰动安装任意basis的P1；已定点修复为有效秩不足时保留旧basis并安全跳过，同时将QR异常纳入保护。定点复审结论为`RESOLVED`，未发现其余P0/P1。

## 3.版本与命令

- 本地工作树：`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\adv3b02-daot-stn-v1`
- 远端Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- 基础checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- 远端输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_daot_stn_rx_v2_e1_r1_manysig_s392005_20260903_r1`
- 精确launcher、Git commit、release路径和启动命令在本地验证后补充。

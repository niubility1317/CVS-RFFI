# ADV3B02-ECRS-V1 ManySig八卡直训实验报告

## 1.状态与变更边界

- run_id：`phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_direct8_20260902_r2`
- 当前状态：`LANDED`；等待是否允许多数GPU达到每卡4个训练任务的明确授权
- 正式实验：R1–R8，共8个
- code_commit：`1fb9fe05d9dcaba5cd21e8fed16270d0745e2e72`
- Git分支：`codex/adv3b02-ecrs-v1-parity-fix-20260901`
- 用户覆盖：不运行共享R0；R1–R8分别从随机初始化开始端到端训练
- 设计一致性：ECRS模块、rung递进、loss、数据、seed、epoch和评测保持报告V1配置；仅共享收敛R0前置被用户明确移除
- 声明边界：这是`USER_OVERRIDE_NON_SHARED_BASELINE`近似，不能用R1–R8差值声明严格共享基线下的单机制因果增益

## 2.冻结数据

- 数据集：`ManySig.pkl`，`equalized=1`
- `seed=392005`
- source receivers：`[1,3,4,6,8]`
- source days：`[1,2,3]`
- source pool：90000
- `L_s=6300(0.07)`
- `U_s=56700(0.63)`
- `V=27000(0.30)`
- target receivers：`[0,2,5,7,9,10,11]`
- target days：`[0,1,2,3]`
- target transmitters：`[0,1,2,3,4,5]`
- target：168000样本/场景
- 评测：`clean`、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`
- `V_cal=0.15`与`V_select=0.15`按当前项目协议合并为单一只读`V=0.30`
- source/target receiver集合不相交；`U_s`不包含TX真值训练元数据；Phase1不访问Phase2 query

## 3.八卡矩阵

|GPU|实验|设计改动|
|---:|---|---|
|0|R1|固定Memory Polynomial＋内容估计＋岭回归|
|1|R2|固定样条响应曲面|
|2|R3|R2＋包内split-fit|
|3|R4|R3＋clean/LEO cross-response与surface约束|
|4|R5|R4＋identifiability shrinkage|
|5|R6|R5＋同TX跨receiver响应迁移|
|6|R7|R6＋response auxiliary classifier与不同TX排序|
|7|R8|R7＋受限残差融合gate|

每个候选从随机初始化开始，不加载R0或任何历史checkpoint。每张GPU只启动1个本run实验。用户已授权每张卡在现有任务基础上再增加1个实验，启动时设置`MAX_GPU_TRAIN_PROCS=3`。

## 4.训练与评测冻结项

- `epochs=200`
- `concat_sat_ce_only=true`
- `lambda_sat_cls=0.68`
- `lambda_sat_cons=0`
- 卫星视图日程：`1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak`
- target训练期重评：E200；训练结束后对最终checkpoint执行clean与三种LEO独立评测
- ECRS：`K=28`、8 anchors、`response_dim=64`、`rho_max=0.25`
- 不启用learnable basis；不启用FastTrust
- 单seed机制筛选，不声明多seed稳定性

## 5.本地验证与审查

- direct launcher测试：5项通过
- direct模式冻结：8个候选、0个R0、0个`--init_checkpoint`、R1→GPU0至R8→GPU7
- `train.py`与`dataset_wisig.py`编译：通过
- `git diff --check`：通过
- 真实ADV3B02 checkpoint无query smoke：既有ECRS V1实证继续适用，已验证前向、反向、checkpoint roundtrip与单LEO推理
- 独立P0/P1审查已在ECRS V1实现上完成；项目规则禁止因本次用户明确矩阵变更增加第二次全量审查
- 追踪项`ECRS-24`：implemented；远端启动核对后更新为verified

## 6.发布与启动

- 本地release：`E:\type10-7\local_artifacts\releases\phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_direct8_20260902_r2_1fb9fe05.zip`
- 本地release SHA256：`EF46C2DC889D3D6F72E36A1131393B32F63AB245AD32133DD55060BCD3743A0B`
- 远端归档：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_direct8_20260902_r2_1fb9fe05.zip`
- 远端release根：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_direct8_20260902_r2_1fb9fe05`
- 远端run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_direct8_20260902_r2`
- 远端log根：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_direct8_20260902_r2`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- CWD：远端release根

```bash
env ROOT=<release-root> PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python RUN_ID=phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_direct8_20260902_r2 RUNS_ROOT=<run-root> LOG_ROOT=<log-root> WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl DIRECT_FROM_SCRATCH=1 MAX_GPU_TRAIN_PROCS=3 bash <release-root>/code/scripts/launch_phase1_adv3b02_ecrs_v1_20260901.sh
```

## 7.停止规则与预期产物

仅数据权限或query泄漏、receiver/day/seed/split错误、输出覆盖、错误checkout、确定性执行异常、无checkpoint或最终评测不闭合、进程归属不清允许停止。低性能或中间指标不佳不得停止实验。

每个R1–R8必须产生：`best.pth`、`latest.pth`、训练指标、clean与三种LEO最终指标、ECRS响应/不确定性/融合诊断及独立日志。

## 8.落地状态与资源阻塞

- release归档本地/远端SHA256一致：`ef46c2dc889d3d6f72e36a1131393b32f63ab245ad32133dd55060bcd3743a0b`
- 远端release解压、Python编译和launcher语法检查：`VERIFIED`
- 新run/log输出根仍不存在，R1–R8尚未启动
- 资源变化：在本run发布前，另一个DAOT-STN八卡矩阵新启动；GPU0、1、2、3、5、6当前各有3个训练进程，GPU4、7各有2个
- 用户此前授权的“每卡多1个”对应总上限3；该新增矩阵已占用多数卡的第3个位置
- 未经进一步明确授权，不把多数GPU扩展到每卡4个训练任务；不停止、不迁移、不修改任何外部实验

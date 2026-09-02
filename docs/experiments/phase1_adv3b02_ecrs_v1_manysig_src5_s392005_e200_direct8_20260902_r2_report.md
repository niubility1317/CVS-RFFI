# ADV3B02-ECRS-V1 ManySig八卡直训实验报告

## 1.状态与变更边界

- run_id：`phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_direct8_20260902_r2`
- 当前状态：`RUNNING`；2026-09-02约11:11（Asia/Hong_Kong）按用户即时指令启动
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

每个候选从随机初始化开始，不加载R0或任何历史checkpoint。每张GPU只启动1个本run实验。用户于2026-09-02明确授权本次启动无视显卡训练进程数限制，启动时设置`MAX_GPU_TRAIN_PROCS=999`；该授权不允许停止、迁移、修改或影响外部实验。

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
- 追踪项`ECRS-24`：verified；R1–R8已按无共享R0、随机初始化边界启动

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
env ROOT=<release-root> PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python RUN_ID=phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_direct8_20260902_r2 RUNS_ROOT=<run-root> LOG_ROOT=<log-root> WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl DIRECT_FROM_SCRATCH=1 MAX_GPU_TRAIN_PROCS=999 bash <release-root>/code/scripts/launch_phase1_adv3b02_ecrs_v1_20260901.sh
```

## 7.停止规则与预期产物

仅数据权限或query泄漏、receiver/day/seed/split错误、输出覆盖、错误checkout、确定性执行异常、无checkpoint或最终评测不闭合、进程归属不清允许停止。低性能或中间指标不佳不得停止实验。

每个R1–R8必须产生：`best.pth`、`latest.pth`、训练指标、clean与三种LEO最终指标、ECRS响应/不确定性/融合诊断及独立日志。

## 8.落地与启动状态

- release归档本地/远端SHA256一致：`ef46c2dc889d3d6f72e36a1131393b32f63ab245ad32133dd55060bcd3743a0b`
- 远端release解压、Python编译和launcher语法检查：`VERIFIED`
- 启动前只读核对：release、launcher与ManySig数据存在；新run/log输出根不存在；磁盘可用7.2TB
- 启动前GPU计算进程数：GPU0为2，GPU1–7各为3；用户已明确授权本次启动无视显卡进程数限制
- 用户于2026-09-02明确授权本次启动无视显卡进程数限制；资源slot guard固定为`MAX_GPU_TRAIN_PROCS=999`
- 原一次性自动任务`16点启动ECRS R1-R8`已在手动启动前暂停，独立读回状态为`PAUSED`，防止16:00重复启动
- 资源授权只解除本run的进程数门槛；仍不得停止、迁移、修改或影响任何外部实验

## 9.启动后绑定核验

- 启动命令退出状态：0；launcher明确返回R1–R8共8个PID
- PID/GPU：R1=`4183316`/GPU0，R2=`4183323`/GPU1，R3=`4183330`/GPU2，R4=`4183337`/GPU3，R5=`4183344`/GPU4，R6=`4183351`/GPU5，R7=`4183358`/GPU6，R8=`4183365`/GPU7
- 8个PID均存活；CWD均为冻结release根；cmdline均绑定本run对应R1–R8输出根；`CUDA_VISIBLE_DEVICES`与GPU0–7映射一致
- 8个PID均被对应GPU的compute-app列表读回，每个占用约3.5GB显存；每个主进程有8个数据子进程，CPU时间持续增加
- 8份独立日志均已创建且非空，首次核验大小均为12645字节；未发现`Traceback`、`RuntimeError`、`CUDA out of memory`或`Error:`
- 初始化阶段每份日志出现2次`unsafe backward/step skipped`警告；当前无退出、无确定性异常和无归属错误，按预登记规则继续运行，不因该非终止警告停止

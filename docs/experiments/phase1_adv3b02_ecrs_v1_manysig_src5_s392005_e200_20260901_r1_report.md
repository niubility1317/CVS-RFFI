# ADV3B02-ECRS-V1 ManySig八实验预登记与运行报告

## 1.状态

- run_id：`phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_20260901_r1`
- 当前状态：`LOCAL_VERIFIED`
- 正式实验数：8（R1–R8）
- 共享先决阶段：R0，仅训练一次，不计入八个正式实验
- code_commit：`ca86bf9ecfd288ccbdedf61acc429addc566a38c`
- Git分支：`codex/adv3b02-ecrs-v1-parity-fix-20260901`
- 独立P0/P1审查：`READY`，原checkpoint污染与R0非同池评测问题均已关闭

## 2.冻结数据与协议

- 数据集：`ManySig`
- N607输入：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- `equalized=1`
- `split_mode=tx_rx_day_1_7_2`
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
- 每个target场景：168000样本
- 场景：`clean`、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`
- target aggregate仅包含显式target receivers；source/target receiver集合不相交
- 用户给出的`V_cal=0.15`和`V_select=0.15`按当前`项目.md`合并为单一只读`V=0.30`，不改变27000个验证样本，不建立第二验证集
- `U_s`训练接口中`y=-1`且不携带真实TX训练元数据
- Phase1不访问Phase2 query或其派生状态

## 3.共享R0与八实验矩阵

R0从随机初始化开始，在上述冻结source上训练200epoch；不加载任何历史checkpoint。R0在同一target池完成clean和三种LEO评测。只有R0正常结束且`ADV3B02_ECRS_R0/best.pth`存在，才允许R1–R8启动。

|阶段/实验|GPU|设计改动|验证问题|
|---|---:|---|---|
|R0共享基线|0|收敛ADV3B02；无ECRS；无历史初始化|提供同source、同target评测池的公平共同基线|
|R1|1|固定Memory Polynomial＋内容估计＋岭回归|响应辨识是否有基本价值|
|R2|2|固定样条响应曲面|样条是否优于固定多项式|
|R3|3|R2＋包内split-fit|无标签自监督是否有效|
|R4|4|R3＋clean/LEO cross-response和surface约束|是否提高星地响应稳定性|
|R5|5|R4＋identifiability shrinkage|是否抑制低激励、病态拟合噪声|
|R6|6|R5＋同TX跨receiver响应迁移|是否增强跨接收机泛化|
|R7|7|R6＋response auxiliary classifier和不同TX排序|响应曲面是否具备TX可分性|
|R8|0|R7＋受限残差融合gate|是否能在不破坏主干的前提下提高识别|

R0完成后GPU0释放，再由R8使用；任何时刻每张GPU最多一个本run训练任务。

## 4.冻结训练配置

- R0、R1–R8：`epochs=200`
- R1–R8统一从本run的R0 `best.pth`初始化
- `concat_sat_ce_only=1`
- `lambda_sat_cls=0.68`
- `lambda_sat_cons=0`
- 卫星视图课程：`1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak`
- target重评策略：`interval_final`，E200正式评测；训练结束后对最终checkpoint执行clean和三种LEO独立评测
- ECRS固定：`K=28`、8 anchors、`response_dim=64`、`rho_max=0.25`
- 不启用learnable basis，不启用FastTrust
- R1–R8为单seed机制筛选，不据此宣称多seed稳定性

## 5.本地验证

- ECRS/Meta-SSL/launcher聚焦测试：全部通过
- launcher/数据协议定点集：10项通过
- `train.py`与`dataset_wisig.py`编译：通过
- `git diff --check`：通过
- 真实ADV3B02 checkpoint无query smoke：通过；验证ECRS前向、反向、checkpoint roundtrip和单LEO推理，未读取Phase2 query
- N607只读数据核对：source 90000、`L_s=6300`、`U_s=56700`、`V=27000`；target 168000/场景
- N607现有run只读检索：不存在匹配source receivers `1,3,4,6,8`和days `1,2,3`的可复用收敛ADV3B02 checkpoint

## 6.发布与启动

- 本地release：`E:\type10-7\local_artifacts\releases\phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_20260901_r1_ca86bf9e.zip`
- 本地release SHA256：`95D21FE27B0647F903D386B3B135B6329E5303DD8F2832295D85CE3D97A5E9D0`
- 计划远端归档：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_20260901_r1_ca86bf9e.zip`
- 计划远端release根：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_20260901_r1_ca86bf9e`
- 远端run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_20260901_r1`
- 远端log根：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_20260901_r1`
- 环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- CWD：远端release根
- launcher：`code/scripts/launch_phase1_adv3b02_ecrs_v1_20260901.sh`

计划启动命令：

```bash
nohup env ROOT=<release-root> PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python RUN_ID=phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_20260901_r1 RUNS_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_20260901_r1 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_20260901_r1 WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl bash <release-root>/code/scripts/launch_phase1_adv3b02_ecrs_v1_20260901.sh > <log-root>/pipeline.out 2>&1 &
```

## 7.停止规则与预期产物

仅以下系统技术事实允许停止本run：数据权限或query泄漏；receiver/day/seed/split错误；错误checkout或输出覆盖；确定性启动/运行异常；无checkpoint或无最终评测闭合；进程归属不清。低性能、负收益或中间指标不佳不得停止实验。

预期产物：

- R0：`best.pth`、`latest.pth`、训练指标、clean与三种LEO最终指标
- R1–R8：各自`best.pth`、`latest.pth`、训练指标、clean与三种LEO最终指标、ECRS响应/不确定性/融合诊断
- pipeline日志与每个候选独立日志
- 完成后在本报告追加同row结果、异常、解释和下一候选决策


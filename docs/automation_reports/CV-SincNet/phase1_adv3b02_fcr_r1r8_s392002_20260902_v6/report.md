# ADV3B02-FCR R1-R8八卡并行实验v6预登记报告

## 状态

- run_id：`phase1_adv3b02_fcr_r1r8_s392002_20260902_v6`
- 当前状态：`LOCAL_VERIFIED`
- protocol_scope：Phase1 source-only；R1-R8；seed392002；E200；无R0和旧ADV3B02基线
- branch：`codex/adv3b02-fcr-20260901`
- predecessor：v5在训练前因15域checkpoint与14域训练split被错误要求全模型形状一致而系统性失败，无性能结果，全部partial artifacts保留。

## 冻结矩阵、GPU与初始化

R1→GPU0、R2→GPU1、R3→GPU2、R4→GPU3、R5→GPU4、R6→GPU5、R7→GPU6、R8→GPU7；每卡只新增一个v6 row。R1-R8语义和v4/v5完全不变，训练split仍为day0/1×receiver0-6。

强制初始化：`phase1_adv3b03_core90seed_near3_day123_e200_20260830_r1/S392002_ADV3B03_MU10_ALPHA20_E200/final_ssdg.pth`。固定候选、seed392002、E200，拒绝旧FCR状态；显式使用checkpoint的`branch_ablation=no_dac`、`domain_branch_ablation=no_stats`。成熟`id_backbone.*`必须100%完整加载；允许4个15→14域分类输出层按当前split重建，36个FCR张量为新增初始化。

## 修复验证与审查

- RED复现：v5八行相同`locked mature base checkpoint is incomplete`指纹。
- GREEN：真实checkpoint在14域模型加载191个兼容旧张量，skipped仅4个域输出层，身份主干无缺失，FCR logits有限。
- Phase1-FCR聚焦测试97/97通过；Python编译和`git diff --check`通过。
- 独立P0/P1定点复审仅覆盖本兼容修复，结论`Ready`，无未闭合P0/P1。

## 路径、命令、停止规则与artifact

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_fcr_r1r8_s392002_20260902_v6`
- run：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fcr_r1r8_s392002_20260902_v6`
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_fcr_r1r8_s392002_20260902_v6.Rk.launcher.out`
- 命令：`bash <release>/docs/automation_reports/CV-SincNet/phase1_adv3b02_fcr_r1r8_s392002_20260902_v6/launch_r1r8_remote.sh`

只在数据/query越界、错误row/seed/split/checkpoint、输出覆盖、错误checkout、不能启动、无prediction闭合、进程归属不明或至少两个row出现同一确定性pre-prediction异常时停止本run精确进程树并保留产物；低性能和既有GPU任务数量不触发停止。每行预期生成`best_joint.pth`、`fcr_diagnostics.json`、`fcr_predictions.json`、`train.log`、`status.txt`和四场景prediction。

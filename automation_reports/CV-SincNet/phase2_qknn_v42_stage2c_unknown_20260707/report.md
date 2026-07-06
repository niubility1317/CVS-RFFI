# qKNNV42 Stage2-C互斥unknown优化审计

## 结论

本轮没有得到可部署的qKNNV42提升。此前N20 hardpair闭集seen-new注册结果（K5 old=94.52%、min_old=85.71%、seen_new=90.14%、min_seen=81.43%）不能作为Stage2-C unknown/FAR成功证据；在完整互斥`target_old/target_new/target_unknown`包上，`target_unknown`只做查询评估时，当前qKNN/PCET路线仍表现为known保持与unknown拒识互相挤压，最低类仍为0。

主包几何审计显示，已知查询与真实unknown的max-cosine分离弱：`known_vs_target_unknown_auroc`约0.6006-0.6045；在95%已知召回点，`target_unknown_fpr95_known`约0.945-0.9475。备用完整Stage2-C包也只有约0.6005-0.6115 AUROC，说明失败不是单一NPZ偶然。

## 协议与版本边界

- 控制文件已按顺序读取：`E:\type10-7\AGENTS.md`、`E:\type10-7\项目.md`。
- 根目录`E:\type10-7`不是Git仓库；Git承载面为`E:\type10-7\github_publish\CVS-RFFI-repo`。
- Git状态：`codex/cvs-rffi-release-20260626`相对远端ahead 683；已有未跟踪`local_artifacts/phase2_adv3b02_proxy_mined_20260704/`和`local_artifacts/phase2_adv3b02_smec_ci_20260704/`，本轮未触碰。
- 本轮未访问N607，未启动远端实验，未修改代码、矩阵或服务器状态。
- 协议边界：K=5/K=10少量目标域支持样本用于旧类域适应和新类注册；`target_unknown`不进入支持集或阈值选择，只用于查询评估；接收样本均为LEO星地信道视图。

## 数据与脚本

主完整Stage2-C包：

`E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_frozen_manytx_unknown_diag_20260706\artifacts\features_stage2c_leo_multirx.npz`

备用完整Stage2-C包：

- `E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_adaptive_manynew_20260705\adv3b02_full_stage2c\features_stage2c_leo_multirx_rxscenario.npz`
- `E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_adaptive_manynew_20260705\adv3b02_full_stage2c\features_stage2c_leo_multirx_rx.npz`

脚本：

- `code/scripts/phase2_proxy_target_geometry_audit.py`
- `code/scripts/phase2_orbit_pcet_ci_eval.py`
- `code/scripts/phase2_collaborative_open_set_qknn_eval.py`

## 主包几何审计

|特征包|K|support policy|known accept quantile|known_vs_target_unknown_auroc|target_unknown_fpr95_known|结论|
|---|---:|---|---:|---:|---:|---|
|frozen_manytx|5|stable_first|0.05/0.10|0.6006|0.9475|unknown与known近邻分布严重重叠|
|frozen_manytx|5|scenario_diverse|0.05/0.10|0.6006|0.9475|支持选择不改变分离边界|
|frozen_manytx|10|stable_first|0.05/0.10|0.6045|0.9450|K增加只带来微弱改善|
|frozen_manytx|10|scenario_diverse|0.05/0.10|0.6045|0.9450|支持选择不改变分离边界|

输出目录：

`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_v42_stage2c_unknown_20260707\artifacts`

## PCET完整协议结果

|选择准则|文件|profile|collab|old_acc|min_old|seen_new_acc|min_seen|unknown_reject|unknown_FAR|known_coverage|known_accepted_acc|主要失败|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|最高old|pcet_k10_scenario_diverse.json|pcet_known_preserving|2|0.4023|0.1167|0.1117|0.0000|0.6373|0.2409|0.4079|0.6120|known保持仍低，unknown误接高|
|最高seen-new|pcet_k10_scenario_diverse.json|pcet_known_preserving|4|0.3879|0.1000|0.1221|0.0000|0.7124|0.2228|0.4038|0.6149|seen-new最低类为0|
|最高unknown拒识|pcet_k5_scenario_diverse.json|pcet_unknown_strict|3|0.0258|0.0000|0.0000|0.0000|0.9845|0.0000|0.0121|1.0000|几乎所有known被拒为unknown|

最高old行的类别级最低项：

|角色|最低类|准确率|决策分布|
|---|---|---:|---|
|old|6-15|0.1167|accept 17、unknown_reject 43|
|old|14-10|0.1379|accept 26、request_more 1、unknown_reject 31|
|old|20-19|0.2667|accept 20、defer 6、unknown_reject 34|
|seen_new|1-14|0.0000|accept 10、defer 3、request_more 4、unknown_reject 26|
|seen_new|1-18|0.0000|accept 15、defer 1、request_more 2、unknown_reject 34|
|seen_new|1-8|0.0000|accept 5、defer 2、request_more 2、unknown_reject 43|

这说明坍塌同时来自两类错误：大量known被拒为unknown，以及被accept的seen-new样本标签不正确。单纯放松unknown门限会引入unknown误接，单纯收紧unknown安全会把known保持压到近零。

## 直接qKNN小扫

直接脚本扫描8个配置：K=5/K=10、`risk_margin`/`candidate_set_cvs`、基础qKNN与V42携带参数（`old_bias=0.001`、`prototype_score_blend=0.34`、`mahalanobis_score_blend=0.025`、`source_old_prototype_shrinkage_alpha=0.50`、`teen_blend alpha=0.20`）。

|选择准则|文件|collab|old_acc|min_old|seen_new_acc|min_seen|unknown_reject|known_coverage|known_acc|结论|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|最高known平均|direct_base_k10_candidate_set_cvs.json|1|0.2213|0.0000|0.0000|0.0000|0.9637|0.1296|0.8105|低于PCET保持型|
|最高old|direct_base_k10_candidate_set_cvs.json|1|0.2213|0.0000|0.0000|0.0000|0.9637|0.1296|0.8105|最低旧类仍为0|
|最高seen-new|direct_base_k5_candidate_set_cvs.json|1|0.1605|0.0000|0.0077|0.0000|0.9663|0.1066|0.7468|seen-new几乎不可用|

V42携带参数没有带来正迁移，K10下`v42carry`最高known平均为0.1049，低于基础K10的0.1106。该结果不支持把N20闭集参数直接推广为完整Stage2-C优化。

## 备用包几何审计

|特征包|K|known_vs_target_unknown_auroc|target_unknown_fpr95_known|结论|
|---|---:|---:|---:|---|
|rxscenario|5|0.6008|0.9263|略优于主包但仍不可分|
|rxscenario|10|0.6005|0.9263|K增加无明显改善|
|rx|5|0.5996|0.9300|同级失败|
|rx|10|0.6115|0.9325|仍不足以支撑可靠unknown拒识|

## 判断

本轮应标记为`NON_DEPLOYMENT_DIAGNOSTIC`。在完整Stage2-C互斥unknown协议下，没有产生可称为“qKNNV42继续优化成功”的候选。可保留的结论是：

1. N20 hardpair闭集seen-new注册结果只能说明`target_new`作为已注册新类时的标签传播/近邻识别能力，不能证明真实unknown拒识。
2. 当前Stage2-C特征空间中known与unknown几何重叠过强，后处理门控无法同时解决旧类保持、新类最低类和unknown拒识。
3. 下一步不应继续只调`old_bias`、原型融合或PCET阈值；需要重建互斥Stage2-C优化目标，例如引入源/代理unknown监督的特征训练、LEO视图一致性约束、或在V53/FFT-logmag辅助特征上重新生成同时包含`target_new`和独立`target_unknown`的包。

## 产物

- 主报告：`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_v42_stage2c_unknown_20260707\report.md`
- 主包几何/PCET/direct sweep产物：`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_v42_stage2c_unknown_20260707\artifacts\`
- 小型汇总：
  - `artifacts\pcet_compact_rankings.csv`
  - `artifacts\pcet_joint_failure_analysis.json`
  - `artifacts\direct_sweep\direct_sweep_compact_rankings.csv`
  - `artifacts\direct_sweep\direct_sweep_best_knownavg_details.json`
  - `artifacts\alt_feature_geometry\*.json`


# qKNNV42 OLD80与派生Stage2-C优化审计

## 结论

本轮取得一个正向边界和一个负向边界：

1. 在冻结完整Stage2-C包`features_stage2c_leo_multirx.npz`上，qKNNV42后处理、source old prototype shrinkage、轻量线性头和MLP adapter都无法达到`OLD80_FIRST`；旧类K5/K10上限约73%-74%，最低类约40%-56%。
2. 在已有repair表征`MANYNEW10_NORM_SEP`上，target-old-only K5/K10已达到OLD80：K5 old=92.89%、min_old=80.00%；K10 old=92.62%、min_old=82.86%。这说明旧类域适应不是无解，关键是更换到support-protected/representation-repair特征基底。
3. 但`MANYNEW10_NORM_SEP`原始包不是完整Stage2-C；它没有显式`target_new`角色。把10个非旧TX离线拆成5个seen-new和5个unknown后，完整Stage2-C仍失败：seen-new闭集最高约46%，PCET保持型K5仅old=70.83%、seen_new=6.00%、unknown_reject=66.00%。因此当前下一步应基于NORM_SEP类表征重新导出真实完整Stage2-C特征包，而不是继续调冻结包阈值。

本轮未访问N607，未启动远端实验，未修改代码。

## 协议与版本边界

- 已读取`E:\type10-7\AGENTS.md`和`E:\type10-7\项目.md`。
- `E:\type10-7`根目录不是Git仓库；Git承载面为`E:\type10-7\github_publish\CVS-RFFI-repo`。
- Git基线：`codex/cvs-rffi-release-20260626`，提交前已有未跟踪`local_artifacts/phase2_adv3b02_proxy_mined_20260704/`和`local_artifacts/phase2_adv3b02_smec_ci_20260704/`，本轮未触碰。
- 所有本地诊断均遵守：K=5/K=10少量目标域旧类/新类支持；`target_unknown`只做评估；派生Stage2-C明确标为diagnostic，不作为部署成功。

## 冻结完整Stage2-C包：旧类上限仍不足

输入：

`E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_frozen_manytx_unknown_diag_20260706\artifacts\features_stage2c_leo_multirx.npz`

|诊断|K|最佳old_acc|最佳min_old|说明|
|---|---:|---:|---:|---|
|old-only qKNN，old候选限定|5|73.33%|43.00%|`20-19`最低|
|old-only qKNN，old候选限定|10|73.17%|51.00%|仍低于OLD80|
|per-receiver memory|10|73.50%|40.00%|多接收机混池不是主因|
|source old prototype shrinkage grid|10|73.50%|56.00%|最低类改善但总体不过80|
|support-only ridge/logistic head|5|74.17%|51.00%|轻量head不过80|
|正式linear probe脚本|10|71.42%|45.53%|严格K total per TX split|
|正式MLP adapter脚本|10|69.70%|30.50%|支持集可拟合但泛化不足|

判断：冻结完整Stage2-C包的旧类目标域几何本身不足；继续调`old_bias`、PCET或unknown阈值不会达成目标。

## repair表征：OLD80成立但不是完整Stage2-C

对7个已有repair/supcon/conflict表征包运行`eval_target_old_only_upper_bound.py`，所有训练/阈值/选择均只用`target_old`。

|特征包|K5 old|min_old|K10 old|min_old|边界|
|---|---:|---:|---:|---:|---|
|`MANYNEW10_NORM_SEP`|92.89%|80.00%|92.62%|82.86%|OLD80通过，最佳|
|`MANYNEW10_HEAD_SEP`|92.44%|77.33%|92.62%|81.43%|K10通过|
|`MANYNEW10_SUPCON_NORM`|91.33%|76.00%|91.67%|80.00%|K10通过|
|`MANYNEW10_SUPCON_HEAD`|91.11%|74.67%|91.67%|80.00%|K10通过|
|`MANYNEW10_IDENTITY`|90.22%|76.00%|90.00%|75.71%|old高但floor不足|
|`MANYNEW10_CONFLICT_NORM`|87.56%|74.67%|87.62%|72.86%|不过floor|
|`MANYNEW10_CONFLICT_HEAD`|87.56%|72.00%|87.86%|72.86%|不过floor|

但这些包的角色为`source/target_old/target_unknown/proxy_unknown`，没有显式`target_new`，因此只能证明Stage2-B/OLD80方向有效，不能声明完整Stage2-C。

## repair表征old/unknown边界

用旧类K-shot支持建立old prototype，`target_unknown`仅评估：

|特征包|K|old_acc|min_old|unknown_FAR@support_q05|known_vs_unknown_AUROC|FPR95|
|---|---:|---:|---:|---:|---:|---:|
|`MANYNEW10_NORM_SEP`|5|92.89%|80.00%|85.50%|0.6740|82.75%|
|`MANYNEW10_NORM_SEP`|10|92.62%|82.86%|90.75%|0.6631|80.63%|
|`MANYNEW10_IDENTITY`|5|90.22%|76.00%|78.37%|0.7124|86.88%|
|`MANYNEW10_SUPCON_HEAD`|5|91.11%|74.67%|77.25%|0.6806|84.50%|

解释：NORM_SEP解决了旧类保持，但未知拒识仍弱。后续需要在NORM_SEP类表征上加入真实`target_new`注册和unknown隔离目标，而不是只做阈值后处理。

## 派生Stage2-C诊断

从`MANYNEW10_NORM_SEP`的10个非旧`target_unknown`TX中离线拆分：

- `target_new={10-10,11-10,18-5,19-3,2-13}`
- `target_unknown={2-5,3-8,4-10,8-18,8-3}`

派生包：

`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_v42_old80_probe_20260707\artifacts\derived_stage2c\MANYNEW10_NORM_SEP_derived_stage2c_5new5unknown.npz`

SHA256：`123e20a2bde83e792c90da1203475778a2df70b56aafe52bf60184c320c914a8`

该包仅用于诊断，因为拆分是事后离线构造。

### 几何结果

|K|old闭集top1|seen-new闭集top1|target_unknown accept@support_q05|known_vs_unknown_AUROC|target_unknown FPR95|
|---:|---:|---:|---:|---:|---:|
|5|90.83%|38.00%|94.00%|0.5950|88.00%|
|10|86.67%|44.00%|96.00%|0.6538|85.00%|

### PCET结果

|K|profile|old_acc|min_old|seen_new_acc|min_seen|unknown_reject|结论|
|---:|---|---:|---:|---:|---:|---:|---|
|5|known_preserving|70.83%|0.00%|6.00%|0.00%|66.00%|完整Stage2-C失败|
|10|known_preserving|55.83%|0.00%|0.00%|0.00%|78.00%|完整Stage2-C失败|
|10|unknown_strict|25.00%|0.00%|0.00%|0.00%|91.00%|拒识损伤known|

## 5/5拆分扫描

对10个非旧TX的所有`5 target_new / 5 target_unknown`拆分扫描，作为离线诊断：

- 若要求`old_acc>=0.80`，最高seen_new_acc约46%，但`min_old`最低可掉到20%-25%，unknown FPR95通常0.84-0.94。
- 加`old_bias`后，能把`old_acc`和`min_old`抬高，但seen-new下降；较好折中为K10、old_bias=0.02，`old_acc=85.8%`、`min_old=60.0%`、`seen_new_acc=39.0%`、`min_seen=35.0%`、FPR95=93.0%。
- 按`min(old_acc,seen_new_acc,1-FPR95)`的粗联合分数，最高仍只有0.28左右。

判断：当前非旧类池内存在强old/new/unknown碰撞。NORM_SEP是旧类域适应的正确方向，但还没有解决新类增多后的最低类坍塌和unknown拒识。

## 下一步

1. 用`MANYNEW10_NORM_SEP`或同类support-protected训练目标重新导出真实完整Stage2-C包，导出阶段就显式设置互斥`Y_new/Y_unknown`，不要事后把`target_unknown`改角色。
2. 新导出包必须同时报告target-old-only、Stage2-C闭集old/seen-new、unknown FAR/FPR95/AUROC。
3. 优化重点应从qKNN阈值转向表征：seen-new类间分离、old/new边界保护、unknown moat。qKNNV42只保留为评估/轻量部署头。

## 产物

- 本报告：`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_v42_old80_probe_20260707\report.md`
- 旧类上限：`artifacts\old80_*`
- 正式脚本输出：`artifacts\formal_target_old_scripts\`
- repair表征OLD80：`artifacts\repr_repair_old80\`
- repair表征old/unknown：`artifacts\repr_repair_old_unknown\`
- 派生Stage2-C：`artifacts\derived_stage2c\`


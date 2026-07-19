# D82地面干扰谱稳健中心与Wiener残差收缩

## 实验登记

- 实验ID：`d82_ground_nuisance_wiener_residual_20260720`；登记时间：2026-07-20 05:49 HKT；操作者：Codex。
- 目标：解决D81仅在1/15 outer folds产生变化、rain场景无改善的问题，使地面压缩原型提供样本级域适应先验，同时不压制新类。
- 对照：同一`rx20-1/seed713101/K10/new5/3场景×5fold`上的D62与D81；所有比较使用同一row完整指标。
- 状态：`PREREGISTERED_LOCAL_VERIFIED_QUERY_NOT_OPENED`。

## 单一机制差异与数学锁

D82保留D81的一步Cauchy稳健类中心，并新增一个参数无关的地面干扰方向残差收缩。84个只读int8地面域×类中心先逐类去中心，得到rank由`ceil(participation-ratio effective rank)`唯一决定的地面干扰基`U`和归一化谱`π`。设rank为`r`，固定信号尺度`s=1/r`，第j方向的Wiener保留率为：

```text
retention_j = s / (π_j + s)
z'_i = robust_center_y + [I - U diag(1-retention) U^T](z_i - mean_y)
```

- 无收缩强度、rank或门控扫描；超参数数为0。
- 每个旧类和新类使用完全相同的公式，不读取具体TX/class ID、old/new角色、receiver或scene。
- K≤2逐位恒等；FFT96/RF32不变；query不参与fit、不更新状态、不增加评分计算。
- 地面原型不进入query类分数，也不与target类做身份匹配；只读谱不写回，最终仍编译为单个INT8 affine head。

## 协议、数据与停止条件

- `protocol_schema=p2_min_v1`；复用D18 `VALIDATED_ONCE` capsule，receiver=`20-1`、seed=`713101`、K10（实际K8）、new5；不重建、不重验数据。
- 固定单次`LEO_weak`接收IQ；support-only适配；query逐样本面对全部注册类；无clean/source/query truth/role Oracle/类配额/global assignment/dense query graph。
- ground NPZ SHA256=`3c08c823d2e8a13c4233f0060ac67c332ecc8d6e8abec7352de975fead0267d7`；manifest SHA256=`15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c`；组件仍为`UNVERIFIED`，结果只能是开发诊断证据。
- 成功门：相对D81，B/A/N/H/F/J、三场景、全部逐类和mean-row floors、old→new/new→old/new→new均不回退，且A/H/F、rain或新类至少一项严格改善。任一联合项回退即判负并停止确认seed/125。

## 本地版本与验证

- `E:\type10-7`根目录不是Git仓库；代码和报告在隔离Git worktree`E:\type10-7\code\snapshots\d81wt`开发，之后精确提交并cherry-pick到`E:\type10-7\github_publish\CVS-RFFI-repo`。
- 新增核心：`code/cvsrffi/stage2_d82_ground_nuisance_wiener_residual.py`，SHA256=`8e8870a6a00d99f4ce64f26deb177253bc9a1b12209fe26e5ebe47363db202d1`。
- 新增执行器：`code/scripts/probe_d82_ground_nuisance_wiener_residual.py`，SHA256=`a4c7c79953a6e5a4d180de4416a13e5473b644a24a09eb26e5c46299539f5f1e`。
- `ssr-gpu`环境：D82专项13/13 PASS；D62/D80/D81/D82相邻链48/48 PASS；`py_compile`与`git diff --check`PASS。

## 运行计划与资源

- 本地执行，不占N607 GPU；复用D18 capsule和runtime authorization，输出到本报告目录下`ground_nuisance_wiener_residual/`，启动前必须不存在。
- runtime root：`E:\type10-7\code\snapshots\d41wt`；probe root：`E:\type10-7\code\snapshots\d81wt`。
- capsule：`E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5`。
- ground：`E:\type10-7\automation_reports\CV-SincNet\d22_int8_anchor_lifecycle_20260717\input\int8_component`。
- class binding：`analysis/d19_adv3b02_class_binding_20260717.json`，SHA256=`39cbf3355c221d604eb005624bffc2595cbdb3c499634274103c7663acb9740b`。
- 预期：105-row完整training log、30个target rows、`RECEIPT.json`、D82 metadata；params≤80k、epochs≤30、steps≤50、state≤256KB、query extra MAC=0。
- 已知风险：固定Wiener先验可能过度压缩target身份残差，从而同时降低旧类和新类；该风险只能由锁定query一次评估判断，不能结果后调整系数。

## 完成结果

待运行后补充完整总体、场景、逐类、fold、混淆、量化、训练、资源、机制审计、缺陷和最终结论。

## 启动异常与最小修复

- 封装尝试1在Python启动前失败：嵌套PowerShell变量被外层提前展开；输出目录不存在、query未打开、无性能数据。改为单层PowerShell，实验参数不变。
- 封装尝试2在authority preflight fail-closed：误把`apply_staging_authority.json`传给`--before/after-formal-policy`，其中本地路径字段被runtime正确拒绝；输出目录不存在、query未打开、无性能数据。改用D18的path-free`formal_execution_policy.json`，seal和签名授权不变。
- 尝试3通过authority并进入第一个support-only fit，但在query评分前因`D43 structured covariance is not positive definite`停止；只产生不完整输出，不能报告性能。根因是Wiener残差压缩使block3协方差出现机器舍入量级的非正定漂移。
- 最小修复只对D82的block3协方差启用闭式机器精度SPD修复：`jitter=max(0,d·eps·lambda_max-lambda_min)`；若负能量超过`sqrt(eps)·lambda_max`仍fail-closed。该修复参数数0、不读held/query、不扫描、不改变D81/D62。
- 修复后probe SHA256=`0a3233e602c28f6cd14b2dbe78fb3a1fc73f047bd8011c8387e15b1a822f1c27`；D82专项14/14、D62/D80/D81/D82相邻链49/49 PASS，`py_compile`与`git diff --check`PASS。下一次输出使用`ground_nuisance_wiener_residual_retry1/`，保留失败目录不覆盖。

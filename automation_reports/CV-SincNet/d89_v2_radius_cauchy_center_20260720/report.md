# D89 v2半径可靠度Cauchy support中心报告

## 预注册

- 状态：`PLANNED_LOCAL_DEVELOPMENT_DIAGNOSTIC`。
- 要修复的失败：D87用地面sigma边界换取旧类收益但损伤新类；D88逐类CE硬保护使9/15行零更新并相对D85退化。D89恢复D81已验证的support可靠性路线，同时使用D85的高效v2组件和真实p90半径。
- 数学机制：先从每个ground cell相对其类内跨域均值的偏移得到有效信号`s_dc=||g_dc-mean_d(g_dc)||²`，再把p90余弦半径转为弦长方差`v_dc=2r_dc`，固定可靠度`rho_dc=s_dc/(s_dc+v_dc)`；在每个ground类内部归一化为`q_dc`，以确保6个ground类等权，再汇总84个cell的加权协方差。固定减去manifest中的`reconstruction_rmse²`噪声底，ground类身份随后丢弃，固定保留`ceil(participation ratio)`谱。每个实际target类继续使用D81同一one-step Cauchy权重和中心平移。
- 单一主要差异：相对D81仅替换地面谱来源；target support中心公式、D62 full/block OOF、INT8 head和query路径不变。相对D85不做共享中心，而把radius用于support样本可靠性。
- 协议：v2组件只读；不访问clean/source/query truth/role/quota，不建立ground→target映射；同一received IQ不生成新物理样本；K1/K2恒等；query独立全类单次仿射评分。
- 停止门：相对D81/D85总体和每场景`A/N/H/J/min-class/row-floor`不退化、`F`不升、三类混淆不增加，且至少一项A/H/F/floor/混淆严格改善；INT8/FP32无outer flip和margin sign flip。失败则不进seed2/125且不扫描radius公式。
- 最小矩阵：development seed713101、receiver20-1、K10/new5、3场景×5fold×7候选，共105行。
- 组件限制：`PHASE1_COMPONENT_PENDING_OUTER_JOINT_SEAL`，强制nonpromotable diagnostic。

## 文件与环境

- worktree：`E:\type10-7\code\snapshots\d81wt`。
- core：`code/cvsrffi/stage2_d89_v2_radius_cauchy_center.py`。
- probe：`code/scripts/probe_d89_v2_radius_cauchy_center.py`。
- 环境：`ssr-gpu`；本地开发cell，不使用N607/SSH。
- 输出：`E:\type10-7\automation_reports\CV-SincNet\d89_v2_radius_cauchy_center_20260720\v2_radius_reliability_cauchy_center`。

## 版本、验证与固定谱

- 实现提交：worktree`5346d1cda0d33cbc7de861b9effc7db255d84cd7`；Git承载面cherry-pick提交`d4a0b353`。根目录`E:\type10-7`不是Git仓库，实验报告同时保存在根报告面与Git承载面。
- 变更：新增D89 core、probe及两个聚焦测试；只给D81 probe增加可选`EXTRA_SOURCE_CLOSURE`合并钩子，默认空字典，不改变D81既有公式。
- SHA256：D89 core=`b1b4cd983fa85c7b89d7bd33b2fe83267aae1a1a0f165d02fff028c59cd6ebe2`；D89 probe=`24d3454147d6dbe51677609dad6c7aed36a13107714b90a8458a9efbd46f7752`；D81 scaffold=`7feb50a4210be8f87bc23b1ae9d084436ad09996f2517cbc17bfae13ca14dea4`。
- 2026-07-20 09:09 CST在`ssr-gpu`中执行`py_compile`及D89、D81、D85相邻测试，共19项通过。
- 真实只读v2组件代入：`rho min/mean/max=0.002458/0.189878/0.656824`；每ground类`sum_d(rho)=2.065887..3.012965`；`trace(G89)=0.0011869368`；`reconstruction_rmse²=2.2686737e-6`；正谱rank18、有效rank10.246800、固定保留rank11、保留能量92.0768%。每类domain权重和最大误差`2.22e-16`，加权残差中心最大误差`1.31e-16`。
- 资源预锁：v2持久状态5,816B、目标INT8 affine head 8,583B、总持久状态14,399B；D89额外可训练参数/optimizer step/query MAC/query state均为0；地面重构与谱统计MAC上界不超过0.5M。

## 固定执行命令

本轮为本地development cell，不使用N607、GPU分配、远端PID或远端日志。固定在`E:\type10-7\code\snapshots\d81wt`激活`ssr-gpu`后执行：

```powershell
python code/scripts/probe_d89_v2_radius_cauchy_center.py --d89-arm v2_radius_reliability_cauchy_center --ground-v2-component-dir E:\type10-7\code\snapshots\d81wt\automation_reports\CV-SincNet\d85_ground_radius_v2_20260720\artifacts\component --ground-v2-manifest-sha256 6990ce8c274d6f46c9a86f7272a76960b4a0055e06ae75e14f83e0b70595f112 --expected-checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 --expected-class-handle-binding-sha256 76735ae6d9b2d7e58f683635ca2644e00fbd27a515246aab9d47488c1ab5111f --expected-pre-sign-content-root-sha256 098badd1e82c05c1029cb02c024fe7d3c433488e8ab22e5c6e2ba0516b8d0055 --runtime-root E:\type10-7\code\snapshots\d41wt --probe-root E:\type10-7\code\snapshots\d81wt --before-root E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\before\enrollment_only --before-seal E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\before_enrollment.seal.json --before-seal-sha256 53ace2863c9da6c2f6cc855d602c99f581df6de3d30a9a3ecb89eb6b6f0d9f75 --before-formal-policy E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\formal_execution_policy.json --before-formal-policy-authorization E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\before_formal_policy_authorization.v2.json --before-signed-policy-authorization-envelope E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\before_signed_policy_authorization_envelope.v2.json --before-signed-policy-authorization-envelope-sha256 31a2ad9918f061b25d5a7ed0cc135df70ae02460c094b2f396bf314817bceb0e --after-root E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\after\enrollment_only --after-seal E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\after_enrollment.seal.json --after-seal-sha256 c70aedf3a8f059e756806201758c1933a2f3e1ba4df415e69a1c776b1a2b50ff --after-formal-policy E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\formal_execution_policy.json --after-formal-policy-authorization E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\after_formal_policy_authorization.v2.json --after-signed-policy-authorization-envelope E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\after_signed_policy_authorization_envelope.v2.json --after-signed-policy-authorization-envelope-sha256 a2483d6e9c9c362d89397029ff1e43f48358be3bdb3a05d717ee112b70a0be76 --component-dir E:\type10-7\code\snapshots\d81wt\automation_reports\CV-SincNet\d85_ground_radius_v2_20260720\artifacts\component --component-manifest-sha256 6990ce8c274d6f46c9a86f7272a76960b4a0055e06ae75e14f83e0b70595f112 --class-binding E:\type10-7\github_publish\CVS-RFFI-repo\analysis\d19_adv3b02_class_binding_20260717.json --class-binding-sha256 bb89a1dbb831acb374fccfc596ae98b660b496b449bdca577dabb962121c901f --output E:\type10-7\automation_reports\CV-SincNet\d89_v2_radius_cauchy_center_20260720\v2_radius_reliability_cauchy_center --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

预期输出包括`training_log.jsonl`、`support_audit.json`、`resource_audit.json`、`geometry_audit.json`、`selection.json`、`RECEIPT.json`、`D81_PROBE_METADATA.json`与`D89_PROBE_METADATA.json`。成功执行要求105/105行、目标INT8/FP32各15行、source closure不变、query未打开；性能晋级另按预注册停止门判断。

## 启动记录

- attempt0于2026-07-20 09:16 CST在进入runner、打开数据或创建输出目录前失败：`UnboundLocalError: resource referenced before assignment`。根因是probe读取manifest的`reconstruction_rmse`时，`resource_audit()`赋值位于其后；这是本地集成顺序错误，不构成实验性能证据，输入与公式锁不变。
- 修复提交：worktree`b780d00a46f9a97b90b387dc0c4c0dfe56e0ccc0`，Git承载面`b719b405`；修复后相邻测试15项通过。当前probe SHA256=`4c9fd98f5d9a81eb0116e2637f3248866d58d13eab94b64e761eb1aa16de335c`。
- retry1于2026-07-20 09:18 CST启动，进程总wall time132.1秒、receipt内部计时123.30秒，退出码0。完整产生105/105行、7候选×3场景×5fold，目标INT8/FP32各15行；query未打开、source closure未变、组件入口/出口逐位相同。

## 最终状态

`COMPLETED_DIAGNOSTIC_EFFICIENCY_POSITIVE_PREDICTION_NEUTRAL_NOT_PROMOTABLE`

D89完整复现D81/D85的性能和15/15个outer预测，同时显著降低地面组件状态与谱计算；但相对D81/D85没有任何严格联合性能改善，因此未通过预注册晋级门，不运行独立seed或125。v2组件仍为`PHASE1_COMPONENT_PENDING_OUTER_JOINT_SEAL`且`formal_phase2_eligible=false`。

## 七候选总体性能

数值均为%；`B/A/N/H/F/J`依次为注册前旧类、注册后旧类、seen-new、同row调和均值、遗忘和joint floor。

|candidate|机制|B|A|N|H|F|J|min B/A/N|row floor B/A/N|o→n/n→o/n→wrong-n|判定|
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
|D42-USLDA-INT8|D89 v2 SNR半径谱＋D81 Cauchy中心|92.78|82.78|84.67|82.94|10.00|26.67|80.00/53.33/73.33|73.33/50.00/46.67|22/8/15|主候选，效率正、性能中性|
|D42-USLDA-FP32-MATCHED|D89 FP32 matched|92.78|82.78|84.67|82.94|10.00|26.67|80.00/53.33/73.33|73.33/50.00/46.67|22/8/15|与INT8同预测|
|B3_SINGLE_IQ_DIAG_FFTRF|B3诊断比较器|87.78|75.56|72.67|73.35|12.22|23.33|80.00/60.00/40.00|53.33/33.33/36.67|33/22/19|弱基线|
|D42-D40-HNBR-INT8-NEGATIVE|HNBR负对照|85.56|85.00|15.33|25.16|0.56|0|66.67/63.33/0|40.00/40.00/0|2/0/0|新类不可达|
|D42-D41-BEC-INT8-NEGATIVE|BEC负对照|86.11|20.56|78.67|31.50|65.56|0|76.67/0/36.67|46.67/0/26.67|142/0/32|旧类崩溃|
|D42-PROTOnet-CDA-ZID160|ProtoNet CDA|71.11|48.33|52.67|48.97|22.78|0|33.33/13.33/3.33|13.33/0/0|0/0/0|弱基线|
|Z0_SUPPORT_ONLY|identity control|71.11|48.33|52.67|48.97|22.78|0|33.33/13.33/3.33|13.33/0/0|0/0/0|选择回退|

## matched差异与项目目标差距

|比较|ΔA|ΔN|ΔH|ΔF|ΔJ|变化row|Δo→n/n→o/n→wrong-n|
|---|---:|---:|---:|---:|---:|---:|---|
|D89−D81|0|0|0|0|0|0/15|0/0/0|
|D89−D85|0|0|0|0|0|0/15|0/0/0|
|D89−D62|+0.56|0|+0.31|−0.56|0|1/15|−1/0/0|
|D89−D88|+0.56|0|+0.31|−0.56|0|1/15|−1/0/0|

D89相对D62只在`low-elev/fold0`纠正1个旧类样本，`A=66.67%→75.00%`，其余14/15行不变；这正是D81已有的稀疏收益，不是D89新增收益。相对项目K10/new5目标，`A`尚差9.22pp，最弱旧类53.33%尚差34.67pp，`N`尚差7.33pp。

## 三场景性能

|场景|B/A/N/H/F/J|min B/A/N|row floor B/A/N|o→n/n→o/n→wrong-n|
|---|---|---|---|---|
|clear|98.33/91.67/98.00/94.44/6.67/50.00|90/70/90|90/60/90|2/1/0|
|low-elev|91.67/80.00/76.00/76.92/11.67/20.00|80/60/50|70/60/20|7/5/7|
|rain|88.33/76.67/80.00/77.45/11.67/10.00|60/30/70|60/30/30|13/2/8|

## 15个outer row

|scene/fold|B/A/N/H/F/J|floor B/A/N|o→n/n→o/n→wrong-n|
|---|---|---|---|
|clear/0|100/100/90/94.74/0/50|100/100/50|0/1/0|
|clear/1|100/83.33/100/90.91/16.67/0|100/0/100|0/0/0|
|clear/2|91.67/83.33/100/90.91/8.33/50|50/50/100|1/0/0|
|clear/3|100/100/100/100/0/100|100/100/100|0/0/0|
|clear/4|100/91.67/100/95.65/8.33/50|100/50/100|1/0/0|
|low/0|100/75/80/77.42/25/50|100/50/50|3/1/1|
|low/1|83.33/58.33/70/63.64/25/0|50/50/0|1/0/3|
|low/2|83.33/91.67/70/79.38/−8.33/0|50/50/0|0/2/1|
|low/3|100/100/70/82.35/0/0|100/100/0|0/1/2|
|low/4|91.67/75/90/81.82/16.67/50|50/50/50|3/1/0|
|rain/0|83.33/83.33/60/69.77/0/0|50/50/0|2/0/4|
|rain/1|100/66.67/90/76.60/33.33/0|100/0/50|4/1/0|
|rain/2|91.67/83.33/80/81.63/8.33/50|50/50/50|1/0/2|
|rain/3|83.33/75/90/81.82/8.33/0|50/0/50|3/0/1|
|rain/4|83.33/75/80/77.42/8.33/0|50/50/0|3/1/1|

## 逐类性能

|TX|角色|B|D89 A/N|遗忘或缺陷|
|---|---|---:|---:|---|
|14-10|旧|96.67|93.33|遗忘3.34pp|
|14-7|旧|80.00|53.33|最弱旧类，遗忘26.67pp|
|20-15|旧|96.67|90.00|遗忘6.67pp|
|20-19|旧|93.33|93.33|无遗忘|
|6-15|旧|93.33|73.33|遗忘20.00pp|
|8-20|旧|96.67|93.33|遗忘3.34pp|
|1-16|新|—|93.33|达到单类92%|
|1-18|新|—|73.33|最弱新类|
|18-10|新|—|90.00|仍低于92%|
|14-11|新|—|76.67|第二弱新类|
|8-3|新|—|90.00|仍低于92%|

## 机制表现与根因

- D89真实使用了全部14个domain×6个ground类中心和84个p90半径，并非“没有使用地面原型”。可靠度`rho`范围`0.002458..0.656824`、均值`0.189878`；每类先在domain轴归一化，保证6个ground类等权。地面谱正rank18、有效rank10.2468、保留rank11和92.08%能量。
- target端2160次support中心变换全部执行；final fit每类中心位移L2总体均值0.03223、最大0.06620，归一化权重范围0.02014..0.20612，有效样本数范围6.962..7.867。类内残差误差最大`2.78e-17`，FFT96/RF32误差严格为0。
- D89与D81内部并非数值相同：final support归一化Cauchy权重绝对差均值0.01033、P95 0.03050、最大0.06439；归一化能量相关系数均值0.91577，最低0.45209；中心位移范数差均值0.002686、最大0.01896。但这些变化没有跨过任何outer决策边界，因此15/15预测仍相同。
- 关键瓶颈是D81公式对每个target类用自身平均能量归一化，消除了谱的全局尺度；D89半径主要重排了dominant nuisance方向，却仍只通过一次公共中心平移间接影响LDA。K8下权重ESS仍接近7.3/8，变化温和；对14-7、6-15、1-18、14-11这些弱类的决策margin不足以产生新增纠错。
- 这一路的正收益主要是压缩和计算，而不是准确率：它证明压缩地面原型能够无损替代D81 v1地面谱，但单纯改进ground谱估计已进入离散决策平台期。下一版本不应扫描radius/rank，也不应回到D87/D88的head侧sigma或硬CE门。

## 完整训练记录

- 解析完整15个目标INT8 row，每row完整20步，共300条Stage2-B记录；没有截取tail或抽样。loss min/mean/max=`0.07560/0.30719/1.11738`，CE=`0.07541/0.30708/1.11738`，support accuracy=`89.58/98.77/100%`。
- D89不增加optimizer step；总step20、epoch20。before/final D62 active fit分别7/15和3/15，接受的row component数分别20和6。
- 完整解析metadata、receipt、support、resource、geometry和selection，所有持久JSON非有限数值计数均为0。前台执行的stdout/stderr未另行持久化，这是本轮观测限制；退出码0及receipt闭合了执行状态。

## 量化、资源与效率

|项目|D89结果|判定|
|---|---:|---|
|INT8/FP32 before outer argmax变化|0|通过|
|INT8/FP32 final outer argmax变化|0|通过|
|before/final support argmax变化|0/0|通过|
|margin sign flip|0|通过|
|最大score绝对误差|0.001509|低于预锁0.002|
|ground/affine/总持久状态|5,816/8,583/14,399B|通过|
|ground transient FP64|68,264B|不持久化|
|trainable parameters|2,016；D89额外0|通过|
|optimizer steps|20；D89额外0|通过|
|ground谱统计MAC|377,312|低于0.5M|
|support中心MAC|17,647,360|通过|
|D89新增适配MAC|18,024,672|通过|
|总适配MAC|24,909,248,642|LDA闭包占主导|
|query MAC/D89额外query MAC|6,624/0|通过|
|peak CUDA|22,886,912B|通过|
|dense query graph|0B|通过|

相对D81，ground状态压缩77.13%、总持久状态压缩57.66%、ground谱统计MAC下降99.58%、support中心MAC下降19.38%、D89新增适配MAC下降83.97%；由于共同LDA闭包占约25G MAC，总适配MAC仅下降0.38%。

## 证据与停止决定

- training log SHA256=`5c18e195e6c6ee85230d32e2fd79c7a8265a0a76d5d316d4e2a2efdc30c083ae`。
- receipt SHA256=`062e5d98a56175c5ad6fba885392b6ed8fb560e5a346e09bfd6a4d36be43be2c`。
- D89 metadata SHA256=`cc5025c7cb6ce566f513abbf43df576fe62d7cd0bd1d0576b0b137027a0453e1`。
- 完整性能汇总：`E:\type10-7\automation_reports\CV-SincNet\d89_v2_radius_cauchy_center_20260720\d89_full_performance_summary.json`，SHA256=`42f9d57754063aa7edf2778f5e99a10f875517bb9638ab6dfa67f851837adb78`。
- 晋级门失败原因：相对D81/D85严格改善项为0，变化row为0/15；同时未满足项目绝对目标，组件未联合封存。决定：不运行seed713102、不运行125、不把D89声明为当前最强性能版本；仅保留为目前最强的D81性能等价高效压缩实现证据。

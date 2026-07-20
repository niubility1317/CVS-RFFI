# D90 v2逐方向Cauchy support中心报告

## 预注册

- 状态：`PLANNED_LOCAL_DEVELOPMENT_DIAGNOSTIC`。
- 要修复的D89缺陷：D89相对D81/D85内部support权重明显变化，但K8有效样本数仍约7.3/8，15/15 outer预测不变；单个ground谱总能量只产生每样本一个标量权重，某一坏方向会连带降低该样本全部160维信息。
- 方法：完全复用D89固定v2 SNR半径谱。在每个target类中先得到D81径向中心；保留其ground子空间正交分量，但在保留的11个ground方向内分别计算无扫描Cauchy中心，以逐方向中心替换D81径向子空间中心。最后仍只对z160做一次类公共平移。
- 创新点：压缩ground原型不再仅产生样本级可靠度，而成为“方向级异常解释器”；一个方向的异常不会丢弃同一样本在其他方向的信息。
- 协议：公式对全部注册类相同；K1/K2逐位恒等；类内残差、FFT96、RF32不变；不读取clean/source/query truth/role/quota，不做ground→target类映射，不更新ground组件，不增加query计算。
- 资源预期：仍为5,816B ground＋8,583B affine＝14,399B；额外参数、optimizer step、query MAC/state均为0。逐方向权重只增加`O(CKr)`support标量运算，远低于LDA闭包。
- 停止门：相对D81/D85/D89总体、场景、逐类和15-row floor不退化、混淆不增加，且至少2/15个row变化、净纠错≥2、旧类和新类各自无correct→wrong；否则不进seed2/125，不扫描方向权重或混合系数。
- 固定cell：receiver20-1、seed713101、K10/new5、3场景×5fold×7候选，共105行；v2组件仍pending joint seal，强制nonpromotable。

## 文件与环境

- worktree：`E:\type10-7\code\snapshots\d81wt`。
- core：`code/cvsrffi/stage2_d90_v2_directionwise_cauchy_center.py`。
- probe：`code/scripts/probe_d90_v2_directionwise_cauchy_center.py`。
- 复用：D89谱、D81/D62 full/block OOF和单一INT8 affine query head。
- 环境：`ssr-gpu`；本地开发cell，不使用N607/SSH。
- 输出：`E:\type10-7\automation_reports\CV-SincNet\d90_v2_directionwise_cauchy_center_20260720\v2_directionwise_cauchy_center`。

## 版本与验证

- 实现提交：worktree`f33c3d8f27f24cdfc16094fc22afb0159bbd196c`，Git承载面`59a2c3b4`。
- SHA256：D90 core=`11138760634ad16e9b4b8b2fad8371f624866d342692cc3e8b4a6fd491fae1eb`；D90 probe=`cddcbae1f051afc547b2dfa32714bc996c0e9d387bb18bd8c62297b1389ceef1`；D89 scaffold=`b1b8e7c0191ddf930d1e796997dfa6ae683a6e073b147be3ac563cd73637db8e`；D81 scaffold=`68d4cd676094924112ee30184b9c544f865dba9dc3147a253b9bc467b6da64ca`。
- `ssr-gpu`中D90/D89/D81聚焦与回归测试20项通过，`git diff --check`通过。
- source closure包含D90 probe/core、D89 probe/core、v2 codec、D85/D81 scaffold及D62以下既有闭包。D81/D89的新增hook默认值为0或空字典，不改变其既有公式。

## 固定执行命令

```powershell
python code/scripts/probe_d90_v2_directionwise_cauchy_center.py --d90-arm v2_directionwise_cauchy_center --ground-v2-component-dir E:\type10-7\code\snapshots\d81wt\automation_reports\CV-SincNet\d85_ground_radius_v2_20260720\artifacts\component --ground-v2-manifest-sha256 6990ce8c274d6f46c9a86f7272a76960b4a0055e06ae75e14f83e0b70595f112 --expected-checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 --expected-class-handle-binding-sha256 76735ae6d9b2d7e58f683635ca2644e00fbd27a515246aab9d47488c1ab5111f --expected-pre-sign-content-root-sha256 098badd1e82c05c1029cb02c024fe7d3c433488e8ab22e5c6e2ba0516b8d0055 --runtime-root E:\type10-7\code\snapshots\d41wt --probe-root E:\type10-7\code\snapshots\d81wt --before-root E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\before\enrollment_only --before-seal E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\before_enrollment.seal.json --before-seal-sha256 53ace2863c9da6c2f6cc855d602c99f581df6de3d30a9a3ecb89eb6b6f0d9f75 --before-formal-policy E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\formal_execution_policy.json --before-formal-policy-authorization E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\before_formal_policy_authorization.v2.json --before-signed-policy-authorization-envelope E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\before_signed_policy_authorization_envelope.v2.json --before-signed-policy-authorization-envelope-sha256 31a2ad9918f061b25d5a7ed0cc135df70ae02460c094b2f396bf314817bceb0e --after-root E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\after\enrollment_only --after-seal E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\after_enrollment.seal.json --after-seal-sha256 c70aedf3a8f059e756806201758c1933a2f3e1ba4df415e69a1c776b1a2b50ff --after-formal-policy E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\formal_execution_policy.json --after-formal-policy-authorization E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\after_formal_policy_authorization.v2.json --after-signed-policy-authorization-envelope E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\after_signed_policy_authorization_envelope.v2.json --after-signed-policy-authorization-envelope-sha256 a2483d6e9c9c362d89397029ff1e43f48358be3bdb3a05d717ee112b70a0be76 --component-dir E:\type10-7\code\snapshots\d81wt\automation_reports\CV-SincNet\d85_ground_radius_v2_20260720\artifacts\component --component-manifest-sha256 6990ce8c274d6f46c9a86f7272a76960b4a0055e06ae75e14f83e0b70595f112 --class-binding E:\type10-7\github_publish\CVS-RFFI-repo\analysis\d19_adv3b02_class_binding_20260717.json --class-binding-sha256 bb89a1dbb831acb374fccfc596ae98b660b496b449bdca577dabb962121c901f --output E:\type10-7\automation_reports\CV-SincNet\d90_v2_directionwise_cauchy_center_20260720\v2_directionwise_cauchy_center --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

成功执行要求105/105行、`D90_PROBE_METADATA.json`闭合、query未打开、组件hash不变。性能晋级另按预注册停止门判断。

## 最终状态

`COMPLETED_DIAGNOSTIC_DIRECTIONWISE_ACTIVE_PREDICTION_NEUTRAL_NOT_PROMOTABLE`

2026-07-20 09:38 CST启动，wall time133.2秒、receipt内部124.07秒，退出码0；完成105/105行、7候选×3场景×5fold，目标INT8/FP32各15行。D90逐方向机制真实激活，但相对D89/D81/D85仍为0/15 outer变化，未通过至少2行变化和严格联合改善门，不运行seed2或125。

## 七候选总体性能

|candidate|机制|B|A|N|H|F|J|min B/A/N|row floor B/A/N|o→n/n→o/n→wrong-n|
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
|D42-USLDA-INT8|D90逐方向Cauchy中心|92.78|82.78|84.67|82.94|10.00|26.67|80/53.33/73.33|73.33/50/46.67|22/8/15|
|D42-USLDA-FP32-MATCHED|FP32 matched|92.78|82.78|84.67|82.94|10.00|26.67|80/53.33/73.33|73.33/50/46.67|22/8/15|
|B3_SINGLE_IQ_DIAG_FFTRF|B3诊断比较器|87.78|75.56|72.67|73.35|12.22|23.33|80/60/40|53.33/33.33/36.67|33/22/19|
|D42-D40-HNBR-INT8-NEGATIVE|HNBR负对照|85.56|85.00|15.33|25.16|0.56|0|66.67/63.33/0|40/40/0|2/0/0|
|D42-D41-BEC-INT8-NEGATIVE|BEC负对照|86.11|20.56|78.67|31.50|65.56|0|76.67/0/36.67|46.67/0/26.67|142/0/32|
|D42-PROTOnet-CDA-ZID160|ProtoNet CDA|71.11|48.33|52.67|48.97|22.78|0|33.33/13.33/3.33|13.33/0/0|0/0/0|
|Z0_SUPPORT_ONLY|identity control|71.11|48.33|52.67|48.97|22.78|0|33.33/13.33/3.33|13.33/0/0|0/0/0|

## matched差异

|比较|ΔA|ΔN|ΔH|ΔF|ΔJ|变化row|混淆变化|
|---|---:|---:|---:|---:|---:|---:|---|
|D90−D89|0|0|0|0|0|0/15|0/0/0|
|D90−D81|0|0|0|0|0|0/15|0/0/0|
|D90−D85|0|0|0|0|0|0/15|0/0/0|
|D90−D62|+0.56|0|+0.31|−0.56|0|1/15|−1/0/0|

相对项目目标仍差：`A`9.22pp、最弱旧类34.67pp、`N`7.33pp。D90没有扩大D81相对D62仅`low-elev/fold0`的1个旧类纠错。

## 三场景性能

|场景|B/A/N/H/F/J|min B/A/N|row floor B/A/N|o→n/n→o/n→wrong-n|
|---|---|---|---|---|
|clear|98.33/91.67/98/94.44/6.67/50|90/70/90|90/60/90|2/1/0|
|low-elev|91.67/80/76/76.92/11.67/20|80/60/50|70/60/20|7/5/7|
|rain|88.33/76.67/80/77.45/11.67/10|60/30/70|60/30/30|13/2/8|

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

|TX|角色|B|D90 A/N|缺陷|
|---|---|---:|---:|---|
|14-10|旧|96.67|93.33|遗忘3.34pp|
|14-7|旧|80|53.33|最弱旧类，遗忘26.67pp|
|20-15|旧|96.67|90|遗忘6.67pp|
|20-19|旧|93.33|93.33|无遗忘|
|6-15|旧|93.33|73.33|遗忘20pp|
|8-20|旧|96.67|93.33|遗忘3.34pp|
|1-16|新|—|93.33|达到92%|
|1-18|新|—|73.33|最弱新类|
|18-10|新|—|90|低于92%|
|14-11|新|—|76.67|第二弱新类|
|8-3|新|—|90|低于92%|

## 机制表现与缺陷

- 方向机制不是空操作：final axis Cauchy权重范围`0.12501..1.00000`、均值`0.65346`；每fit最弱方向ESS均值6.5869、范围6.5364..6.6476。D90用逐方向中心替换D89径向子空间中心的L2均值0.005616、最大0.013515。
- final总中心位移L2总体均值0.032687、最大0.066424；类内残差误差≤`4.17e-17`，FFT/RF严格0。逐方向变换没有破坏类对称、support-only或K1/K2恒等边界。
- 尽管方向中心替换幅度非零，15/15 outer预测仍与D89相同。这比D89进一步说明瓶颈不是“地面谱未进入support”，而是仅移动类公共中心这一作用点在当前D62/LDA决策几何中已经饱和：它能稳定复现一个低仰角旧类纠错，却无法触及14-7、6-15、1-18和14-11的系统性边界错误。
- D90仍保留D81正交径向中心，因此没有退化；但这个保守性也限制了可跨越的margin。下一步若继续，应改变合法作用点，而不是增强同一中心位移：优先研究ground谱约束的类对称“支持协方差收缩强度/共享metric融合”，但必须吸取D80/D83直接把ground covariance/precision注入分类器的负经验，采用support证据控制且不按old/new分支。

## 完整训练、量化与资源

- 完整解析15个目标row×20步=300条记录；loss min/mean/max=`0.07560/0.30719/1.11738`，CE=`0.07541/0.30708/1.11738`，support accuracy=`89.58/98.77/100%`。全部持久JSON非有限数值计数0；stdout/stderr未单独持久化是观测限制。
- INT8/FP32 before/final outer argmax变化0/0，support变化0/0，margin sign flip0；最大score误差0.001350。
- ground/head/总状态=`5,816/8,583/14,399B`；参数2,016；step20；ground谱377,312 MAC；support中心17,982,464 MAC；新增适配18,359,776 MAC；总适配24,909,583,746 MAC；query6,624 MAC；D90额外query MAC/state=0/0；peak CUDA22,886,912B；dense query graph0。
- 相对D89，逐方向计算只增加335,104 MAC，占其新增适配1.86%、占总适配0.00135%；持久状态、参数、step和query路径完全不变。顶层`resource_audit.json`属于forced-selection Z0而hash不变；目标候选15行的resource字段已一致记录上述新增MAC。

## 证据与决定

- training log SHA256=`2a03c4ba8ccb82b2f6888ce941c8cf7451dbb9157dc57353411087c108ac3ee8`。
- receipt SHA256=`374d0b225542099c0f868827909c2e3a8e13b3d51aa4df96e06cbfdf6fcd9f9b`。
- D90 metadata SHA256=`d6d1c6e8df8e4c08781e3e52ee7b66860f77726aff89550474582677af547101`。
- 完整汇总：`E:\type10-7\automation_reports\CV-SincNet\d90_v2_directionwise_cauchy_center_20260720\d90_full_performance_summary.json`，SHA256=`7ab0951225c395823ea974f0ce9658e30eca726d5e1ac28ea7f204a084b9bd18`。
- 决定：D90停止，不进seed2/125、不扫描方向混合或强度；保留为“方向机制有效执行但中心作用点饱和”的负证据。

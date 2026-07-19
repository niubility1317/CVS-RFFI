# D74类中心正交nuisance方向删除实验报告

## 1.实验身份与状态

|字段|值|
|---|---|
|实验ID|`d74_orthogonal_nuisance_direction_removal_probe_20260720`|
|候选|`orthogonal_nuisance_direction_removal`|
|operator|Codex `/root`|
|状态|`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`|
|目标|删除一个不承载类中心差异、但具有最大类内残差能量的非可逆方向，检验能否突破D62/D73等价边界|
|比较目标|D62：`B/A/N/H/F/J=92.78/82.22/84.67/82.62/10.56/26.67%`|

## 2.协议与机制锁

- `p2_min_v1`、D18匹配`VALIDATED_ONCE` capsule、receiver`20-1`、seed`713101`、K10/new5、3场景×5fold，outer-fit实际K8。
- 单LEO_weak观测、support-only、query逐样本全注册类argmax；无clean/source/query truth/role/quota/global assignment。
- before精确D62；final在D42特征中删除一个与中心化类均值span正交的最大类内残差方向，冻结D62 final头并把`W(I−uuT)`编译进单一int8头。
- rank固定1，无阈值、强度、场景、类、角色或结果扫描；地面组件输入0。

## 3.开发门

相对D62要求`A/N/H/min-A/min-N`不退化且至少一项严格提高，同时`B/F`、场景和混淆无交换伤害。失败即负向关闭，不开第二seed或125矩阵。

## 4.版本、验证、运行和结果占位

`E:\type10-7`不是Git仓库；所有代码、测试、追溯和完整报告进入`E:\type10-7\github_publish\CVS-RFFI-repo`，根目录只保留同步镜像。实现后补录commit、clean worktree、source SHA、完整命令、PID/GPU/log/output、105行闭包、7候选、3场景、11类、15fold、机制、训练、量化、资源、artifact、缺陷和最终判定。

|candidate|机制|receiver/TX|K/seed|B|A|N|H|F|J|min-B/A/N|混淆|资源|判定|
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
|D74|rank-1非可逆nuisance删除＋冻结D62 final头|20-1/new5|K10/713101|92.78|80.56|79.33|78.81|12.22|26.67|80.00/53.33/63.33|24/13/18|见第17节|负向，不晋级|

## 5.实现锁定

|文件|用途|SHA256|
|---|---|---|
|`code/cvsrffi/stage2_d74_orthogonal_nuisance_removal.py`|中心span、类内残差SVD、rank-1投影和不变量审计|`6584e14a918b2217e96093feb2ffefbf60009257d16674913588931b8e455444`|
|`code/scripts/probe_d74_orthogonal_nuisance_direction_removal.py`|D62包装、`W(I−uuT)`编译、资源/source/闭包|`3661618f848f94d29c3a188d68b6eba8de22ca0ad014cb55ceb9502db81ed375`|
|`tests/test_stage2_d74_orthogonal_nuisance_removal.py`|非可逆、中心保护、置换等变、K1/fail-closed|`b292166f4278d683251e0e5f0a7ef18158867b76b4943c5703b4062ea10f5e5d`|
|`tests/test_probe_d74_orthogonal_nuisance_direction_removal.py`|D62继承、资源公式、调用和协议闭包|`a4053995f901adb0a35ab61ea35fbe63b9a69798cc137e851cbf2667170187e5`|

`ssr-gpu`专项测试8/8通过。首次实现预期增加D62 refit，但R1因严格降秩与D43 SPD前提不兼容而改为冻结强头；R1不增加closed-form fit、optimizer step或epoch，投影方向编译后不持久化，query额外MAC/state0。

## 6.完整验证与运行锁

- 实现commit=`eb22322c9e2e6d24817cbcee0ba0778e5d424df2`；clean worktree=`E:\type10-7\code\snapshots\d74wt`，detached HEAD且clean。
- 主工作树与clean worktree的D42–D74相邻42文件、385项测试均通过，用时82.7/82.9秒；core/probe `py_compile`通过。
- clean执行SHA：probe=`e65db3025fc9bd834ff530544b23f9d5b8a935e5567a8b5675b20533f7056fe4`、core=`2f098c8c3311ce0da9a62ace354c3c005d68da1161a82a265e70976d221e0f2f`、D62 helper=`38ae1114a06d135bca806f470417cd28a634fec0da449888665c6843615d4a20`。
- 01:22:35启动前输出目录不存在；GPU0 RTX5070Ti显存`954/16303MiB`、利用率0%。本轮本地执行，不访问N607。
- 首次启动预期闭包已由R1替代；R1锁定为105行、30目标行、30 top fit、0额外D62 refit、1080 component execution、30份rank-1投影audit、ground/query-fit/clean/source/role/quota访问0。

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d74wt\code\scripts\probe_d74_orthogonal_nuisance_direction_removal.py' `
  --d74-arm orthogonal_nuisance_direction_removal `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' --probe-root 'E:\type10-7\code\snapshots\d74wt' `
  --before-root 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\before\enrollment_only' `
  --before-seal 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\before_enrollment.seal.json' --before-seal-sha256 53ace2863c9da6c2f6cc855d602c99f581df6de3d30a9a3ecb89eb6b6f0d9f75 `
  --before-formal-policy 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\formal_execution_policy.json' `
  --before-formal-policy-authorization 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\before_formal_policy_authorization.v2.json' `
  --before-signed-policy-authorization-envelope 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\before_signed_policy_authorization_envelope.v2.json' --before-signed-policy-authorization-envelope-sha256 31a2ad9918f061b25d5a7ed0cc135df70ae02460c094b2f396bf314817bceb0e `
  --after-root 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\after\enrollment_only' `
  --after-seal 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\after_enrollment.seal.json' --after-seal-sha256 c70aedf3a8f059e756806201758c1933a2f3e1ba4df415e69a1c776b1a2b50ff `
  --after-formal-policy 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\formal_execution_policy.json' `
  --after-formal-policy-authorization 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\after_formal_policy_authorization.v2.json' `
  --after-signed-policy-authorization-envelope 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\after_signed_policy_authorization_envelope.v2.json' --after-signed-policy-authorization-envelope-sha256 a2483d6e9c9c362d89397029ff1e43f48358be3bdb3a05d717ee112b70a0be76 `
  --component-dir 'E:\type10-7\automation_reports\CV-SincNet\d22_int8_anchor_lifecycle_20260717\input\int8_component' --component-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c `
  --class-binding 'E:\type10-7\github_publish\CVS-RFFI-repo\analysis\d19_adv3b02_class_binding_20260717.json' --class-binding-sha256 bb89a1dbb831acb374fccfc596ae98b660b496b449bdca577dabb962121c901f `
  --output 'E:\type10-7\automation_reports\CV-SincNet\d74_orthogonal_nuisance_direction_removal_probe_20260720\orthogonal_nuisance_direction_removal' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

## 7.启动与监控

- 2026-07-20 01:24:12启动唯一执行，PID`23556`；只读命令行与锁定参数一致，stderr 0B。
- 当前只读离散监控，不重复启动；进程退出后验证105行、projection audit、RECEIPT和metadata。

## 8.首次启动结构失败与R1

- PID`23556`在首个outer row前失败，输出目录为空，无training log/RECEIPT/可评分结果。精确异常为D74严格降秩后的support进入D62 refit时，D43 block协方差触发`structured covariance is not positive definite`。
- 不采用jitter或伪逆绕过正定门，因为会把非可逆机制改回近似可逆并削弱fail-closed边界。
- R1保留同一`u/P`，冻结既有D62 final头，直接编译`W'=W(I−uuT)`；不再新增D62 fit。它不读取任何性能结果，且更直接检验“非可逆删除能否改变固定强头边界”。
- 原空目录和launcher stderr保留；R1完成测试、commit和新clean worktree后只使用`orthogonal_nuisance_direction_removal_retry1`。

R1锁定commit=`23f43510f13a8c98ce325d51f93aa1c39462037c`；专项8/8、主工作树D42–D74完整链385/385（82.7秒）通过。clean worktree=`E:\type10-7\code\snapshots\d74r1wt`，专项8/8与`py_compile`通过且clean。执行SHA：probe=`427be77328700c524173689567423b861bd18dd57fb8d96d7a4fcd5c6d4e363d`、core=`2f098c8c3311ce0da9a62ace354c3c005d68da1161a82a265e70976d221e0f2f`、D62 helper=`38ae1114a06d135bca806f470417cd28a634fec0da449888665c6843615d4a20`。01:30:53检查retry1目录不存在；GPU显存1422MiB、瞬时利用率2%，只读进程检查无Python任务，允许本地单实例启动。

## 9.R1完成状态与artifact闭包

- 2026-07-20 01:32:15启动retry1，PID`24160`；只读检查确认Python命令行、D18封存输入、D22组件清单、D19类绑定、clean worktree和独立输出目录与锁定报告一致。
- PID于01:34:30前正常退出，Runner实测`elapsed_seconds=127.4620`，launcher stderr为0B。
- RECEIPT状态=`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，`selected_candidate_id=Z0_SUPPORT_ONLY`，`selected_positive_route=false`，`query_opened=false`；不允许formal/performance claim。
- 闭包：105/105行、7候选×15fold、30个target量化/FP32行、30次top fit、0次D74额外D62 refit、1080次D62 component execution、30份投影audit；目标行20/20训练步，ground/query-fit/clean/source/query-role/quota访问0。
- 完整摘要：`E:\type10-7\automation_reports\CV-SincNet\d74_orthogonal_nuisance_direction_removal_probe_20260720\d74_r1_full_performance_summary.json`，98,741B，SHA256=`9771486b2ba61b0a150b4ec73069645cb900ad780f619f7ded43ae896b20ead1`。

|artifact|字节|SHA256|
|---|---:|---|
|`training_log.jsonl`|14,739,786|`bd2488b1b736249f3b1ddf641c6e42c119b820002befd74556de48cacf3905fd`|
|`support_audit.json`|313,680|`03e2742ece7f493f1aa50ecb9a1b944854ee78ca59f833e98ce0d1cb939442d2`|
|`resource_audit.json`|6,498|`00f364e567e0462feb955321e6d414c0dc95493dab35d3ed00dfacd71a8d6e2b`|
|`geometry_audit.json`|5,132|`ae4b735a45fdae38eaf8bb6bfd23e57df9d60560144027a1e3e33937556300dc`|
|`selection.json`|2,991|`85e035aa88d9eb59da8bad606c7be9bcd366f3a78165b0beeefb806398f54f84`|
|`RECEIPT.json`|5,030|`fc3d668db23d902ce115fa5bb4de9506010159728ac3326c132a1ce13e550636`|
|`D74_PROBE_METADATA.json`|2,467|`4b8e5bef32f7d31e5e8552ed55f9f9778b901b732530444c5bfb048528143fd2`|

## 10.同row总体结果与开发门

|candidate|机制|receiver/TX|K/seed|B old|A old|seen-new|unknown|H|forgetting|joint|min-B/A/N|row-floor B/A/N|混淆O→N/N→O/N→N|量化|判定|
|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---|---|---|---|---|
|D74 INT8|D62固定头＋中心正交rank-1残差删除|20-1/new5|K10实际fit K8/713101|92.78|80.56|79.33|N/A，本开发单元无unknown query|78.81|12.22|26.67|80.00/53.33/63.33|73.33/46.67/46.67|24/13/18|INT8=FP32 argmax|负向，不晋级|
|D62 INT8|冻结D42 metric＋crossfitted Fisher row splice|20-1/new5|同上|92.78|82.22|84.67|N/A|82.62|10.56|26.67|80.00/53.33/73.33|73.33/50.00/46.67|23/8/15|INT8=FP32 argmax|当前最强|

D74相对D62：`ΔA=−1.67pp`、`ΔN=−5.33pp`、`ΔH=−3.81pp`、`ΔF=+1.67pp`、`Δmin-N=−10.00pp`；12/15个outer prediction SHA改变。严格开发门失败，不运行第二seed、K1/K5/K20或125矩阵。相对活动K10目标，`A`差11.44pp、`min-A`差34.67pp、`new5`差12.67pp。

## 11.三场景表现

|场景|B|A|N|H|F|J|min-B/A/N|row-floor B/A/N|混淆O→N/N→O/N→N|相对D62主要变化|
|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
|LEO clear weak|98.33|88.33|96.00|91.36|10.00|40.00|90.00/70.00/90.00|90.00/50.00/90.00|3/2/0|A−3.33pp，N−2.00pp，J−10.00pp|
|LEO low elev weak|91.67|80.00|70.00|73.20|11.67|20.00|80.00/60.00/50.00|70.00/60.00/20.00|8/7/8|A+1.67pp但N−6.00pp，H−2.78pp|
|LEO rain weak|88.33|73.33|72.00|71.87|15.00|20.00|60.00/30.00/50.00|60.00/30.00/30.00|13/4/10|A−3.33pp，N−8.00pp，H−5.58pp，min-N−20.00pp|

删除方向在三个场景都没有形成联合改善。低仰角只获得1.67pp旧类均值，代价是6.00pp新类；雨衰同时伤害旧类、新类和下尾新类，是主要负向来源。

## 12.逐类总体准确率

类编号按Runner注册顺序，仅用于报告；D74公式对类置换等变，不使用类ID、old/new角色或场景专用分支。

|类|before-old|after-old|遗忘/变化|
|---|---:|---:|---:|
|O1|96.67|86.67|−10.00pp|
|O2|96.67|93.33|−3.33pp|
|O3|96.67|86.67|−10.00pp|
|O4|80.00|53.33|−26.67pp|
|O5|93.33|80.00|−13.33pp|
|O6|93.33|83.33|−10.00pp|

|类|seen-new准确率|
|---|---:|
|N1|63.33|
|N2|86.67|
|N3|73.33|
|N4|86.67|
|N5|86.67|

最弱旧类仍为O4=53.33%；最弱新类N1从D62的73.33%降至63.33%。D74没有修复遗忘下尾，还扩大了新类注册下尾。

## 13.15个outer fold完整同row表

|场景|fold|B|A|N|H|F|J|floor B/A/N|混淆O→N/N→O/N→N|
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
|clear|0|100.00|100.00|80.00|88.89|0.00|50.00|100/100/50|0/2/0|
|clear|1|100.00|83.33|100.00|90.91|16.67|0.00|100/0/100|0/0/0|
|clear|2|91.67|75.00|100.00|85.71|16.67|50.00|50/50/100|2/0/0|
|clear|3|100.00|91.67|100.00|95.65|8.33|50.00|100/50/100|0/0/0|
|clear|4|100.00|91.67|100.00|95.65|8.33|50.00|100/50/100|1/0/0|
|low-elev|0|100.00|75.00|80.00|77.42|25.00|50.00|100/50/50|3/1/1|
|low-elev|1|83.33|58.33|70.00|63.64|25.00|0.00|50/50/0|2/0/3|
|low-elev|2|83.33|91.67|60.00|72.53|−8.33|0.00|50/50/0|0/2/2|
|low-elev|3|100.00|100.00|60.00|75.00|0.00|0.00|100/100/0|0/2/2|
|low-elev|4|91.67|75.00|80.00|77.42|16.67|50.00|50/50/50|3/2/0|
|rain|0|83.33|75.00|50.00|60.00|8.33|0.00|50/50/0|2/1/4|
|rain|1|100.00|66.67|80.00|72.73|33.33|0.00|100/0/50|4/1/1|
|rain|2|91.67|83.33|80.00|81.63|8.33|50.00|50/50/50|1/0/2|
|rain|3|83.33|66.67|60.00|63.16|16.67|0.00|50/0/0|3/2/2|
|rain|4|83.33|75.00|90.00|81.82|8.33|50.00|50/50/50|3/0/1|

## 14.机制激活、训练与失败机理

- 15/15个fold均真实启用rank-1删除，且方向SHA有15种；中心span秩固定10、正交残差秩固定77、最终投影秩287。它不是D73那种被refit完全吸收的等价变化。
- 删除的首个正交残差奇异方向占正交残差能量9.53%–14.94%、均值11.27%；占全部类内残差能量7.08%–9.35%、均值7.79%。类中心方向残量最大`2.78e−17`，类中心两两距离漂移最大`1.11e−16`，几何约束严格成立。
- support准确率投影前均值99.85%、投影后99.70%，15折合计仅2个support预测改变；然而outer 12/15fold改变并整体下降。该反差说明“support内高能且类中心正交”不等于“域外nuisance”：弱场景泛化依赖的判别方向可能不改变support类中心，却体现在细粒度类内结构和边界margin中。
- D42 Stage2-B训练完整保留：epoch1 loss均值1.031996、support acc95.14%、gradient norm1.08376；epoch20 loss0.102685、support acc100%、gradient norm0.13535。D74是闭式SVD投影，Stage2-C optimizer step=0，query rows=0。
- 相对D62，outer旧→新错误+1、新→旧错误+5、新→新误分+3。主要伤害并非旧类单向被新类吞噬，而是删除后新类边界整体变弱并转向旧类或其他新类。

## 15.与近期版本的matched比较

|比较|ΔB|ΔA|ΔN|ΔH|ΔF|ΔJ|Δmin-B/A/N|prediction hash变化|混淆ΔO→N/N→O/N→N|解释|
|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
|D74−D62|0.00|−1.67|−5.33|−3.81|+1.67|0.00|0/0/−10.00|12/15|+1/+5/+3|非可逆删除真实生效但联合恶化|
|D74−D73|0.00|−1.67|−5.33|−3.81|+1.67|0.00|0/0/−10.00|12/15|+1/+5/+3|D73等价D62，因此同样被支配|
|D74−D72|−0.56|−2.22|−3.33|−2.78|+1.67|0.00|0/0/−6.67|11/15|+2/+2/+3|不如bagging负向版|
|D74−D71|+1.67|−1.67|−4.67|−3.52|+3.33|0.00|−3.33/0/−10.00|13/15|+1/+4/+3|更高B不能补偿A/N/H退化|
|D74−D61|+2.78|−2.78|+3.33|−0.15|+5.56|0.00|+3.33/−6.67/+20.00|15/15|+6/−3/−2|D61旧类保护更强，D74仍无联合优势|

## 16.量化表现

- D74 INT8与matched FP32的before outer、final outer、before support、final support argmax变化均为0，margin符号翻转0。
- 最大score绝对量化误差：fold最小0.000408、均值0.000877、最大0.001907。
- 最差margin仍明显为负：old-new最小−2.0702、new-old最小−4.9000、new-new最小−1.0880。失败来自FP32边界本身，不是INT8近似。

## 17.资源表现

|资源|D74|D62|增量/说明|
|---|---:|---:|---|
|trainable parameters|2,016|2,016|不增加|
|optimizer steps/epochs|20/20|20/20|投影为闭式0步|
|closed-form component fits|72|72|不增加|
|LDA fit MAC|18,000,009,216|18,000,009,216|不增加|
|D74投影＋编译MAC|18,190,656|0|一次性support适配|
|total adaptation MAC|24,909,414,626|24,891,223,970|+18,190,656，增加0.0731%|
|query MAC|6,624|6,624|额外0|
|persistent/registry state|8,583/941B|8,583/941B|方向已编译，额外0|
|peak CUDA memory|22,886,912B|22,886,912B|不增加|
|dense query graph|0B|0B|通过|
|ground int8 component input|0|0|D22未获正式资格|

D74满足≤80k参数、≤30epoch、≤50step、≤256KB状态和无dense query graph上限，计算增量很小；但它在A、N、H、F和min-N上被D62严格支配，资源合规不能转化为性能晋级。

## 18.缺陷、结论与下一步

1.核心缺陷是nuisance判据不充分：只保护类中心span会把support中高能、但对outer弱场景边界有用的类内判别方向误当作nuisance。
2.support内部几乎无变化却outer显著下降，说明不能再用support重构能量或support准确率单独筛删除方向；下一版必须用support-only的交叉拟合稳定性/留一类margin代理约束“删除后不伤害旧、新两侧”。
3.D74全程没有使用地面压缩旧类原型。D22仍为`formal_phase2_eligible=false`且provenance unverified；D66曾合法读取84个ground int8单元但结果为负。不能为了降低遗忘绕过数据协议。
4.停止D74的rank、删除强度、SVD顺序、场景/类/角色门、第二seed和125矩阵；这些都缺少正向开发证据。
5.下一轮D75应保留D62固定强头和非可逆/可编译原则，但只允许删除同时满足“中心正交”和“旧/新support交叉拟合margin不下降”的方向；若无安全方向则identity。该门必须类对称、角色盲、query零访问，并对旧/新注册同等约束。

最终判定：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。当前最强协议合法开发版本仍为D62，而不是D74。

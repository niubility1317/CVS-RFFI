# D66地面域可靠性残差开发探针

## 1.执行前登记

- 实验ID：`d66_ground_domain_reliability_residual_probe_20260719`。
- 时间：2026-07-19；operator：Codex。
- 目标：真正使用不可变Phase1地面int8域×类聚合知识，同时避免旧类专属anchor导致的新类塌缩；检验共享地面域可靠性变换能否在D62基础上同时改善旧类域适应与新类注册。
- 比较目标D62：before92.78%、after82.22%、new84.67%、H82.62%、forget10.56pp、joint26.67%、min-before80%、min-after53.33%、min-new73.33%、混淆23/8/15。
- cell：receiver`20-1`、seed`713101`、K10/new5、3场景×5 outer fold，实际K8；复用匹配`VALIDATED_ONCE/p2_min_v1`的D18 enrollment-only support，不重验数据。
- 根目录`E:\type10-7`不是Git仓库；版本化实现和本报告镜像位于`E:\type10-7\github_publish\CVS-RFFI-repo`。执行前Git HEAD为`51e375ada1ffcd56516b01dce88dd0b5b359d937`；工作树存在大量不属于本轮的既有修改，本轮只暂存D66精确路径。

## 2.机制与历史边界

D66从84个有效的地面域×旧类int8聚合单元计算每个z160坐标的类间身份方差`B`和同类跨域漂移`W`，固定`r=(B+eps)/(B+W+2eps)`、`s=sqrt(1+r)`。z160使用共享尺度`s`，FFT96/RF32恒等；D62全部支持拟合在共享坐标执行，再把系数编译回原坐标。对旧类、新类和未来query没有不同公式，query零额外MAC/state。

历史停止项：D19/D25/D36旧类专属anchor中心融合、独立半径似然、角色offset/IRLS、D30 old-old DALI、旧anchor Procrustes/transport及query batch统计。D66不复用这些机制，不持久化反量化ground bank，不读取clean/source样本或query。

当前组件manifest标记`UNVERIFIED_UNDER_CURRENT_PROTOCOL`，所以本轮严格限定为开发support内部held-rank诊断，formal/query/performance claim和125权限均为false。组件必须只读，入口/出口SHA均应为`3c08c823d2e8a13c4233f0060ac67c332ecc8d6e8abec7352de975fead0267d7`。

## 3.预注册门

- 完成七候选×三场景×五折=105行，query/clean/source/role/quota/global assignment访问均为0。
- 相对D62总体、三场景、逐类floor、遗忘、混淆和量化不得交换伤害，并至少严格改善after、forgetting、joint或任一floor。
- 必须报告七候选、三场景、11类、15fold、地面尺度统计、FP32/int8量化、训练/适配MAC、状态、延迟和完整artifact。
- 失败即状态`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`并停止本路线；成功也先完成D64–D66三轮回顾，不直接启动125。

## 4.待完成实现与运行信息

初版专项4/4、D42–D66完整回归303/303通过。额外随机合成的真实D42＋D62烟测在任何项目support/query打开前触发`D42 sklearn coefficient deployment prediction drift`；移除D66后，未改动D62在同一合成数据上复现完全相同错误，故不能归因于共享尺度。未放宽D42闭包断言，保留预注册的共享坐标拟合公式；真正集成判据为锁定项目enrollment-only support上的105行fail-closed运行。

待补：本地变更、验证命令、Git提交、干净worktree、精确运行命令、环境、输出路径、运行时、完整结果与下一实验建议。

## 5.实现、验证与版本状态

- 新增`code/scripts/probe_d66_ground_domain_reliability_residual.py`：组件策略/allowlist/SHA闭包、规范registry排序、84-cell反量化、共享可靠性尺度、D62坐标注入、系数编译、资源和输出验证。
- 新增`tests/test_probe_d66_ground_domain_reliability_residual.py`：组件只读与策略fail-closed、类置换逐bit尺度不变、尺度边界、全类统一编译等价和无角色/场景/可调分支。
- 预注册提交`fc7c0977`；实现提交`684e110edddf5adaafe22200cb044ddd56059bcd`；实现脚本SHA将在运行artifact中自动锁定。
- 主工作树专项4/4、D42–D66完整26文件303/303通过；干净worktree`E:\type10-7\code\snapshots\d66wt`在同一提交再次303/303通过，用时85.2s；`py_compile`和`git diff --check`通过。
- 真实组件：26域×6类、84个有效cell，每类14个；可靠性0.0242749–0.9999186，尺度1.0120647–1.4141848，条件数1.3973265，尺度SHA256=`70a8e94327e7100695f691d6ae49e246305036cefd92579e977e3d536c37df6c`；组件逻辑状态25,428B，瞬时反量化53,760B，统计58,880MAC。
- 本轮完全本地，不需要SSH/SCP，不占用或干预N607。Conda/Python环境为`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`；运行设备`auto`，由锁定runner记录实际GPU/CPU和峰值显存。

## 6.精确运行命令

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d66wt\code\scripts\probe_d66_ground_domain_reliability_residual.py' `
  --d66-arm ground_domain_reliability_residual `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' --probe-root 'E:\type10-7\code\snapshots\d66wt' `
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
  --output 'E:\type10-7\automation_reports\CV-SincNet\d66_ground_domain_reliability_residual_probe_20260719\ground_domain_reliability_residual' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

预期输出为105行training log、support/query/selection/receipt、geometry/resource和D66 metadata。任何组件、策略、support、D42/D62、编译、资源或输出闭包失败均停止，不覆盖输出、不重跑同目录。

## 7.首次运行与Resource-R1预注册

- 首次运行完成105/105行、query0、Runner129.0378s、外层137.7s，组件入口/出口SHA、source closure和D66 metadata均通过。
- 完整日志解析发现资源主字段漏加地面组件：`d66_ground_component_logical_state_bytes=25,428`和`ground_int8_component_input_count=84`正确，但runner后置逻辑把`persistent_state_bytes`覆盖为仅仿射头8,583B。正确组件含总状态为34,011B，仍低于256KB。
- 该缺陷不影响预测或性能，但首次artifact不能封为最终资源证据，先不发布最终D66性能判定。首次目录原样保留。
- Resource-R1只修资源后置加总与硬断言，不改公式、support、候选、训练、量化或预测；新输出为`ground_domain_reliability_residual_resource_r1`，不得覆盖首次目录。

## 8.Resource-R1执行前闭包

- 修复提交：`b7388395e9d905db2a8b7f01b047370b30276028`。D66外层在锁定runner返回后保留仿射头状态`d66_compiled_affine_state_bytes`，并强制`persistent_state_bytes=d66_compiled_affine_state_bytes+d66_ground_component_logical_state_bytes=8,583+25,428=34,011B`；同时新增组件含总状态一致性断言和摘要字段。
- 干净worktree：`E:\type10-7\code\snapshots\d66r1wt`，detached HEAD=`b7388395e9d905db2a8b7f01b047370b30276028`，创建后`git status -sb`仅为`## HEAD (no branch)`。
- 干净全链验证：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest -q <D42-D66的26个测试文件> --basetemp local_artifacts\d66r1_clean_full_chain`，304/304通过，用时81.6s。
- Resource-R1仍使用相同D18 enrollment-only support、receiver`20-1`、seed`713101`、K10/new5、3场景×5fold、七候选和`development_select_unverified_component`；组件manifest仍为`UNVERIFIED_UNDER_CURRENT_PROTOCOL`，不产生formal/query/performance claim，也不获得125权限。
- 唯一允许变化的是D66脚本/source closure及资源字段；逐fold标签、预测、分数、候选、训练轨迹、量化诊断和全部性能字段必须与首次运行一致。若任何预测或性能变化，Resource-R1不得作为等价重封。

Resource-R1精确命令与第6节相同，但使用：

```text
script/probe-root = E:\type10-7\code\snapshots\d66r1wt
output = E:\type10-7\automation_reports\CV-SincNet\d66_ground_domain_reliability_residual_probe_20260719\ground_domain_reliability_residual_resource_r1
runtime-root = E:\type10-7\code\snapshots\d41wt
device = auto
mode = development_select_unverified_component
candidate-set = d42_v1
```

成功闭包：105/105行、query/clean/source/role/quota/global assignment访问为0；每行`persistent_state_bytes=34,011`、仿射头8,583B、组件25,428B且256KB cap通过；首次与R1逐fold预测和全部性能严格等价；无非有限值、异常或错误marker。完成后必须生成D66专属完整摘要并补齐总体、场景、11类、15fold、量化、训练、资源、artifact及D62/D64/D65对照表。

## 9.Resource-R1完成状态与等价性

- 运行完成105/105行，退出码0；锁定runner耗时124.4298s，外层命令耗时132.7s；七候选各15行，query未打开。
- receipt状态：`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`；`selected_candidate_id=Z0_SUPPORT_ONLY`、`selected_positive_route=false`、formal/performance claim均为false。
- 首次运行与Resource-R1的105行键`candidate_id/scenario/fold_index`逐行一致。递归比较全部JSON字段后，差异只有：30行新增`d66_compiled_affine_state_bytes=8,583`与`d66_component_inclusive_persistent_state_bytes=34,011`，这30行主`persistent_state_bytes`由8,583改为34,011；另15行仅有运行时`adaptation_latency_sec`变化。除此以外的预测、性能、训练轨迹、量化、几何、support和拟合字段全部严格相同。
- 协议闭包：105行`query_rows_used_for_fit=0`、`query_opened=false`、`source_sample_access=false`；clean/source、query feature/label/truth、role Oracle、true batch class count、class quota、global assignment、query-dependent optimization均为false或不适用。组件入口/出口SHA一致且未更新；训练日志没有非有限值。

## 10.七候选完整同排性能

以下各行均为receiver`20-1`、seed`713101`、声明K10/实际K8、new5、clear/low/rain各5fold。指标均是同一候选15行联合统计；百分数单位为%，forget为pp。所有行coverage/rollback/defer均不适用，因为本轮未开query、未作部署或回滚决定。

|候选|机制/类别|before|after|new|H|forget|joint|min-before|min-after|min-new|旧→新|新→旧|新→错新|结论|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|`B3_SINGLE_IQ_DIAG_FFTRF`|单IQ FFT/RF诊断|87.78|75.56|72.67|73.35|12.22|23.33|80.00|60.00|40.00|33|22|19|负对照|
|`D42-D40-HNBR-INT8-NEGATIVE`|HNBR int8负控|85.56|85.00|15.33|25.16|0.56|0.00|66.67|63.33|0.00|2|0|0|新类塌缩|
|`D42-D41-BEC-INT8-NEGATIVE`|BEC int8负控|86.11|20.56|78.67|31.50|65.56|0.00|76.67|0.00|36.67|142|0|32|旧类塌缩|
|`D42-PROTOnet-CDA-ZID160`|ProtoNet-CDA对照|71.11|48.33|52.67|48.97|22.78|0.00|33.33|13.33|3.33|0|0|0|联合失败|
|`D42-USLDA-FP32-MATCHED`|D66匹配FP32诊断|93.33|83.33|83.33|82.59|10.00|23.33|80.00|53.33|66.67|20|9|16|仅量化对照|
|`D42-USLDA-INT8`|D66正式开发行|93.33|83.33|83.33|82.59|10.00|23.33|80.00|53.33|66.67|20|9|16|诊断负结果|
|`Z0_SUPPORT_ONLY`|support-only零路线|71.11|48.33|52.67|48.97|22.78|0.00|33.33|13.33|3.33|0|0|0|锁定fallback|

D66行使用20epoch/20step适配，epoch1平均loss1.031996、support accuracy95.14%；epoch20平均loss0.102685、support accuracy100%，prototype anchor loss由近0增长到0.003828。训练收敛不能弥补held-rank的旧/新联合泛化缺口。

## 11.D66按场景、逐类与逐fold表现

|场景|before|after|new|H|forget|joint|min-before|min-after|min-new|旧→新|新→旧|新→错新|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`leo_clear_weak`|98.33|91.67|98.00|94.44|6.67|50.00|90.00|70.00|90.00|2|1|0|
|`leo_low_elev_weak`|91.67|81.67|72.00|75.88|10.00|10.00|80.00|60.00|30.00|5|6|8|
|`leo_rain_weak`|90.00|76.67|80.00|77.45|13.33|10.00|60.00|30.00|70.00|13|2|8|

clear场景接近门槛，但low-elev的新类与rain的旧类分别成为主要短板；这说明共享地面可靠性尺度没有解决场景最坏类，而不是总体平均数被单一坏fold偶然拉低。

|类别|Phase1 TX|before-old|after-old|遗忘/seen-new|
|---|---|---:|---:|---:|
|old `cls_1f33`|20-15|96.67|93.33|3.33pp|
|old `cls_33bb`|8-20|96.67|93.33|3.33pp|
|old `cls_75aa`|14-10|96.67|93.33|3.33pp|
|old `cls_8b02`|14-7|80.00|53.33|26.67pp|
|old `cls_a53c`|6-15|96.67|73.33|23.33pp|
|old `cls_f8df`|20-19|93.33|93.33|0.00pp|
|new `cls_09f8`|未绑定Phase1旧TX|—|—|66.67|
|new `cls_1c2a`|未绑定Phase1旧TX|—|—|93.33|
|new `cls_b8fb`|未绑定Phase1旧TX|—|—|76.67|
|new `cls_d3af`|未绑定Phase1旧TX|—|—|90.00|
|new `cls_f608`|未绑定Phase1旧TX|—|—|90.00|

旧类遗忘高度集中在`14-7`和`6-15`；新类最低为`cls_09f8`。共享尺度保护了已经容易的旧类，却没有给这些困难类提供类条件校正。

|场景/fold|before|after|new|H|forget|joint|floor(b/a/n)|混淆(旧→新/新→旧/新→错新)|预测SHA前12位|
|---|---:|---:|---:|---:|---:|---:|---|---|---|
|clear/0|100.00|100.00|90.00|94.74|0.00|50.00|100/100/50|0/1/0|`a1501e459c6a`|
|clear/1|100.00|83.33|100.00|90.91|16.67|0.00|100/0/100|0/0/0|`0faa5815eb82`|
|clear/2|91.67|83.33|100.00|90.91|8.33|50.00|50/50/100|1/0/0|`eb2de0e55cb1`|
|clear/3|100.00|100.00|100.00|100.00|0.00|100.00|100/100/100|0/0/0|`431bed499602`|
|clear/4|100.00|91.67|100.00|95.65|8.33|50.00|100/50/100|1/0/0|`81087026ef31`|
|low/0|100.00|83.33|80.00|81.63|16.67|50.00|100/50/50|2/1/1|`a766632563c1`|
|low/1|83.33|58.33|70.00|63.64|25.00|0.00|50/50/0|1/0/3|`961635069170`|
|low/2|83.33|91.67|70.00|79.38|-8.33|0.00|50/50/0|0/2/1|`2ed783b01303`|
|low/3|100.00|100.00|70.00|82.35|0.00|0.00|100/100/0|0/1/2|`85440ae549df`|
|low/4|91.67|75.00|70.00|72.41|16.67|0.00|50/50/0|2/2/1|`cf089f6fa273`|
|rain/0|83.33|83.33|60.00|69.77|0.00|0.00|50/50/0|2/0/4|`73040cc58c31`|
|rain/1|100.00|66.67|90.00|76.60|33.33|0.00|100/0/50|4/1/0|`7961cf300bb2`|
|rain/2|91.67|83.33|80.00|81.63|8.33|50.00|50/50/50|1/0/2|`9422dcbc4036`|
|rain/3|91.67|75.00|90.00|81.82|16.67|0.00|50/0/50|3/0/1|`de35410e8146`|
|rain/4|83.33|75.00|80.00|77.42|8.33|0.00|50/50/0|3/1/1|`da6cf9f0d1f1`|

## 12.地面知识、D62门控、量化与资源

- 地面知识确实参与拟合：26域×6旧类、84个有效域类cell，每类14个；每个before/final D66拟合均记录`ground_int8_component_input_count=84`，总组件拟合调用1,080次，组件入口/出口SHA均为`3c08c823...0267d7`且只读。
- 可靠性`r`最小/均值/最大0.024275/0.769892/0.999919；共享尺度`s`为1.012065/1.324064/1.414185，条件数1.397327，尺度SHA=`70a8e943...df6c`。它只改变z160坐标，FFT96/RF32恒等，并对旧/新类使用完全相同公式。
- D62门控在before的15fold中8fold激活、6fold atomic fallback、1fold无行fallback，共接纳22个类行；final仅3fold激活、12fold atomic fallback，共接纳7个类行，rain final为0fold激活。因此最终大部分预测退回D46，地面共享尺度并未让注册后门控稳定工作。
- 系数编译回原坐标的最大误差：before`1.2064e-8`、final`1.1075e-8`，均远低于逐fold约`1e-5`容差。
- int8相对匹配FP32：before argmax变化0、outer argmax变化1、margin符号翻转0；最大分数误差最小/均值/最大为0.000250/0.000815/0.001781。虽然总体指标因该单个argmax未变化，严格量化零变化门仍失败。

|资源项|D66实测/审计值|
|---|---:|
|LDA闭式fit|72|
|LDA fit MAC|18,000,009,216|
|D62额外fit/MAC|36/16,934,178,816|
|Fisher上界MAC|6,879,707,136|
|D62 gate标量MAC|10,048|
|D66地面统计＋K8变换MAC|58,880＋21,760=80,640|
|总适配MAC|24,891,304,610|
|每query MAC|6,624；D66额外0|
|训练参数|2,016|
|仿射头/地面组件/组件含总状态|8,583B/25,428B/34,011B|
|瞬时反量化ground内存|53,760B|
|峰值CUDA内存|22,886,912B|
|适配epoch/optimizer step|20/20|

资源cap全部通过，但适配成本主要来自D62嵌套LDA/Fisher；地面可靠性本身仅增加80,640MAC。Resource-R1证明了地面压缩知识不仅被读取，也已正确计入持久状态。

## 13.与当前关键版本同排比较

|版本|before|after|new|H|forget|joint|min-before|min-after|min-new|混淆旧→新/新→旧/新→错新|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|D62|92.78|82.22|84.67|82.62|10.56|26.67|80.00|53.33|73.33|23/8/15|
|D64|92.78|74.44|77.33|75.39|18.33|43.33|86.67|60.00|66.67|37/16/18|
|D65|92.22|86.11|59.33|67.12|6.11|16.67|80.00|70.00|46.67|16/28/33|
|D66|93.33|83.33|83.33|82.59|10.00|23.33|80.00|53.33|66.67|20/9/16|

相对D62，D66的before+0.56pp、after+1.11pp、forget改善0.56pp、旧→新减少3次；代价是new-1.33pp、H-0.03pp、joint-3.33pp、min-new-6.67pp、新→旧和新→错新各增加1次。D66把少量错误从旧类保护转移到了新类，但没有形成联合增益。D65仍是旧类保持最强点，却因new59.33%而不能视为最强联合版本；D62仍是当前聚合联合最强开发版本。

## 14.最终判定、目标缺口与停止项

- 状态：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。D66未通过预注册“不交换伤害”门，也未通过量化零argmax变化门，不启动第二seed或125。
- 相对K10目标：after83.33%距92%差8.67pp；最差旧类after53.33%距88%差34.67pp；new5为83.33%距92%差8.67pp。主要失败不是平均遗忘，而是`14-7`/`6-15`旧类floor及low-elev新类floor。
- D66回答了“是否利用地面压缩旧类原型”：是，已通过84-cell输入、1,080次拟合调用、固定尺度SHA、组件只读SHA闭包和34,011B组件含状态直接证明；但地面知识只形成全类共享坐标可靠性，缺少困难类的support条件残差，所以只能轻微降低平均遗忘，不能修复联合floor。
- 停止重复：不得继续扫描共享尺度alpha/rank/阈值，也不得回到旧类ground中心融合、半径似然、role offset、Procrustes/transport或query batch统计。D64–D66已完成连续三轮，下一候选前必须执行强制回顾，重新核对目标、协议、既有路线和同一运行的注册前/后指标。

最终artifact位于`ground_domain_reliability_residual_resource_r1`。关键SHA：training log=`da0a0f8b...2cc19`、receipt=`a046828f...934dc`、support audit=`4a6bc4dd...ce4c5`、geometry=`ae4b735a...00dc`、resource=`00f364e5...e2b`、D66专属完整摘要=`0ab1833c...e22b9`；共8个文件，无异常、错误marker或非有限值。

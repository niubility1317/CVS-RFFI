# D67交叉拟合registry-consistent连续行堆叠探针

## 1.执行前登记

- 实验ID：`d67_crossfitted_registry_consistent_row_stacking_probe_20260719`；operator：Codex；最终状态：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。
- 目标：综合D62的旧/新联合表现与D65的旧类保持信号，用support-only、类别身份无关的连续闭式行融合提高after-old、遗忘和旧类floor，同时不牺牲seen-new、H、joint和新类floor。
- 当前联合最强D62：before92.78%、after82.22%、new84.67%、H82.62%、forget10.56pp、joint26.67%、min-before/min-after/min-new=80.00%/53.33%/73.33%，混淆23/8/15。
- D65正/负信号：after86.11%、forget6.11pp、min-after70%，但new59.33%、min-new46.67%；说明冻结旧决策有价值，而其新类标尺不可直接采用。
- cell：receiver`20-1`、seed`713101`、K10/new5、三场景×五outer fold、实际K8；直接复用同一`VALIDATED_ONCE/p2_min_v1`D18 enrollment-only support，不重验数据。
- 根目录`E:\type10-7`非Git；代码、追踪和本报告镜像进入`E:\type10-7\github_publish\CVS-RFFI-repo`。执行前主分支含D66最终报告提交`4b9819fa`和三轮回顾提交`eb8e8661`，工作树的其余大量修改均不属于本轮。

## 2.唯一机制与公式

对每个stage和每个匿名注册类`c`，分别生成D62与D65仿射专家`g_e,c(x)`。在四个预锁定physical-rank cross-fit折中，每折held两个rank、train六个rank；所有专家只在train support拟合。

对每个专家行，以train support的一对多统计计算：

```text
center_e,c = (mean_positive + mean_negative) / 2
within_e,c = sqrt((var_positive + var_negative) / 2)
gap_e,c = abs(mean_positive - mean_negative) / 2
scale_e,c = max(within_e,c, gap_e,c, float32_eps)
z_e,c(x) = (g_e,c(x) - center_e,c) / scale_e,c
```

inner-held目标为正类`+1`、其他类`-1`，每类正/负总权重各0.5。令`d=z_65-z_62`，闭式权重为：

```text
alpha_c = clip(sum_i w_i d_i (target_i-z_62,i) / sum_i w_i d_i^2, 0, 1)
h_c(x) = (1-alpha_c) z_62,c(x) + alpha_c z_65,c(x)
g_out,c(x) = center_62,c + scale_62,c * h_c(x)
```

若分母不大于机器精度，`alpha_c=0`，映回后即原始D62行。full support只重算两个专家的center/scale并使用已锁定`alpha_c`；所有`g_out,c`再删除类公共仿射项并编译为一个全注册类affine state。没有role/class ID/scene/receiver分支，没有阈值、温度、alpha扫描、难类名单、outer-held/query拟合或跨query操作。K≤4精确回退D62，避免不合法的小K四折估计。

## 3.假设、判门与停止条件

- 假设：D65旧类保持信号可通过匿名support-held残差获得非零连续权重；其新类失配行会自动得到接近0的权重并回到D62。该判断必须由同一公式产生，不能读取old/new角色。
- 主门：相对D62总体before/after/new/H/joint、三项全局class floor、三场景同类指标、遗忘和三类混淆不得交换伤害，并至少严格改善after、forgetting、joint或任一floor。
- 量化：INT8相对matched FP32的before/final support与outer argmax变化、margin sign flip都必须为0；全部分数有限。
- 结构：四折partition exact-once，所有held rank不得参与对应专家拟合/归一化；`alpha∈[0,1]`，类置换等变；最终只保留单一affine state，query额外MAC/state为0。
- 失败即`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`并停止D62/D65连续行融合；不扫描fold数、alpha温度、ridge、阈值或按角色设权。即使通过也先运行第二development seed，不直接启动125。

## 4.实施与预期证据

待新增独立D67 probe、专项测试和专属摘要；不修改D62/D65历史实现或artifact。测试至少覆盖：四折physical-rank exact-once、闭式权重解析解、类置换等变、K≤4精确D62回退、D65 final专家由旧support冻结协方差后追加、full affine编译等价、无角色/场景/outer-held/query分支及资源闭包。

真实运行前补齐Git提交、干净worktree、完整回归、精确命令、输出路径和资源估计。运行完成后必须在本报告补齐七候选、三场景、11类、15fold、alpha分布/专家贡献、量化、训练、资源、artifact、D62/D64/D65/D66同排对照和目标缺口；不得只报告缺陷。

本轮先本地开发和验证，不访问N607。只有本地锁定候选需要大规模独立seed/matrix时，才按`AGENTS.md`执行N607 preflight、报告、Git、SCP及短连接闭环。

## 5.实现与主工作树验证

- `code/cvsrffi/stage2_d67_row_stacking.py`：四折rank partition、train-only一对多仿射标准化、class-balanced闭式凸权重、映回D62尺度与共同仿射中心化。
- `code/scripts/probe_d67_crossfitted_registry_consistent_row_stacking.py`：D62/D65专家构造、Stage2-B/C lifecycle、嵌套cross-fit、审计、资源和锁定runner接线。
- `tests/test_stage2_d67_row_stacking.py`与`tests/test_probe_d67_crossfitted_registry_consistent_row_stacking.py`：共9项D67专项，覆盖解析解、类置换、exact-once、K≤4回退、生命周期、编译等价与禁止分支。
- `py_compile`通过；D67专项9/9通过；D42–D67完整测试链313/313通过，用时78.6s；尚未运行真实105行，当前没有性能结论。

资源审计将外层D62、每stage四个inner D62、每stage五次D65专家（四inner＋一full）、标准化/闭式权重/编译全部计入适配MAC和fit数；最终持久状态仍是D42单一量化affine，D67 query额外MAC/state为0。下一步必须提交实现、建立干净worktree并复跑313项，再补精确真实命令与输出目录。

## 6.干净验证、版本与真实运行命令

- 实现提交：`6cfef75b1dd0f82e45b5216e93b3b6b18bfd55af`。
- 干净worktree：`E:\type10-7\code\snapshots\d67wt`，detached HEAD为上述提交，`git status -sb`仅`## HEAD (no branch)`。
- 干净D42–D67完整链313/313通过，用时79.3s；与主工作树313/313一致。
- 本轮本地执行，不使用SSH/SCP/N607；Python为`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`，实际设备由锁定runner记录。
- 输出目录固定为`E:\type10-7\automation_reports\CV-SincNet\d67_crossfitted_registry_consistent_row_stacking_probe_20260719\crossfitted_registry_consistent_row_stacking`，运行前不存在；不得覆盖或在失败后原目录重跑。

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d67wt\code\scripts\probe_d67_crossfitted_registry_consistent_row_stacking.py' `
  --d67-arm crossfitted_registry_consistent_row_stacking `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' --probe-root 'E:\type10-7\code\snapshots\d67wt' `
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
  --output 'E:\type10-7\automation_reports\CV-SincNet\d67_crossfitted_registry_consistent_row_stacking_probe_20260719\crossfitted_registry_consistent_row_stacking' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

预期闭包：105行、30个D67 before/final fit、2,760个nested D62 component fit、每个fit四个held/train交集0的partition；query/clean/source/role/quota/global assignment访问0。任何source、lifecycle、partition、alpha、量化、资源或artifact断言失败均停止并保留原目录。

## 7.首次真实运行完成与PostRun-R1计数修复

- 锁定runner已完成105/105行并写出training log、support/selection/geometry/resource/receipt，runner耗时391.7147s，外层401.5s；receipt状态为`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`、query未打开、selected positive route=false。
- 外层随后在D67 metadata前退出：实现预估D67 fit调用30次，实际runner对INT8/FP32两条目标路径分别执行before/final，共60次。相应nested D62 component记录应为60×92=5,520，不是2,760。
- 这是artifact完成后的自检计数缺陷，不是算法、数据、性能、资源或协议失败。原输出目录原样保留，禁止重复401秒计算。
- 使用执行脚本SHA`5a6baa86...97872`只读调用原D67 verifier，已通过：105行、30条目标candidate row、60个fit audit、240个cross-fit partition，`alpha`最小/均值/最大0/0.025459/0.216726，query0；training log SHA=`30e6fdf0...e1430`，receipt SHA=`d2e4eeab...97b6d`。
- PostRun-R1只把正常执行的预期计数修正为60/5,520，并新增`--verify-existing --executed-probe-script`模式。该模式要求既有105行输出和原执行脚本source closure完整、metadata尚不存在，只写新的D67 metadata；不拟合、不预测、不覆盖任何已有artifact。

原第6节“30个fit/2,760记录”的预估现由实测调用结构更正为60个fit/5,520记录。性能仍须在PostRun-R1封存后完整解析，不能从receipt负状态或alpha分布单独判断缺陷。

PostRun-R1首次封存尝试在metadata写入前被source closure拒绝：当前主工作树因Git换行策略产生的D62/D65/core字节SHA与执行用干净worktree不同。修复为从`--executed-probe-script`所属probe root解析并哈希三项helper；不接受当前工作树替代执行字节，也不放宽SHA。该失败未写或覆盖任何artifact。

## 8.PostRun-R1封存与完整性能结论

PostRun-R1从原执行脚本所属干净worktree核对helper闭包后成功，只读封存`D67_PROBE_METADATA.json`。最终闭包为105/105行、7候选×15行、60个D67 fit audit、240个cross-fit partition、5,520个nested D62 component fit execution；所有held/train交集为0，query/clean/source/role/quota/global assignment访问均为0。元数据状态为`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE_POST_RUN_CLOSURE`，不改变任何预测。

### 8.1七候选同排结果

下表每行均来自同一candidate的15个outer row；`F`为遗忘百分点，`J`为同row联合floor均值，`min B/A/N`为跨15行聚合后的最差类准确率，`O→N/N→O/N→Nw`为最终混淆计数。

| candidate | 类别/机制 | B | A | N | H | F | J | min B/A/N | 混淆 | 判定 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|`D42-USLDA-INT8`|D67目标，INT8单affine|92.78|82.78|83.33|82.16|10.00|26.67|80.00/53.33/73.33|22/11/14|负向，不晋级|
|`D42-USLDA-FP32-MATCHED`|matched FP32|92.78|82.78|83.33|82.16|10.00|26.67|80.00/53.33/73.33|22/11/14|量化完全匹配|
|`B3_SINGLE_IQ_DIAG_FFTRF`|单IQ诊断基线|87.78|75.56|72.67|73.35|12.22|23.33|80.00/60.00/40.00|33/22/19|弱于D67|
|`D42-D40-HNBR-INT8-NEGATIVE`|历史负向头|85.56|85.00|15.33|25.16|0.56|0.00|66.67/63.33/0.00|2/0/0|新类失效|
|`D42-D41-BEC-INT8-NEGATIVE`|历史负向边界扩张|86.11|20.56|78.67|31.50|65.56|0.00|76.67/0.00/36.67|142/0/32|旧类坍塌|
|`D42-PROTOnet-CDA-ZID160`|prototype诊断|71.11|48.33|52.67|48.97|22.78|0.00|33.33/13.33/3.33|0/0/0|整体不足|
|`Z0_SUPPORT_ONLY`|support-only对照|71.11|48.33|52.67|48.97|22.78|0.00|33.33/13.33/3.33|0/0/0|整体不足|

### 8.2三场景、逐类与逐fold表现

| 场景 | rows | B | A | N | H | F | J | min B/A/N | 混淆 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`leo_clear_weak`|5|98.33|91.67|96.00|93.57|6.67|50.00|90.00/70.00/90.00|2/2/0|
|`leo_low_elev_weak`|5|91.67|80.00|74.00|75.45|11.67|20.00|80.00/60.00/50.00|8/7/6|
|`leo_rain_weak`|5|88.33|76.67|80.00|77.45|11.67|10.00|60.00/30.00/70.00|12/2/8|

逐类聚合显示遗忘集中在两个旧类，而不是均匀下降：`cls_8b02d999`由80.00%降至53.33%，`cls_a53ca128`由96.67%降至73.33%；其余四个旧类after均为90.00%～93.33%。五个新类准确率分别为73.33%、90.00%、76.67%、86.67%、90.00%。完整类哈希及三场景逐类矩阵封存在`d67_full_performance_summary.json`。

| 场景/fold | B | A | N | H | F | J | floor B/A/N | 混淆 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|clear/0|100.00|100.00|90.00|94.74|0.00|50.00|100/100/50|0/1/0|
|clear/1|100.00|83.33|100.00|90.91|16.67|0.00|100/0/100|0/0/0|
|clear/2|91.67|83.33|90.00|86.54|8.33|50.00|50/50/50|1/1/0|
|clear/3|100.00|100.00|100.00|100.00|0.00|100.00|100/100/100|0/0/0|
|clear/4|100.00|91.67|100.00|95.65|8.33|50.00|100/50/100|1/0/0|
|low/0|100.00|75.00|80.00|77.42|25.00|50.00|100/50/50|3/1/1|
|low/1|75.00|58.33|70.00|63.64|16.67|0.00|50/50/0|2/0/3|
|low/2|91.67|91.67|70.00|79.38|0.00|0.00|50/50/0|0/2/1|
|low/3|100.00|100.00|60.00|75.00|0.00|0.00|100/100/0|0/3/1|
|low/4|91.67|75.00|90.00|81.82|16.67|50.00|50/50/50|3/1/0|
|rain/0|83.33|83.33|60.00|69.77|0.00|0.00|50/50/0|2/0/4|
|rain/1|100.00|66.67|90.00|76.60|33.33|0.00|100/0/50|4/1/0|
|rain/2|91.67|83.33|80.00|81.63|8.33|50.00|50/50/50|1/0/2|
|rain/3|83.33|75.00|90.00|81.82|8.33|0.00|50/0/50|3/0/1|
|rain/4|83.33|75.00|80.00|77.42|8.33|0.00|50/50/0|2/1/1|

### 8.3机制行为：为何低遗忘信号没有转化为晋级

INT8目标的before阶段90个匿名类行中，`alpha`均值1.885%、最大13.402%、48行精确回退D62；final阶段165行中均值2.906%、最大21.673%、82行精确回退D62。仅作事后诊断时，final旧类`alpha`均值3.614%、新类均值2.058%；公式本身没有读取old/new角色。

支持内final一对多风险从D62专家的0.532406降至堆叠后的0.524892，下降1.41%；D65专家自身风险为4.139319，约为D62的7.78倍。因此闭式解只允许很小的D65贡献。该小贡献在outer上把D62的after提高0.56pp、forget降低0.56pp，却同时使new下降1.33pp、H下降0.47pp、new→old增加3次，且15个row中5个预测哈希发生变化。结论是support-held平方风险与outer旧/新联合决策仍不一致，连续行堆叠没有获得可泛化的无交换改善。

这也回答“是否利用地面压缩旧类原型”：D67按预注册**没有**读取ground组件。D65的低遗忘来自冻结Stage2-B旧类决策几何后追加新类，不来自地面原型；D66才是实际读取84个ground int8聚合cell、14个cell/旧类的版本。D66得到B/A/N/H/F=93.33/83.33/83.33/82.59/10.00，仍未复现D65的6.11pp遗忘，并把新类floor交换掉。因此不能把D65的低遗忘归因于地面压缩特征，也不能在D67中无证据强行加入D66组件。

### 8.4相邻版本同排比较与目标缺口

| 版本 | B | A | N | H | F | J | min B/A/N | 混淆 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|D62|92.78|82.22|84.67|82.62|10.56|26.67|80.00/53.33/73.33|23/8/15|当前联合最强|
|D65|92.22|86.11|59.33|67.12|6.11|16.67|80.00/70.00/46.67|16/28/33|旧类保持强，新类失败|
|D66|93.33|83.33|83.33|82.59|10.00|23.33|80.00/53.33/66.67|20/9/16|ground真实使用但负向|
|D67|92.78|82.78|83.33|82.16|10.00|26.67|80.00/53.33/73.33|22/11/14|交换伤害，不晋级|

D67相对D66为B−0.56pp、A−0.56pp、N持平、H−0.44pp、F持平、J+3.33pp、min-new+6.67pp，但三类混淆变化为+2/+2/−2；仍非支配D66。相对K10/new5开发门，A距离92%尚差9.22pp，min-after距离88%尚差34.67pp，N距离92%尚差8.67pp。不得进入第二seed或125矩阵。

### 8.5量化、训练、资源与artifact

- 最终验证显式加载Conda PowerShell hook后确认解释器为`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`；D42–D67完整测试链315/315通过，用时80.9s。pytest进程exit0后仅在清理`pytest-current`临时链接时出现Windows权限告警，不影响测试结论或项目artifact。D67专属汇总器`py_compile`和真实105行摘要生成均通过。
- INT8与matched FP32的before outer、final outer、before support、final support argmax变化均为0，margin sign flip为0；最大分数绝对误差均值0.001389、最大0.002015。量化不是本轮缺陷来源。
- 20epoch support训练中，平均loss从epoch1的1.031996降至epoch20的0.102685，support accuracy从95.14%升至100%，gradient norm从1.083757降至0.135354；20个epoch的完整loss/CE/anchor/gradient/accuracy账均在摘要JSON，query rows始终为0。
- 每个目标row计306次LDA闭式fit，其中D67 inner D62为224次、D65 covariance expert为10次；D67新增适配MAC为101,525,087,053，总适配MAC为126,416,311,023。query为6,624MAC/样本且D67额外query MAC为0；参数2,016，最终持久state8,583B，registry941B，峰值CUDA显存22,886,912B，D67额外持久state为0。
- 真实runner用时391.7147s；最终artifact为7个基础文件＋1个PostRun元数据，另生成120,083B完整性能摘要。关键SHA：training log`30e6fdf0...e1430`、metadata`2632a85f...16112`、summary`6c8349ae...60592`、receipt`d2e4eeab...97b6d`、support`66e07db5...7de67`、geometry`ae4b735a...300dc`、resource`00f364e5...d6e2b`、selection`bc79d229...ade80`。

## 9.最终判定与下一轮边界

D67完成且协议、量化、资源、artifact均闭环，但性能为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。D62继续作为当前联合最强开发基座。按预注册停止D62/D65连续行堆叠路线，不扫描alpha、fold、温度、ridge、阈值，不运行第二seed或125。

下一候选必须直接处理已定位的主要矛盾：旧类遗忘集中在少数旧类，但任何保护规则仍须类身份无关；ground int8只能作为不可更新的Phase1聚合知识，不能通过角色名单定向注入；新候选必须用support-only的统一证据同时验证before/after registration、seen-new、H、逐类旧准确率和forgetting。D67专属汇总器`code/scripts/summarize_d67_performance.py`已生成完整证据账，后续报告不得用单项最大值替代同row比较。

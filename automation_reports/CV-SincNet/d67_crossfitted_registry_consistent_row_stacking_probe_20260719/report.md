# D67交叉拟合registry-consistent连续行堆叠探针

## 1.执行前登记

- 实验ID：`d67_crossfitted_registry_consistent_row_stacking_probe_20260719`；operator：Codex；状态：`PREREGISTERED_IMPLEMENTATION_PENDING`。
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

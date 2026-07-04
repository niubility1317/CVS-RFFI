# phase2_adv3b02_opu_ci_20260704

## 基本信息

| 字段 | 内容 |
|---|---|
| 实验ID | phase2_adv3b02_opu_ci_20260704 |
| 时间 | 2026-07-04 |
| 操作方 | Codex |
| 目标 | 在ADV3B02_CORE90_SOFT_E200+qknn8上评估OPU-CI（old-protected unknown confirmation）卫星群协同推理 |
| 基础特征 | `remote_artifacts\phase2_adv3b02_features\features.npz` |
| 状态 | N607_COMPLETED_NEGATIVE_DIAGNOSTIC |

## 算法设计

OPU-CI针对单协同推理不足的问题，把每个接收机事件分成两层证据：

1. known route：保留qknn8的old/seen-new候选、score、margin与class evidence。
2. safety route：只基于support envelope、class conformal、virtual unknown、class negative、class shell生成unknown风险，不使用unknown query调阈值。
3. fusion：采用`old_protected_unknown_confirm_cvs`。强known支持时优先保护old/seen-new；弱known且多源unknown证据成立时拒识；证据不足且延迟预算允许时请求更多接收机，否则defer。

双路风险：

```text
score_discount = max(0, 1 - known_score / score_anchor)
margin_discount = max(0, 1 - known_margin / margin_anchor)
dual_unknown_risk = max(
  known_route_unknown_risk,
  safety_weight * safety_route_unknown_risk * discount
)
```

OPU-CI新增的可部署策略：

| 策略 | 目的 |
|---|---|
| `opu_old_preserve` | 优先保持旧类准确率，允许温和unknown拒识增益 |
| `opu_old_guarded` | 更积极使用跨接收机unknown确认 |
| `opu_balanced` | 平衡策略，当前过严 |
| `opu_unknown_strict` | unknown严格策略，当前打穿known，仅作负诊断 |

## 本地变更

| 文件 | 作用 | SHA256 |
|---|---|---|
| `E:\type10-7\code\scripts\phase2_old_protected_unknown_confirm_ci_eval.py` | OPU-CI评估脚本，输出1..N协同摘要 | `F8A66E7260BA535EFB55F70417F61823A374D3D1ADEA32F19EB2D421D8533872` |
| `E:\type10-7\code\tests\test_phase2_old_protected_unknown_confirm_ci_eval.py` | OPU-CI策略解析与打分单测 | `241FA059DE02902791EBB377CECE04BAFFD469C13760D532DDA44848123EAC9F` |
| `E:\type10-7\code\snapshots\phase2_adv3b02_opu_ci_20260704\phase2_old_protected_unknown_confirm_ci_eval.py` | 非Git根代码快照 | 待同步 |
| `E:\type10-7\code\snapshots\phase2_adv3b02_opu_ci_20260704\test_phase2_old_protected_unknown_confirm_ci_eval.py` | 非Git根测试快照 | 待同步 |

## 本地验证

| 命令 | 结果 |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_old_protected_unknown_confirm_ci_eval.py code\tests\test_phase2_old_protected_unknown_confirm_ci_eval.py` | PASS |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_old_protected_unknown_confirm_ci_eval.py -q` | PASS，3 passed；`.pytest_cache`权限警告不影响测试 |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe code\scripts\phase2_old_protected_unknown_confirm_ci_eval.py --feature_npz remote_artifacts\phase2_adv3b02_features\features.npz --output_dir local_artifacts\phase2_adv3b02_opu_ci_20260704 --force --policies all` | PASS，20 rows |

本地输出：

| 文件 | 内容 | SHA256 |
|---|---|---|
| `E:\type10-7\local_artifacts\phase2_adv3b02_opu_ci_20260704\opu_ci_summary.csv` | OPU-CI 1..5协同摘要 | `94B0613E3E7162BF439FB95A8FC414CEBC283A0AAA73B6C49B885CA813BA8643` |
| `E:\type10-7\local_artifacts\phase2_adv3b02_opu_ci_20260704\opu_ci_summary.json` | 完整结果JSON | 待归档 |
| `E:\type10-7\local_artifacts\phase2_adv3b02_opu_ci_20260704\opu_ci_evidence.csv` | 合并双路evidence | 待归档 |

## 本地结果

| policy | k | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | defer | bytes/event | latency_ms_p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| opu_old_guarded | 4 | 0.7701 | 0.3000 | 0.6667 | 0.6500 | 0.2500 | 0.5500 | 0.1042 | 496.3 | 0.1585 |
| opu_old_guarded | 3 | 0.6310 | 0.2000 | 0.6500 | 0.6250 | 0.2667 | 0.5000 | 0.0847 | 414.3 | 0.1585 |
| opu_old_preserve | 4 | 0.8021 | 0.3000 | 0.7833 | 0.7250 | 0.1833 | 0.7500 | 0.0423 | 496.3 | 0.1585 |
| opu_old_preserve | 3 | 0.7594 | 0.2000 | 0.7167 | 0.7000 | 0.1167 | 0.5833 | 0.0423 | 414.3 | 0.1585 |
| opu_old_guarded | 5 | 0.7647 | 0.2500 | 0.6667 | 0.6500 | 0.2000 | 0.5667 | 0.1792 | 547.2 | 0.1585 |
| opu_old_preserve | 5 | 0.7968 | 0.2500 | 0.7833 | 0.7250 | 0.0833 | 0.7500 | 0.0619 | 547.2 | 0.1585 |

## 本地解释

OPU-CI相对SO-CAPR双路折扣有两个明确结论：

1. `opu_old_preserve,k=4`在`old_acc=0.8021`时把unknown拒识提升到0.1833；同等旧类水平附近的SO-CAPR`threshold=0.4,k=1`为`old_acc=0.8021,unknown_reject=0.0667`。这说明跨接收机unknown确认有实际增益。
2. 该路线仍远低于目标：`old_acc 99%/min_old 95%/seen_new 97%/min_seen 93%/unknown_reject 99%`均未满足，尤其`unknown_FAR`仍高。不能写成部署成功，只能作为新算法负诊断与下一步训练目标。

下一步需要从“后处理拒识”转向“训练阶段塑造open-set表征”：source/proxy unknown或support-only contrastive boundary应进入ADV3B02后续轻量训练或adapter训练，否则高置信unknown落入known空间的问题无法靠阈值完全解决。

## N607计划

| 字段 | 内容 |
|---|---|
| 远端根目录 | `/home/szu2070436088/2510044040/CV-SincNet` |
| 远端环境 | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python` |
| 运行GPU | 选择当前显存最低的GPU，OPU-CI为CPU/qknn评估，显存占用应接近空闲 |
| 远端输出 | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_opu_ci_20260704/` |
| 远端日志 | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_opu_ci_20260704/opu_ci.log` |
| 远端命令 | `CUDA_VISIBLE_DEVICES=<gpu> /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_old_protected_unknown_confirm_ci_eval.py --feature_npz runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz --output_dir runs/phase2_adv3b02_opu_ci_20260704 --force --policies all` |

## N607验证与运行

N607预检：

| 字段 | 内容 |
|---|---|
| 预检时间 | 2026-07-04 14:02:30 CST |
| SSH目标 | direct `N607`，配置`E:\type10-7\tools\n607_ssh_config` |
| 项目根目录 | `/home/szu2070436088/2510044040/CV-SincNet` |
| GPU状态 | 8张RTX 3090均为`10/24576 MiB`，选择GPU0 |
| 活跃进程 | 未见当前用户训练进程，仅有系统/VSCode相关Python进程 |

同步目标：

| 本地文件 | N607目标 |
|---|---|
| `E:\type10-7\code\scripts\phase2_old_protected_unknown_confirm_ci_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_old_protected_unknown_confirm_ci_eval.py` |
| `E:\type10-7\code\tests\test_phase2_old_protected_unknown_confirm_ci_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_old_protected_unknown_confirm_ci_eval.py` |
| `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_opu_ci_20260704\report.md` | `/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/phase2_adv3b02_opu_ci_20260704/report.md` |

远端验证命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
PY=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
CUDA_VISIBLE_DEVICES=0 $PY -m py_compile code/scripts/phase2_old_protected_unknown_confirm_ci_eval.py code/tests/test_phase2_old_protected_unknown_confirm_ci_eval.py
CUDA_VISIBLE_DEVICES=0 $PY code/tests/test_phase2_old_protected_unknown_confirm_ci_eval.py
CUDA_VISIBLE_DEVICES=0 $PY code/scripts/phase2_old_protected_unknown_confirm_ci_eval.py --feature_npz runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz --output_dir runs/phase2_adv3b02_opu_ci_20260704 --force --policies all
```

异常记录：第一次远端验证命令因PowerShell未正确转义远端`$PY`，在进入Python前失败为`bash: line 1:  -m: command not found`；SSH清理后已用正确转义重跑成功。

远端输出：

| 文件 | 本地归档 | SHA256 |
|---|---|---|
| `runs/phase2_adv3b02_opu_ci_20260704/opu_ci_summary.csv` | `E:\type10-7\local_artifacts\phase2_adv3b02_opu_ci_20260704\remote\opu_ci_summary.csv` | `DE2A643C2AAA280B7DFF60718621F24FC505E94261757B96FE6F8C4CB697F06D` |
| `runs/phase2_adv3b02_opu_ci_20260704/opu_ci_summary.json` | `E:\type10-7\local_artifacts\phase2_adv3b02_opu_ci_20260704\remote\opu_ci_summary.json` | `D31A7A913B71BE420AFE24E24D3856DF04CC36A4C2A9B1AEB214E3A26E37AD8C` |
| `logs/phase2_adv3b02_opu_ci_20260704/opu_ci.log` | `E:\type10-7\local_artifacts\phase2_adv3b02_opu_ci_20260704\remote\opu_ci.log` | `9261C44FCA13BA05EABABA1832CC2879A0E55FE25C1D6CA4D878E8319F3B5CBC` |

远端结果：

| policy | k | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | request_more | defer | bytes/event | latency_ms_p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| opu_old_guarded | 4 | 0.7701 | 0.3000 | 0.6667 | 0.6500 | 0.2500 | 0.5500 | 0.0749 | 0.1042 | 496.3 | 0.2489 |
| opu_old_guarded | 3 | 0.6310 | 0.2000 | 0.6500 | 0.6250 | 0.2667 | 0.5000 | 0.1922 | 0.0847 | 414.3 | 0.2489 |
| opu_old_preserve | 4 | 0.8021 | 0.3000 | 0.7833 | 0.7250 | 0.1833 | 0.7500 | 0.0000 | 0.0423 | 496.3 | 0.2489 |
| opu_old_preserve | 3 | 0.7594 | 0.2000 | 0.7167 | 0.7000 | 0.1167 | 0.5833 | 0.0912 | 0.0423 | 414.3 | 0.2489 |
| opu_old_guarded | 5 | 0.7647 | 0.2500 | 0.6667 | 0.6500 | 0.2000 | 0.5667 | 0.0000 | 0.1792 | 547.2 | 0.2489 |
| opu_old_preserve | 5 | 0.7968 | 0.2500 | 0.7833 | 0.7250 | 0.0833 | 0.7500 | 0.0000 | 0.0619 | 547.2 | 0.2489 |

远端运行后GPU仍为8张RTX 3090均`10/24576 MiB`。SCP/SSH后本地`ssh.exe`、到`172.31.111.215:22`和`172.31.105.18:22`的`ESTABLISHED`连接均为空。

## 最终解释

OPU-CI验证了“旧类保护优先的跨接收机unknown确认”比单路/折扣门控更合理，但仍不是达标解：

1. 最保旧的可比点是`opu_old_preserve,k=4`：`old_acc=0.8021,seen_new_acc=0.7833,unknown_reject=0.1833`。相较SO-CAPR同old水平附近的`unknown_reject=0.0667`有提升，但FAR仍为0.75。
2. 最强unknown折中是`opu_old_guarded,k=4`：`unknown_reject=0.25,FAR=0.55`，但`old_acc`降至0.7701，不能满足“旧类准确性不能下降”的主约束。
3. `opu_unknown_strict`和`opu_balanced`会把known几乎全部打穿，属于明确负证据。

结论：后处理协同推理可以提高unknown拒识，但当前ADV3B02冻结表征中的unknown与known重叠严重。下一步应进入表征训练/轻量adapter阶段：在不使用unknown query调阈值的前提下，用source proxy-unknown、support-only虚拟边界和old-class retention loss训练open-set margin，使高置信unknown不再落入old/seen-new候选空间。

## 子agent审查结论

| 角色 | 结论 |
|---|---|
| 文献/方法 | 优先路线应为类级原型+EVT/GPD、energy/OOD校准、Mahalanobis few-shot open-set、OpenMax/MetaMax和proxy_unknown outlier exposure；这些方法都可保持星上推理轻量化，并避免使用unknown query拟合阈值。 |
| 高效率算法 | 建议下一步从OPU-CI升级为feature-level轻量adapter：冻结ADV3B02主干，只训练低秩adapter/temperature/prototype边界，损失包含旧类CE、seen-new CE、旧类蒸馏、proxy-open约束和adapter幅度正则。 |
| 完成监督 | 本轮已完成本地实现、1..5协同、本地测试、N607同步、CVS-RFFI远端运行、报告和Git提交；但最终目标未达成，仍是negative diagnostic。 |
| 查漏补缺review | 必须避免把`receiver_domain_ranked`夸大为真实same-event卫星群同步协同；当前结果只能写成deployment proxy ensemble。后续adapter训练必须硬性断言`target_unknown`不进入loss、BN/affine统计、阈值、早停或profile选择。 |

后续算法验收硬门槛：

```text
target_unknown_eval_only = true
unknown_query_used_for_threshold_fit = false
old_acc_adapter >= old_acc_base - epsilon
collab_count covers 1..receiver_count
same-row metrics required for every claim
```

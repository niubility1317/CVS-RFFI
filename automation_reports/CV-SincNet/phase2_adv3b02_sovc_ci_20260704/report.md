# phase2_adv3b02_sovc_ci_20260704

## 基本信息

- 时间：2026-07-04
- 操作员/agent：Codex
- 目标：在ADV3B02_CORE90_SOFT_E200特征上实现并验证SOVC-CI协同推理，用support-side open-set verifier优先改善未知类拒识，同时检查旧类准确率是否下降。
- 场景：Stage2-C，target receiver domain与source receiver domain不相交；target-old、seen-new、unknown TX互斥；unknown query仅用于评估，不参与阈值拟合。
- 对照：ORBIT-C3R与PCET-CI qknn8协同推理结果。
- 权重/特征来源：`SA33_sa27_ch2_leo3_ce0p7_r010_20260527_204104`对应的`ADV3B02_CORE90_SOFT_E200`特征包，输入`E:\type10-7\remote_artifacts\phase2_adv3b02_features\features.npz`。

## 算法设计

SOVC-CI使用PCET-CI生成的qknn8协同证据作为底座，再叠加轻量source/support-side open-set verifier calibration。每个receiver只上传固定标量证据，包含top-1 raw score、verified score、second verified score、conformal pvalue、receiver-class reliability、class negative risk、class shell risk和verifier changed标志。

核心风险：

```text
sovc_verifier_risk =
  0.23 * verified_score_drop_risk
+ 0.18 * verified_margin_risk
+ 0.16 * (1 - verifier_pvalue)
+ 0.16 * (1 - receiver_class_reliability)
+ 0.12 * verifier_unknown_risk
+ 0.08 * class_negative_risk
+ 0.04 * class_shell_risk
+ 0.03 * verifier_changed
```

最终风险：

```text
sovc_unknown_risk = max(base_unknown_risk,
                        sovc_base_weight * base_unknown_risk
                      + sovc_verifier_weight * sovc_verifier_risk)
```

若验证器显示强旧类证据，则启用safe-known cap，避免把高置信旧类误拒识。该设计不训练主干、不使用unknown query校准，适合星上推理、原型更新和阈值微调边界；但当前实现仍是诊断性后处理，不是已达标部署算法。

## 本地变更

| 文件 | 作用 |
|---|---|
| `E:\type10-7\code\scripts\phase2_orbit_sovc_ci_eval.py` | 新增SOVC-CI协同open-set评估脚本，支持协同数量从1到目标接收机可用上限。 |
| `E:\type10-7\code\tests\test_phase2_orbit_sovc_ci_eval.py` | 新增SOVC风险、safe-known cap和Stage2协议评估单元测试。 |
| `E:\type10-7\code\snapshots\phase2_adv3b02_sovc_ci_20260704\` | 根目录非Git仓库时的本地同步前快照。 |
| `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_sovc_ci_20260704\report.md` | 本报告。 |

根目录`E:\type10-7`不是Git仓库，Git-backed镜像在`E:\type10-7\github_publish\CVS-RFFI-repo`，当前分支为`codex/cvs-rffi-release-20260626`。

## 本地验证

| 命令 | 结果 |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_orbit_sovc_ci_eval.py code\tests\test_phase2_orbit_sovc_ci_eval.py` | PASS |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_orbit_sovc_ci_eval.py -q` | PASS，3 passed；`.pytest_cache`写入被拒绝的warning不影响测试结论。 |
| `phase2_orbit_sovc_ci_eval.py --feature_npz ... --collab_counts all --collab_group_policy available_up_to_k --k_shot 8 --qknn_k 8` | PASS，生成summary/evidence/json。 |
| `phase2_orbit_c3r_failure_audit.py --input_json ...\sovc_ci.json` | PASS，`floor_failure_count=67`，`no_event_count=0`，无schema错误。 |

## 本地结果摘要

目标门槛：`old_acc>=0.99`，`min_old>=0.95`，`seen_new_acc>=0.97`，`min_seen>=0.93`，`unknown_reject>=0.99`。

| profile | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | bytes/event | resource_pass | target_pass | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| sovc_known_preserving | 1 | 0.6825 | 0.0000 | 0.1500 | 0.0000 | 0.4333 | 0.5667 | 128.0 | true | false | 保留最高old均值，但unknown不足。 |
| sovc_known_preserving | 4 | 0.6296 | 0.4000 | 0.4833 | 0.3750 | 0.5333 | 0.4333 | 376.5 | true | false | seen-new较高，但unknown和old floor不足。 |
| sovc_known_preserving | 5 | 0.5714 | 0.3000 | 0.4333 | 0.3750 | 0.5833 | 0.3667 | 414.2 | true | false | 与PCET known-preserving一致，未解决unknown。 |
| sovc_old_safe | 5 | 0.4868 | 0.3000 | 0.4333 | 0.3750 | 0.6833 | 0.3167 | 414.2 | true | false | unknown提升有限，但旧类下降，不可作为主线。 |
| sovc_balanced | 4 | 0.1376 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 376.5 | true | false | unknown达标但旧类/seen-new塌陷，违反约束。 |
| sovc_unknown_strict | 4 | 0.1323 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 376.5 | true | false | 严格拒识诊断，不可部署。 |

## 监督审查

合理性审查：

- SOVC-CI没有使用unknown query拟合阈值，符合Stage2-C边界。
- 协同数量使用`available_up_to_k`覆盖1到目标接收机可用上限，符合用户要求。
- 所有结果均满足资源约束，但性能未达目标，不能写成部署成功。
- 当前后处理式拒识存在根本矛盾：强拒识可以压低unknown FAR，但会拒掉旧类与seen-new；宽松旧类保护保留旧类均值，但unknown FAR仍高。

查漏补缺：

- 仅用后验阈值/风险融合无法满足“unknown优先且old不下降”。下一步应把unknown拒识前移到源侧训练与open-set episode verifier，而不是继续调SOVC阈值。
- 建议实现训练期轻量Siamese/Proto-verifier或energy-regularized source open-set head：源域构造pseudo-unknown episode、receiver-held-out episode和satellite-channel hard negatives，使旧类prototype边界在训练时就可分，而不是部署后临时切阈值。
- 星上只部署冻结backbone、prototype bank、verifier head和少量校准参数；允许更新target-old/seen-new prototype、temperature/bias/threshold，不进行full-model fine-tuning。

## N607计划

- 远端工作目录：`/home/szu2070436088/2510044040/CV-SincNet`
- 远端环境：`CVS-RFFI`
- 远端同步：
  - `E:\type10-7\code\scripts\phase2_orbit_sovc_ci_eval.py` -> `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_orbit_sovc_ci_eval.py`
  - `E:\type10-7\code\tests\test_phase2_orbit_sovc_ci_eval.py` -> `/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_orbit_sovc_ci_eval.py`
- 远端输出：`runs/phase2_adv3b02_sovc_ci_20260704/`
- GPU策略：选择显存占用最低GPU；该任务只做特征级推理评估，预期显存占用约10MiB。

## N607执行结果

远端preflight通过：直连`N607`可用，项目根目录存在，8张RTX 3090均约10MiB显存占用。实际使用GPU0，执行后`nvidia-smi --query-gpu=index,memory.used`显示GPU0到GPU7均为10MiB。

| 远端命令/步骤 | 结果 |
|---|---|
| `py_compile code/scripts/phase2_orbit_sovc_ci_eval.py code/tests/test_phase2_orbit_sovc_ci_eval.py` | PASS |
| `PYTHONPATH=code:code/scripts CUDA_VISIBLE_DEVICES=0 ... test_phase2_orbit_sovc_ci_eval.py` | PASS，3 tests OK |
| `CUDA_VISIBLE_DEVICES=0 ... phase2_orbit_sovc_ci_eval.py --collab_counts all --collab_group_policy available_up_to_k` | PASS |
| `CUDA_VISIBLE_DEVICES=0 ... phase2_orbit_c3r_failure_audit.py --input_json .../sovc_ci.json` | PASS，`floor_failure_count=67`，`no_event_count=0`，无schema错误 |

远端结果已拉回：

| 远端文件 | 本地归档 |
|---|---|
| `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_sovc_ci_20260704/sovc_ci.json` | `E:\type10-7\local_artifacts\phase2_adv3b02_sovc_ci_20260704\remote\sovc_ci.json` |
| `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_sovc_ci_20260704/sovc_ci_summary.csv` | `E:\type10-7\local_artifacts\phase2_adv3b02_sovc_ci_20260704\remote\sovc_ci_summary.csv` |
| `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_sovc_ci_20260704/sovc_ci_failure_audit.json` | `E:\type10-7\local_artifacts\phase2_adv3b02_sovc_ci_20260704\remote\sovc_ci_failure_audit.json` |
| `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_sovc_ci_20260704/sovc_ci_class_floor.csv` | `E:\type10-7\local_artifacts\phase2_adv3b02_sovc_ci_20260704\remote\sovc_ci_class_floor.csv` |
| `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_sovc_ci_20260704/sovc_ci_confusion.csv` | `E:\type10-7\local_artifacts\phase2_adv3b02_sovc_ci_20260704\remote\sovc_ci_confusion.csv` |

SSH/SCP后均检查本地`ssh.exe`进程和到`172.31.111.215:22`的ESTABLISHED连接，未发现残留连接。

## 当前结论

SOVC-CI模块实现和本地全量评估完成，但没有达到目标。当前最重要证据是：在现有ADV3B02特征上，unknown拒识与旧类保留呈强冲突；满足unknown_reject=1.0的profile会把old_acc压到0.11到0.14区间。因此，下一阶段不应继续后验阈值搜索，应转向源侧open-set episode训练/轻量verifier head，先提升特征空间中unknown与old/seen-new边界的可分性。

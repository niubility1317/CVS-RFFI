# phase2_adv3b02_orbit_c3r_guard_20260704

## 基本信息

| 项目 | 内容 |
|---|---|
| 实验ID | phase2_adv3b02_orbit_c3r_guard_20260704 |
| 时间 | 2026-07-04 |
| 操作方 | Codex |
| 目标 | 在`ADV3B02_CORE90_SOFT_E200`特征和在轨`qknn8`少样本old/new support基础上，实现可部署的卫星群协同开集拒识ORBIT-C3R Guard，并报告`collab_count=1..N`、时延、通信和目标差距 |
| 协议边界 | Stage2-C；`Y_unknown`只用于query评估，不参与support、阈值拟合或receiver选择监督 |
| 输入feature | `E:\type10-7\remote_artifacts\phase2_adv3b02_features\features.npz` |
| 本地输出 | `E:\type10-7\local_artifacts\phase2_adv3b02_orbit_c3r_guard_20260704\` |
| 协同范围 | `collab_count=1..5`；区别于`K-shot=8` |
| 资源预算 | 128B/receiver/event，默认最大1152B/event，最大20ms/event |

## 本地改动

| 文件 | 目的 |
|---|---|
| `E:\type10-7\code\scripts\phase2_orbit_c3r_guard_eval.py` | 新增ORBIT-C3R部署式协同评估封装：生成qknn8证据，运行old-preserving、old-guarded、balanced、unknown-strict四档profile，输出目标差距和资源字段 |
| `E:\type10-7\code\tests\test_phase2_orbit_c3r_guard_eval.py` | 验证profile输出、`collab_count=1..N`、资源字段、`unknown_query_eval_only=True`和非正shot拒绝 |

版本状态：`E:\type10-7`不是Git仓库；将创建快照并同步到Git-backed镜像`E:\type10-7\github_publish\CVS-RFFI-repo`。

## 本地验证

| 命令 | 结果 |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_orbit_c3r_guard_eval.py code\tests\test_phase2_orbit_c3r_guard_eval.py` | PASS |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_orbit_c3r_guard_eval.py -q` | PASS，2 passed；根目录pytest cache写入被Windows拒绝，不影响测试结果 |

## 算法配置

ORBIT-C3R Guard当前实现为部署层封装，不改`项目.md`协议：

```text
base feature: ADV3B02 z_id
in-orbit support: target old + seen-new K-shot support
local learner: qknn8
unknown calibration: support-generated virtual/class-negative/class-shell risk only
unknown query: evaluation-only
fusion: SCG qknn evidence guard with support-quality receiver selection
```

四档profile：

| profile | 用途 |
|---|---|
| old_preserving | 宽松support-confirmed known基线，先观察旧类/新类保护能力 |
| old_guarded | 旧类保护+强多源unknown风险拒识 |
| balanced | 折中旧类/新类接受与unknown共识拒识 |
| unknown_strict | 严格unknown安全档，用于暴露旧类/新类损伤 |

## 本地结果

| profile | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer | request_more | bytes/event | latency_ms | target_pass | resource_pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| old_preserving | 1 | 0.6720 | 0.0000 | 0.1500 | 0.0000 | 0.4333 | 0.5667 | 0.0000 | 0.0000 | 128.0 | 0.0 | false | true |
| old_preserving | 2 | 0.6078 | 0.3500 | 0.4898 | 0.4138 | 0.1250 | 0.4792 | 0.0000 | 0.3750 | 256.0 | 0.0 | false | true |
| old_preserving | 3 | 0.5417 | 0.0000 | 0.5250 | 0.5000 | 0.4500 | 0.3000 | 0.0000 | 0.2500 | 384.0 | 0.0 | false | true |
| old_preserving | 4 | 0.6092 | 0.0000 | 0.6129 | 0.4545 | 0.5938 | 0.3438 | 0.0000 | 0.0625 | 512.0 | 0.0 | false | true |
| old_preserving | 5 | 0.5294 | 0.0000 | 0.5500 | 0.0000 | 0.8000 | 0.0500 | 0.0000 | 0.0000 | 640.0 | 0.0 | false | true |
| old_guarded | 1 | 0.3175 | 0.0000 | 0.0500 | 0.0000 | 0.7500 | 0.2333 | 0.0000 | 0.0167 | 128.0 | 0.0 | false | true |
| old_guarded | 2 | 0.1307 | 0.0000 | 0.0000 | 0.0000 | 0.7083 | 0.0417 | 0.0000 | 0.2500 | 256.0 | 0.0 | false | true |
| old_guarded | 3 | 0.1583 | 0.0000 | 0.0000 | 0.0000 | 0.8750 | 0.0000 | 0.0000 | 0.1250 | 384.0 | 0.0 | false | true |
| old_guarded | 4 | 0.0460 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 512.0 | 0.0 | false | true |
| old_guarded | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 640.0 | 0.0 | false | true |
| balanced | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 640.0 | 0.0 | false | true |
| unknown_strict | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 640.0 | 0.0 | false | true |

## 解释

本地结果表明ORBIT-C3R部署层封装满足通信资源预算，但不能解决目标问题。`old_preserving`在`collab_count=5`可把`unknown_FAR`降到0.05，但`old_acc=0.5294`、`min_old=0.0000`，旧类性能严重不合格。更严格profile可达到`unknown_reject=1.0000`，但旧类和新类识别被全部拒掉或错误路由，不能作为候选路线。

该结果与oracle unknown holdout负证据一致：当前ADV3B02`z_id`下，部署层拒识门控会在unknown与old/seen-new之间产生强冲突。下一步不能继续只调门控，应回到地面表示学习，加入source/proxy outlier exposure、energy/open-space margin、receiver-invariant identity约束或旧类类地板约束，再回到ORBIT-C3R做部署层验证。

## N607执行

远端工作目录：`/home/szu2070436088/2510044040/CV-SincNet`
远端环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
远端输出：`runs/phase2_adv3b02_orbit_c3r_guard_20260704/`
拉回目录：`E:\type10-7\local_artifacts\phase2_adv3b02_orbit_c3r_guard_20260704\remote\`

N607只读preflight通过：直接`N607`目标、项目根目录和8张RTX 3090均可见。运行前GPU占用均为`10MiB`，未发现本用户训练进程；选择低占用GPU0执行诊断。运行结束后`nvidia-smi`显示8张GPU仍为`10MiB`。

同步文件：

| 本地 | 远端 |
|---|---|
| `code\scripts\phase2_orbit_c3r_guard_eval.py` | `code/scripts/phase2_orbit_c3r_guard_eval.py` |
| `code\tests\test_phase2_orbit_c3r_guard_eval.py` | `code/tests/test_phase2_orbit_c3r_guard_eval.py` |
| `remote_artifacts\phase2_adv3b02_features\features.npz` | `runs/phase2_adv3b02_orbit_c3r_guard_20260704/input/features.npz` |

远端验证：

| 命令 | 结果 |
|---|---|
| `py_compile` ORBIT脚本和测试 | PASS |
| `PYTHONPATH=code:code/scripts ... code/tests/test_phase2_orbit_c3r_guard_eval.py` | PASS，2 tests OK；负值shot测试产生预期argparse错误文本 |
| ORBIT-C3R全profile全`collab_count=1..5`评估 | PASS，输出JSON/CSV |

远端结果与本地一致：

| profile | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer | request_more | bytes/event | latency_ms | target_pass | resource_pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| old_preserving | 1 | 0.6720 | 0.0000 | 0.1500 | 0.0000 | 0.4333 | 0.5667 | 0.0000 | 0.0000 | 128.0 | 0.0 | false | true |
| old_preserving | 2 | 0.6078 | 0.3500 | 0.4898 | 0.4138 | 0.1250 | 0.4792 | 0.0000 | 0.3750 | 256.0 | 0.0 | false | true |
| old_preserving | 3 | 0.5417 | 0.0000 | 0.5250 | 0.5000 | 0.4500 | 0.3000 | 0.0000 | 0.2500 | 384.0 | 0.0 | false | true |
| old_preserving | 4 | 0.6092 | 0.0000 | 0.6129 | 0.4545 | 0.5938 | 0.3438 | 0.0000 | 0.0625 | 512.0 | 0.0 | false | true |
| old_preserving | 5 | 0.5294 | 0.0000 | 0.5500 | 0.0000 | 0.8000 | 0.0500 | 0.0000 | 0.0000 | 640.0 | 0.0 | false | true |
| old_guarded | 3 | 0.1583 | 0.0000 | 0.0000 | 0.0000 | 0.8750 | 0.0000 | 0.0000 | 0.1250 | 384.0 | 0.0 | false | true |
| old_guarded | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 640.0 | 0.0 | false | true |
| balanced | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 640.0 | 0.0 | false | true |
| unknown_strict | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 640.0 | 0.0 | false | true |

SSH/SCP清理：preflight、进程检查、同步、运行和结果拉回之后均检查本地`ssh.exe`与到`172.31.111.215:22`/`172.31.105.18:22`的`ESTABLISHED`连接；结果均为`none`。

最终判定：本轮ORBIT-C3R实现是可部署资源约束下的诊断/候选评估工具，但当前ADV3B02+qknn8表示无法达到目标。所有profile的`target_pass=false`，因此不能登记为Stage2-C成功或部署证据。

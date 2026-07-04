# phase2_adv3b02_profile_guarded_opr_opu_ci_20260704

## 基本信息

- 时间：2026-07-04
- 操作方：Codex
- 目标：为天基RFFI卫星群协同推理增加profile-guarded OPR-OPU-CI诊断路线，在不使用target_unknown做训练、阈值拟合或profile选择的前提下，引入source/support known保真门与source-side proxy_unknown代理拒识门；若adapter破坏known或代理拒识安全性，则回滚到base qknn8/OPU-CI。
- 对比目标：前序OPU-CI在高known特征上unknown不足；前序OPR-OPU在proxy_unknown特征上unknown较高但known被打穿。本实验检查是否能用硬保真选择器阻止这种跷跷板。

## 算法设计

新增`profile_guarded_OPR-OPU-CI`：

1. 强制包含`base`候选，不训练adapter。
2. 对`conservative/known_tight/open_light`等轻量低秩残差adapter候选，仅使用`source`、source-side`proxy_unknown`、`target_old_support`、`target_new_support`训练；`target_unknown_training_count=0`。
3. profile选择阶段只读取：
   - source/support known原型准确率与下降量；
   - source/support adapter residual drift；
   - source-side proxy_unknown最大logit下降等query-free拒识代理。
4. profile选择阶段不读取`unknown_reject_rate`、`unknown_FAR`、AUROC、FPR95或任何target_unknown query统计。
5. 最终OPU-CI评估仍报告`old_acc`、`seen_new_acc`、`unknown_reject_rate`、`unknown_FAR`、coverage、defer和协同接收机数量；这些只作为评估结果，不参与profile选择。

## 本地变更

| 文件 | 目的 |
|---|---|
| `code/scripts/phase2_profile_guarded_opr_opu_ci_eval.py` | 新增profile-guarded OPR-OPU-CI封装、known/proxy-surrogate硬门、base回滚、OPU-CI同row输出。 |
| `code/tests/test_phase2_profile_guarded_opr_opu_ci_eval.py` | 覆盖默认profile、proxy_unknown代理恶化拒绝、target_unknown训练泄漏拒绝、base回滚、guard score排序。 |

Git承载目录：`E:\type10-7\github_publish\CVS-RFFI-repo`。

## 本地验证

| 命令 | 结果 |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_profile_guarded_opr_opu_ci_eval.py code\tests\test_phase2_profile_guarded_opr_opu_ci_eval.py` | PASS |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_profile_guarded_opr_opu_ci_eval.py -q` | PASS，5 passed |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_profile_guarded_opr_opu_ci_eval.py code\tests\test_phase2_opr_opu_adapter_ci_eval.py -q` | PASS，6 passed；`.pytest_cache`写入权限警告，不影响测试 |
| local smoke：`phase2_profile_guarded_opr_opu_ci_eval.py --feature_npz local_artifacts\phase2_adv3b02_proxy_unknown_ci_20260704\remote\features_proxy_unknown.npz --output_dir local_artifacts\phase2_adv3b02_profile_guarded_opr_opu_ci_20260704\local_smoke_v2 --device cpu --profiles base,conservative --policies opu_old_preserve,opu_old_guarded --adapter_epochs 2 --k_shot 4 --query_per_class 10 --max_event_latency_ms 20.0` | PASS，输出20行OPU summary，选择`base` |

## 本地短流程结果

Profile guard：

| profile | guard_pass | source_proto_drop | support_proto_drop | proxy_logit_reduction | proxy_surrogate_pass | known_drift_pass | target_unknown_training_count |
|---|---:|---:|---:|---:|---|---|---:|
| base | true | 0.000000 | 0.000000 | 0.000000 | true | true | 0 |
| conservative | false | 0.000833 | -0.012500 | 1.268270 | true | false | 0 |

Selected profile：`base`；selection reason：`known_and_proxy_surrogate_guard_score`；selection metric scope：`source_support_known_plus_source_proxy_unknown_surrogate`；`target_unknown_selection_count=0`。

Selected base同row前8行：

| profile | selected | policy | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| base | true | opu_old_guarded | 4 | 0.140625 | 0.000000 | 0.000000 | 0.000000 | 0.916667 | 0.083333 |
| base | true | opu_old_preserve | 4 | 0.161458 | 0.000000 | 0.000000 | 0.000000 | 0.900000 | 0.083333 |
| base | true | opu_old_guarded | 5 | 0.140625 | 0.000000 | 0.000000 | 0.000000 | 0.916667 | 0.083333 |
| base | true | opu_old_preserve | 5 | 0.161458 | 0.000000 | 0.000000 | 0.000000 | 0.883333 | 0.083333 |
| base | true | opu_old_guarded | 3 | 0.067708 | 0.000000 | 0.000000 | 0.000000 | 0.900000 | 0.083333 |
| base | true | opu_old_preserve | 3 | 0.088542 | 0.000000 | 0.000000 | 0.000000 | 0.883333 | 0.083333 |
| base | true | opu_old_preserve | 1 | 0.109375 | 0.000000 | 0.000000 | 0.000000 | 0.883333 | 0.116667 |
| base | true | opu_old_guarded | 1 | 0.072917 | 0.000000 | 0.000000 | 0.000000 | 0.900000 | 0.100000 |

解释：本地短流程证明硬门会阻断known漂移过大的adapter。由于输入proxy feature本身known极差，当前结果仍是diagnostic-negative，不能声明Stage2-C成功。

## N607同步与测试计划

本地到远端映射：

| local | remote |
|---|---|
| `E:\type10-7\code\scripts\phase2_profile_guarded_opr_opu_ci_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_profile_guarded_opr_opu_ci_eval.py` |
| `E:\type10-7\code\tests\test_phase2_profile_guarded_opr_opu_ci_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_profile_guarded_opr_opu_ci_eval.py` |

远端环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。

远端输入：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_proxy_unknown_ci_20260704/features_proxy_unknown.npz`。

远端输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_profile_guarded_opr_opu_ci_20260704/`。

远端日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_profile_guarded_opr_opu_ci_20260704/profile_guarded_opr_opu_ci.log`。

计划命令：

```bash
CUDA_VISIBLE_DEVICES=<low-vram-gpu> /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_profile_guarded_opr_opu_ci_eval.py \
  --feature_npz runs/phase2_adv3b02_proxy_unknown_ci_20260704/features_proxy_unknown.npz \
  --output_dir runs/phase2_adv3b02_profile_guarded_opr_opu_ci_20260704 \
  --device cuda:0 \
  --profiles base,conservative,known_tight,open_light \
  --policies opu_old_preserve,opu_old_guarded \
  --adapter_epochs 20 \
  --k_shot 4 \
  --query_per_class 20 \
  --max_event_latency_ms 2.0
```

## 风险与后续

- 当前proxy feature的known基础很差，因此profile guard大概率会回滚base；这是安全行为，不是完成目标。
- 下一步真正改善unknown且旧类不降，应转向高known特征上的query-free拒识代理：高斯原型/EVT、class-wise envelope、support变换一致性、source leave-one-old-out impostor风险，再接OPU-CI协同层。
- 若远端结果仍回滚base且unknown未达标，应标记为diagnostic-negative，不能写成部署成功。

## N607完成状态

远端预检：

- 直接`N607`预检通过；项目目录可见：`/home/szu2070436088/2510044040/CV-SincNet`。
- 远端环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，Python3.10.19。
- 输入feature存在：`runs/phase2_adv3b02_proxy_unknown_ci_20260704/features_proxy_unknown.npz`，约7.1MB。
- 启动前GPU状态：GPU0约643MiB、GPU1约1481MiB已有其他任务；GPU2-7约10MiB。选择GPU2运行本次测试。
- 本地SSH连接在每次SSH/SCP后检查并清理，最终无`ssh.exe`和无到`172.31.111.215:22`/`172.31.105.18:22`的ESTABLISHED连接。

远端验证：

| 命令 | 结果 |
|---|---|
| `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/scripts/phase2_profile_guarded_opr_opu_ci_eval.py code/tests/test_phase2_profile_guarded_opr_opu_ci_eval.py` | PASS |
| `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m pytest code/tests/test_phase2_profile_guarded_opr_opu_ci_eval.py -q` | BLOCKED：`No module named pytest`，未安装pytest，未改远端环境 |
| `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/tests/test_phase2_profile_guarded_opr_opu_ci_eval.py` | PASS，5 tests OK |

远端运行：

```bash
CUDA_VISIBLE_DEVICES=2 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_profile_guarded_opr_opu_ci_eval.py \
  --feature_npz runs/phase2_adv3b02_proxy_unknown_ci_20260704/features_proxy_unknown.npz \
  --output_dir runs/phase2_adv3b02_profile_guarded_opr_opu_ci_20260704 \
  --device cuda:0 \
  --profiles base,conservative,known_tight,open_light \
  --policies opu_old_preserve,opu_old_guarded \
  --adapter_epochs 20 \
  --k_shot 4 \
  --query_per_class 20 \
  --max_event_latency_ms 2.0
```

启动PID：`2174096`。该PID在首次检查时已退出，日志显示正常完成。输出40行summary，覆盖4个profile、2个OPU policy和协同数量`1..5`。

远端结果拉取到：`E:\type10-7\local_artifacts\phase2_adv3b02_profile_guarded_opr_opu_ci_20260704\remote\`。

| 文件 | SHA256 |
|---|---|
| `profile_guard_report.csv` | `EFFCF123A5661114E657211FC489BF92B7A765A8D07A23EE10EAF854337140BD` |
| `profile_guarded_opr_opu_ci_summary.csv` | `355E8C0606BF08F32F905A0CC57CDEE5BD51C86D1BACDCCAF2B4955A6585BCB0` |
| `profile_guarded_opr_opu_ci_summary.json` | `2BDA7534A0B6DF7626DC4591F8564B61A85CEE93BCA462F1C69C557626411B06` |
| `profile_guarded_opr_opu_ci.log` | `8E25837C3988A37EB859307B37234C66412239ABAFD582329AC512BC8BA38424` |

## N607结果表

Profile guard：

| profile | guard_pass | guard_score | source_drop | support_drop | proxy_logit_reduction | proxy_surrogate_pass | known_drift_pass | target_unknown_training_count |
|---|---:|---:|---:|---:|---:|---|---|---:|
| base | true | 2.000000 | 0.000000 | 0.000000 | 0.000000 | true | true | 0 |
| open_light | false | 2.032243 | -0.004167 | -0.131250 | 12.139352 | true | false | 0 |
| known_tight | false | 2.022845 | -0.004167 | -0.131250 | 11.885309 | true | false | 0 |
| conservative | false | 1.997487 | -0.003750 | -0.118750 | 11.556518 | true | false | 0 |

选择结果：`base`。选择理由：`known_and_proxy_surrogate_guard_score`。选择阶段`target_unknown_selection_count=0`，`target_unknown_training_count=0`。

Selected profile逐协同数量结果：

| profile | selected | policy | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| base | true | opu_old_guarded | 4 | 0.140625 | 0.000000 | 0.000000 | 0.000000 | 0.916667 | 0.083333 |
| base | true | opu_old_preserve | 4 | 0.161458 | 0.000000 | 0.000000 | 0.000000 | 0.900000 | 0.083333 |
| base | true | opu_old_guarded | 5 | 0.140625 | 0.000000 | 0.000000 | 0.000000 | 0.916667 | 0.083333 |
| base | true | opu_old_preserve | 5 | 0.161458 | 0.000000 | 0.000000 | 0.000000 | 0.883333 | 0.083333 |
| base | true | opu_old_guarded | 3 | 0.067708 | 0.000000 | 0.000000 | 0.000000 | 0.900000 | 0.083333 |
| base | true | opu_old_preserve | 3 | 0.088542 | 0.000000 | 0.000000 | 0.000000 | 0.883333 | 0.083333 |
| base | true | opu_old_preserve | 1 | 0.109375 | 0.000000 | 0.000000 | 0.000000 | 0.883333 | 0.116667 |
| base | true | opu_old_guarded | 1 | 0.072917 | 0.000000 | 0.000000 | 0.000000 | 0.900000 | 0.100000 |
| base | true | opu_old_guarded | 2 | 0.067708 | 0.000000 | 0.000000 | 0.000000 | 0.883333 | 0.083333 |
| base | true | opu_old_preserve | 2 | 0.088542 | 0.000000 | 0.000000 | 0.000000 | 0.850000 | 0.083333 |

各profile最佳诊断行：

| profile | selected | policy | collab_count | old_acc | min_old | seen_new_acc | unknown_reject | unknown_FAR | known_coverage |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| base | true | opu_old_guarded | 4 | 0.140625 | 0.000000 | 0.000000 | 0.916667 | 0.083333 | 0.111111 |
| conservative | false | opu_old_guarded | 4 | 0.151042 | 0.000000 | 0.000000 | 0.950000 | 0.050000 | 0.115079 |
| known_tight | false | opu_old_guarded | 4 | 0.156250 | 0.000000 | 0.016667 | 0.950000 | 0.033333 | 0.123016 |
| open_light | false | opu_old_guarded | 4 | 0.130208 | 0.000000 | 0.016667 | 0.966667 | 0.033333 | 0.103175 |

## 完成结论

该模块完成了“profile选择/回滚”的工程目标：未知类评估不参与训练、阈值拟合或profile选择；协同数量覆盖`1..5`；远端使用`CVS-RFFI`环境；在低显存空闲GPU2运行并完成。

但结果是diagnostic-negative，不满足最终科研目标。原因是输入proxy feature的known基础极差，selected base的`old_acc`最高约0.1615、`min_old=0`、`seen_new_acc=0`，即使未选adapter能把unknown_FAR降到0.0333-0.0500，也因known drift硬门失败不能部署。下一步应在高known特征上实现query-free拒识代理：class-wise高斯原型/EVT、support变换一致性、source leave-one-old-out impostor风险和class envelope门控，再接OPU-CI协同层。

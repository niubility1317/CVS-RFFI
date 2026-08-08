# Phase1 DualReadout-Disagreement窄实验报告

状态：`REVIEWED / READY_TO_RELEASE`

目标模式：`GOAL_MODE=ACTIVE`

## 1.实验身份

| 字段 | 值 |
|---|---|
| run ID | `phase1_dualreadout_disagree_20260808_v1` |
| 日期 | 2026-08-08 |
| 主Agent | `/root` |
| 唯一N607 runner | `/root/n607_geosat_lite_runner`（复用同一Luna/max Runner） |
| 目标 | 验证分类与拒识解耦能否保留C的类别读出并改善B的source-proxy拒识 |
| 比较目标 | `B_ANGULAR_Z0`单读出：proxy FAR=38.25%、source full acc=88.813%、held-known FAR=21.25% |

## 2.方法与数据锁

- angular readout：冻结B checkpoint/NPZ，提供confidence、margin、energy、top-1；
- robust readout：冻结C checkpoint/NPZ，提供最终registered class；
- B/C必须逐行匹配非空`sig_id`及TX/RX/day/equalization/view元数据；同一普通元数据桶内的物理行重排也会失败；
- source联合正确样本上冻结JS divergence Q0.95；接受需同时满足B三门、B/C top-1一致和JS门；
- source TX=`14-10,14-7,20-15,20-19`；proxy=`8-20`；held-known=`6-15`；
- 直接复用上一run的同序clean-view NPZ，每臂source=1600、target_old=400、proxy_unknown=400；不重新过backbone；
- confidence/margin/energy/JS quantile=`0.05/0.05/0.95/0.95`；unknown FAR target=0.05；
- 不训练、不使用held TX校准、不扫参、不fallback、不增加receiver/day/channel对齐。

## 3.本地实现与验证

| 文件 | 目的 |
|---|---|
| `code/scripts/eval_phase1_dualreadout_disagreement.py` | 双读出source-only分歧拒识评估入口 |
| `code/tests/test_phase1_dualreadout_disagreement_eval.py` | 读出职责、zero-held-fit和元数据同序负测 |
| `analysis/phase1_dualreadout_disagreement_design_20260808.md` | 冻结方法、矩阵和判定门 |

验证命令在`ssr-gpu`中执行：

```text
python -m py_compile code/scripts/eval_phase1_dualreadout_disagreement.py
pytest -q code/tests/test_phase1_dualreadout_disagreement_eval.py code/tests/test_phase1_logits_open_set_reject_eval.py code/tests/test_phase1_open_set_reject_eval.py
```

结果：主Agent本地8项通过。初次独立审查发现物理行唯一ID未绑定这一项P0；现已要求非空`sig_id`逐行相同，并补“同桶重排失败”和“缺ID失败”负测。原审查者复核：`STATUS=PASS; P0=0; P1=0; ALLOW_RELEASE=YES`，独立复跑11项通过。

## 4.N607交接锁

| 字段 | 冻结值 |
|---|---|
| Python | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python` |
| CWD | 新Git archive release的`code`目录 |
| 输入B | `runs/phase1_geosat_lite_4arm_20260808_v1/postfreeze_audit_v1/B_ANGULAR_Z0/features.npz` |
| 输入C | `runs/phase1_geosat_lite_4arm_20260808_v1/postfreeze_audit_v1/C_LEO_CONS_Z0/features.npz` |
| output root | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_dualreadout_disagree_20260808_v1` |
| log root | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_dualreadout_disagree_20260808_v1` |
| 资源 | CPU单进程；GPU不占用 |

冻结执行两行：

```text
python scripts/eval_phase1_dualreadout_disagreement.py --angular_npz <B.npz> --robust_npz <C.npz> --source_tx_ids 14-10,14-7,20-15,20-19 --unknown_tx_ids 8-20 --known_query_roles source --unknown_query_roles proxy_unknown --calibration_roles source --conf_quantile 0.05 --margin_quantile 0.05 --energy_quantile 0.95 --js_quantile 0.95 --unknown_far_target 0.05 --output_json <out>/proxy_metrics.json --score_table_csv <out>/proxy_scores.csv
python scripts/eval_phase1_dualreadout_disagreement.py --angular_npz <B.npz> --robust_npz <C.npz> --source_tx_ids 14-10,14-7,20-15,20-19 --unknown_tx_ids 6-15 --known_query_roles source --unknown_query_roles target_old --calibration_roles source --conf_quantile 0.05 --margin_quantile 0.05 --energy_quantile 0.95 --js_quantile 0.95 --unknown_far_target 0.05 --output_json <out>/held_known_metrics.json --score_table_csv <out>/held_known_scores.csv
```

预期artifact：两份JSON、两份逐样本CSV、完整stdout日志、退出码和SHA256清单。

## 5.健康、判定与边界

元数据/shape不一致、输入hash/checkout错误、输出覆盖或确定性异常时停止并保留证据；不得按FAR或accuracy中止。晋级窄门：proxy FAR<38.25%，source full accuracy不低于86.813%，held-known FAR<=21.25%。即使通过但FAR>5%，也只允许进入下一次LEO弱信道窄实验，不得写成Phase3真实unknown或正式deployment bundle。

## 6.结果（待回收）

| row | known count | unknown count | source full acc | unknown FAR | JS AUROC | verdict |
|---|---:|---:|---:|---:|---:|---|
| proxy | pending | pending | pending | pending | pending | pending |
| held-known | pending | pending | pending | pending | pending | pending |

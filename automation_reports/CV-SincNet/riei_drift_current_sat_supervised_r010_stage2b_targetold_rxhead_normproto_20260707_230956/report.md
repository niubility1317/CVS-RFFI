# RIEI/DRIFT Stage2-B ReceiverConditionedSupportHead-CDA修复实验

## 基本信息

- 实验ID：`riei_drift_current_sat_supervised_r010_stage2b_targetold_rxhead_normproto_20260707_230956`
- 时间：2026-07-07 23:09 CST
- 操作者：Codex
- 目标：修复全局ProtoNet/SupportHead-CDA在多目标receiver上混合support导致的receiver-specific响应模糊问题。
- 协议：CVS Stage2-B target-old few-shot domain adaptation；`R_s=day0/day1,rx0-rx6`源域训练，目标域为剩余receiver；support/query均来自目标域旧类`Y_old`；启用LEO星地信道视图；不做新类学习，不做未知类拒识主线。
- 声明边界：本实验只报告target-old域适应；不报告Stage2-C seen-new，不报告unknown拒识成功，不声明部署成功。

## 触发诊断

SourceLogitBias-CDA结果显示源分类器logit不能直接修复目标LEO错位；更重要的是错误分布同时按TX和receiver集中。典型现象：

- `20-15`和`8-20`稳定较高，`14-7`、`6-15`、`14-10`显著较低。
- `7-14`明显好于`20-1`和`3-19`。
- 这表明把5个目标receiver的support合成一个全局head/prototype，可能会把receiver-specific响应混在一起。

## 方法

新增`adaptation_mode="receiver_conditioned_support_head"`：

- 对每个`rx_label`单独取本receiver的support。
- 复用已验证的normalized prototype-initialized support head。
- 只预测同receiver的query。
- 如果某receiver缺support才回退到全局support；本实验中每个target receiver都有完整support，不应触发回退。
- 不使用query标签，不使用unknown query调阈值，不更新backbone。

## 本地变更

Git承载面：`E:\type10-7\github_publish\CVS-RFFI-repo`

提交：

- `f94e746 Add receiver conditioned support head CDA`

变更文件：

- `paper_reproduction/cvs_aligned/evaluate.py`
- `tests/test_paper_reproduction_cvs_aligned.py`
- `paper_reproduction/configs/*rxhead_normproto_n607.json`
- `run_riei_drift_current_sat_supervised_r010_cvs_stage2b_leo_rxhead_normproto_n607.sh`

本地快照：

- `E:\type10-7\code\snapshots\riei_drift_current_sat_supervised_r010_stage2b_targetold_rxhead_normproto_20260707_230956\`

本地哈希：

| 文件 | SHA256 |
|---|---|
| `evaluate.py` | `09395B851F9B2F2BA12B1927514A40A2D989E86F7D9B19E6AA7E3A0908284F4F` |
| `test_paper_reproduction_cvs_aligned.py` | `C88A72610929368018108035EFCE8C64234A3271B78BA946BB3B6C521B63F39A` |
| `run_riei_drift_current_sat_supervised_r010_cvs_stage2b_leo_rxhead_normproto_n607.sh` | `A10D06F0DD5D238D4D2C2C3AA79935B445EB52BCD297089704DDD89A0093E5CB` |
| `drift_current_sat_supervised_r010_cvs_stage2b_k5_leo_rxhead_normproto_n607.json` | `D1BC9F1AA0B55D56CD599D46ADD0737728F8E02166DA62B91CAE0BEF37E82E64` |
| `drift_current_sat_supervised_r010_cvs_stage2b_k10_leo_rxhead_normproto_n607.json` | `80A9603E96EB4338DC013876D6AF82E182AB8EBEF9340A9DE3433FCF1D4E3834` |
| `riei_fd_current_sat_supervised_r010_cvs_stage2b_k5_leo_rxhead_normproto_n607.json` | `31847B64B07641BE1812E51BDCCDB62354AED2D8B9C1719FD86B03CD1725F93E` |
| `riei_fd_current_sat_supervised_r010_cvs_stage2b_k10_leo_rxhead_normproto_n607.json` | `E01D8B1A1404EB71821ABCA816622B58AE9058ECC963A7C71F73F83964835E0D` |

## 本地验证

| 命令 | 结果 |
|---|---|
| `conda run -n ssr-gpu python -m py_compile paper_reproduction\cvs_aligned\evaluate.py` | PASS |
| `conda run -n ssr-gpu python -m pytest tests\test_paper_reproduction_cvs_aligned.py -k "receiver_conditioned or source_logit or support_head or prototype_predict" -q` | PASS，7 passed、12 deselected，只有pytest cache权限warning |
| `bash -n run_riei_drift_current_sat_supervised_r010_cvs_stage2b_leo_rxhead_normproto_n607.sh` | PASS |
| 四个`*rxhead_normproto_n607.json`配置formal dry-run | PASS |

## N607计划

- 远端根目录：`/home/szu2070436088/2510044040/CV-SincNet`
- 远端命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
RUN_ID=riei_drift_current_sat_supervised_r010_stage2b_targetold_rxhead_normproto_20260707_230956 bash run_riei_drift_current_sat_supervised_r010_cvs_stage2b_leo_rxhead_normproto_n607.sh
```

## 结果

完成状态：4个候选均已完成，GPU回到空闲；远端错误扫描为空，本地artifact已拉回：

- `E:\type10-7\automation_reports\CV-SincNet\riei_drift_current_sat_supervised_r010_stage2b_targetold_rxhead_normproto_20260707_230956\artifacts\`

主结果：

| candidate | baseline | K | old_acc | clear | low | rain | coverage | lat_ms | fallback |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `riei_fd_current_sat_k5_leo_rxhead_normproto_seed1337` | RIEI | 5 | 0.6583 | 0.6683 | 0.6550 | 0.6517 | 1.0000 | 0.970 | `[]` |
| `riei_fd_current_sat_k10_leo_rxhead_normproto_seed1337` | RIEI | 10 | 0.6533 | 0.6583 | 0.6317 | 0.6700 | 1.0000 | 1.118 | `[]` |
| `drift_current_sat_k10_leo_rxhead_normproto_seed1337` | DRIFT | 10 | 0.6400 | 0.6650 | 0.6433 | 0.6117 | 1.0000 | 1.043 | `[]` |
| `drift_current_sat_k5_leo_rxhead_normproto_seed1337` | DRIFT | 5 | 0.5539 | 0.5550 | 0.5683 | 0.5383 | 1.0000 | 0.983 | `[]` |

对比前序路线：

| baseline | K | rxhead old_acc | global support-head | delta pp | source-logit | delta pp | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| RIEI | 5 | 0.6583 | 0.5556 | +10.27 | 0.5422 | +11.61 | receiver-conditioned修复有效 |
| RIEI | 10 | 0.6533 | 0.5417 | +11.16 | 0.5361 | +11.72 | receiver-conditioned修复有效 |
| DRIFT | 5 | 0.5539 | 0.4972 | +5.67 | 0.5333 | +2.06 | 小幅有效 |
| DRIFT | 10 | 0.6400 | 0.5650 | +7.50 | 0.5183 | +12.17 | receiver-conditioned修复有效 |

逐TX和逐receiver分解：

| candidate | weak TX | strong TX | weak receiver | strong receiver | remaining top confusion |
|---|---|---|---|---|---|
| DRIFT K5 | `20-19` 0.347、`6-15` 0.373、`14-7` 0.507 | `8-20` 0.837、`20-15` 0.667 | `20-1` 0.444、`3-19` 0.531 | `7-14` 0.642、`7-7` 0.589 | `6-15->1`、`20-19->1/4/0` |
| DRIFT K10 | `20-19` 0.480、`6-15` 0.493、`14-7` 0.563 | `8-20` 0.873、`20-15` 0.727 | `20-1` 0.578、`3-19` 0.592 | `8-8` 0.697、`7-7` 0.681 | `6-15->1`、`20-19->1/0/4` |
| RIEI K5 | `6-15` 0.473、`20-19` 0.477、`14-7` 0.540 | `8-20` 0.930、`20-15` 0.873 | `3-19` 0.553、`8-8` 0.614 | `7-7` 0.728、`7-14` 0.711 | `20-19->1`、`14-7->3`、`6-15->1` |
| RIEI K10 | `6-15` 0.430、`20-19` 0.557、`14-7` 0.567 | `8-20` 0.917、`20-15` 0.813 | `3-19` 0.481、`8-8` 0.619 | `20-1` 0.753、`7-14` 0.714 | `20-19->1`、`6-15->3/1`、`14-7->3` |

日志扫描：

- 4个`.out`均完整解析；无`Traceback`、`RuntimeError`、CUDA OOM、`Killed`、参数错误或`ValueError`。
- `receiver_conditioned_fallback_groups=[]`，说明所有目标receiver都有本receiver support，没有回退到全局support。

## 解释

本次修复确认了一个真实问题：全局prototype/head会把多目标receiver的support合并，导致receiver-specific响应被混合。按receiver分组后，最强RIEI K5达到0.6583，超过全局support-head K5的0.5556，也超过全局K20/K50诊断中的RIEI K50 0.5756；DRIFT K10达到0.6400，超过全局support-head K10的0.5650和DRIFT K50 0.5878。

但结果仍低于PHASE2_ADAPT_NEWCLASS_FIRST旧类阶段门槛`old_acc>=0.80`。残余错误主要集中在`20-19`、`6-15`、`14-7`几类之间，说明receiver-conditioned修复解决了跨receiver混合，但类间identity特征仍有纠缠。下一步应优先做receiver-conditioned K20/K50饱和诊断，若仍低于0.70，则转向小adapter/BN affine或特征空间重训练，而不是继续调source classifier logit。

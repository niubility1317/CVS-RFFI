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

待N607运行完成后追加。

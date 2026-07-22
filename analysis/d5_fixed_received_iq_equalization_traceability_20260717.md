# D5固定已接收LEO_weak IQ均衡探索追踪

日期：2026-07-17
范围：`receiver=20-1`、`seed=713101`、`K=10`、真实`new5`、三种物理样本互斥LEO_weak场景。
声明边界：development-only，不是正式确认矩阵；不提交Git。

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| D5-01 | `项目.md`7.1/7.1.1 | 只读取每个物理样本唯一sealed LEO_weak IQ，不读取clean/source，不生成额外LEO状态 | `automation_reports/CV-SincNet/d5_fixed_iq_equalization_20260717/explore_fixed_iq_equalization.py` | verified | sealed bundle pre-open PASS；`view_lineage.json`逐view绑定parent IQ SHA；跨场景token/SHA互斥 | view不增加K |
| D5-02 | 用户要求 | 预登记DC去除、RMS/robust幅度归一化、相邻样本自相关CFO去旋、轻度频谱收缩 | 同上 | verified | 5个operator实际提取并生成冻结prediction；有限性检查通过 | 每个operator逐样本0拟合参数 |
| D5-03 | `项目.md`7.2/10.3.1 | 只用support LOO选择operator/聚合；query无标签拟合、无角色Oracle、无quota、无batch重排 | 同上 | verified | predictor CLI无truth参数；prediction COMMIT SHA256=`67f79fcad3d92f81edf57a0df668e49532ef4cbe7ac8a3a237316147f85abde4`后独立score | before/after×三场景support等权 |
| D5-04 | 用户要求 | 在`dev_k10_new5_r2`报告base、单变换、最多3-view聚合的old-before、old-after、seen-new、floor | `automation_reports/CV-SincNet/d5_fixed_iq_equalization_20260717/run_dev_k10_new5_r2` | verified | `score_summary.json`、`report.md`、SCORE COMMIT SHA256=`c7a5ca86a43dfeff7849d2a25baa65ed45f5d78938a620c956594dd0cb565a54` | 全部7个variant先冻结prediction |
| D5-05 | 用户要求/星上部署 | 报告MAC、backbone forward、FFT分支、状态与时延 | 同上 | verified | 每variant before/after resource JSON | 算子FFT与特征FFT分开计数；时延为本地batch提取观测，不冒充singleton部署延迟 |

最高风险：单样本CFO去旋可能同时消除发射机稳定CFO指纹；必须由support-only LOO决定是否采用，不能用query结果反向选择。

## 结果与冻结解释

support-only规则锁定`view3_base_rms_cfo`，但冻结query结果为：

- `old_before=0.6722`
- `old_after=0.5639`
- `old_floor_after=0.4167`
- `seen_new=0.5333`
- `H=0.5482`

该锁定候选失败。主要原因是严格lexicographic worst-slice floor把偶然取得`0.10`的CFO混合候选置于其它worst floor为`0`的候选之前，尽管其support overall LOO仅`0.6058`；单独`dc_rms_cfo`的冻结query `H=0.1955`进一步表明直接去旋破坏了稳定TX频偏指纹。

`view3_base_rms_spec15`的冻结query诊断为`old_after=0.6833`、`old_floor_after=0.5000`、`seen_new=0.7100`、`H=0.6964`，相对base分别改善`+5.28pp`、`+6.67pp`、`+1.67pp`和约`+3.60pp`。但这些数值在prediction COMMIT之后才由scorer获得，因此只能用于提出下一development seed的预登记假设，不能反向改写本row winner或作为当前统一超参数。

## 下一轮合法support-only改进

下一独立development seed应在运行前预登记以下选择门，不再读取本row query调参：

1.先设support overall LOO保真门，例如候选overall LOO不得比base下降超过2pp；CFO路线若未通过直接淘汰。
2.对worst-slice floor使用Wilson下界、Beta-Binomial平滑或跨slice出现次数门，避免`0.10`对`0`的一次性偶然值劫持排序。
3.使用leave-two-physical-samples-out或重复分层support subsampling稳定性；选择依据仍只来自support。
4.把base与候选的逐support样本margin差做配对bootstrap；只有floor、overall和margin下界均不恶化时才允许多view。
5.优先预登记`base+dc_rms+spec15`假设；CFO去旋仅保留为消融，不进入下一轮可选集合，除非先证明CFO不承载身份信息。

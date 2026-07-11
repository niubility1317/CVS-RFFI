# Phase1 P1已完成64行阶段分析

时间：2026-07-12 02:40 CST。证据范围为`phase1_dgleo_p1verify128r1_20260710`已完成16行和`phase1_dgleo_p1verify112r2_20260710`已完成48行。仍在运行或排队的候选不进入本次排名。

## 证据完整性

- 完整解析64个stdout，共575877行；完整解析64个`metrics_epoch.csv`，共12788个epoch记录。
- 63个候选完成E200，`P1V128_G0_FULL_DG_PROTECT_S2`在E188触发`STOPPED_TAIL`。
- 64/64具有final权重、完整held-out评估，评估checkpoint SHA256与final权重一致。
- critical train loss、epoch和val TX字段无NaN/Inf；无Traceback、RuntimeError、OOM、Killed或FATAL。
- 63个候选为`NON_PROMOTABLE_GUARD_BLOCKED`，1个为`STOPPED_TAIL`；`endpoint_export_ready=0/64`、`promotion_ready=0/64`、prototype均`SKIPPED_FAIL_CLOSED`。
- 64/64的reference-to-final tail状态不完整，无法得到可信的p99/cvar扩张闭环。

## 核心结论

当前没有可称为“open-set表现最好且可推进”的候选。63/64候选在达到98.62%-98.68%的早期source-val峰值后，于E24-E43跌到近随机水平；只有4个final source-val高于20%，只有1个高于40%。低`final_proxy_vaccept`、低local p95/p99主要对应known core accept约5%的塌缩解，不能解释为拒识能力改善。

当前闭集DG、星地压力和legacy proxy联合最强的诊断候选是`P1V128_G0_FULL_DG_PROTECT_S2`。它明显优于其余已完成候选，但E188触发tail stop，open-set几何仍严重失败，不能promotion或导出endpoint。

## 联合候选表

|candidate|final val|overall|strict UDU|receiver floor|sat mean/floor|legacy proxy_vaccept|source episode overflow|final p95/p99|final proxy/bridge|final tail/overflow accept|radius/inter|clean/sat core accept|结论|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|`G0_FULL_DG_PROTECT_S2`|68.07|49.76|44.81|35.59|26.49/23.03|0.6052|0.8996|90.00/90.00|0.2454/0.4476|0.8775/0.1337|68.80|0.5610/0.2144|性能最强、legacy proxy最不差，但tail严重扩张，E188停止|
|`G1_SAT_CE_CONTROL_S2`|29.15|24.73|23.86|16.14|17.68/17.44|0.6717|0.4947|1.20/1.76|0.0520/0.0688|0.0593/0.0555|4.95|0.0581/0.0648|local代理低来自known拒绝塌缩|
|`G1_SAT_INV_FOCUS_S2`|28.22|23.70|22.74|16.95|18.84/17.95|0.6719|0.4447|0.95/1.52|0.0484/0.0617|0.0582/0.0578|3.88|0.0556/0.0616|星地略强，但known core和DG已塌缩|
|`G0_FULL_BALANCED_S2`|25.57|22.12|21.12|10.78|18.86/18.18|0.6746|0.4979|0.81/0.85|0.0415/0.0522|NaN/0.0523|1.73|0.0522/0.0522|弱receiver更差，代理低无效|

## 最强诊断候选具体表现

`P1V128_G0_FULL_DG_PROTECT_S2`使用seed`710211`，final-only权重实际停在E188：

- receiver：rx7`40.76`、rx8`35.59`、rx9`51.06`、rx10`55.44`、rx11`50.98`。
- satellite：clear`27.92`、low-elev`27.13`、rain`26.58`、storm`23.03`、geo`29.82`、mixed-orbit`24.46`。
- dynamic direct metric：proxy`0.5231`、bridge`0.5507`、overflow accept`0.3006`、source overflow`0.4629`、radius/inter`32.66`、p95/p99/cvar均约`90°`。
- legacy proxy：proxy_vaccept`0.6052`、hard accept`0.7254`、bridge accept`1.0000`、low-density accept`0.00425`、shell accept`0.7932`、proxy AUC`0.3914`、radius/inter`39.40`。
- fixed final geometry：proxy`0.2454`、bridge`0.4476`、low-density`0.01195`、tail accept`0.8775`、overflow accept`0.1337`、source overflow`0.3698`、radius/inter`68.80`。
- U_s：direct active`0`、weighted direct loss`0`；trusted core`0.1144`、ambiguous tail`0.00134`、outside reject`0.8842`。
- leakage excess：receiver`0.0770`、day`0.0216`、channel`0.1488`。

## 单项最小值为何不能当最优

|单项|候选|数值|无效原因|
|---|---|---:|---|
|fixed final proxy_vaccept最低|`G3_UINV_CHANNEL_S1`|0.03215|overall/strict/satellite均16.67%，clean core accept仅0.0536，legacy proxy仍0.6746、bridge仍1.0|
|source episode overflow最低|`G2_LINV_FULL_S2`|0.2490|overall/strict均16.67%，core accept仅0.0522，U_s全部outside|
|dynamic dm proxy最低|`G3_UINV_CHANNEL_S4`|0.22366|overall/strict均16.67%，部分dm字段缺失，U_s direct仍inactive|

## seed稳定性

`G0_FULL_DG_PROTECT`四个seed中只有S2保留DG能力：S1/S3/S4的overall和strict均为16.67%，clean core accept约0.052，legacy proxy约0.673-0.674。S2的overall为49.76%、strict为44.81%，但p95/p99为90°且source episode overflow为0.8996。这是seed特异的“保住分类但tail爆炸”分支，不是稳定机制收益。

## 阶段决策

- 当前最值得保留做故障诊断的是`P1V128_G0_FULL_DG_PROTECT_S2`。
- 当前没有Stage2/Phase3真实unknown评估候选，也没有可导出的`endpoint_accept_v1`候选。
- 不能声明proxy_vaccept、bridge、tail或low-density拒识已经改善；dynamic/local指标下降与known接收率塌缩同时发生，legacy bridge仍接近1。
- 完整矩阵结束前仍需观察后续机制单元是否出现“final DG不塌缩、known core accept充足、legacy proxy同步下降”的候选；若没有，本矩阵应判为训练稳定性/P1门控诊断负例。

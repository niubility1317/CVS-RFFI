# D70交叉拟合原子生命周期行替换追踪

## 复盘来源与待修复问题

D67连续stacking以N下降换A/F小幅改善；D68逐行标定破坏绝对joint尺度；D69冻结全部D62 before旧行后，B精确92.78%，但N下降10pp、新→旧增加15。三轮复盘表明D65的生命周期稳定性只能在D62 final joint坐标系内选择性使用，不能连续混合、逐行等幅化或盲目替换全部旧行。

## 唯一数学机制

始终以D62 Stage2-C final joint head为base。K>=2时按physical rank构造两个互斥、exact-once的nested support-held fold；K8时每折hold4 rank/类、train4 rank/类。每折分别只用train support拟合：

```text
G_B^f = D62(train old support)
G_C^f = D62(train old union new support)
```

在该折held的全部已注册类support上，以`G_C^f`为base。对每个可追溯的Stage2-B旧行`c`，只把score列`c`替换成`G_B^f`对应列，计算全held集合的一对多正确数TP和误吸收数FP。单行初选要求：

```text
TP_candidate[c] >= TP_base[c]
FP_candidate[c] <= FP_base[c]
and at least one strict improvement
```

把全部初选旧行同时替换后，再对当前全部11类要求`TP_joint>=TP_base`且`FP_joint<=FP_base`逐类成立；否则整组mask清零。full support阶段分别拟合D62 before和D62 final，只在final joint head上按binary mask替换旧行；新行始终保留D62 final joint行。K1精确回退D62。

## 单一差异与边界

相对D62，唯一差异是crossfitted atomic-safe的少量旧行可来自before D62；base、新行、未接受旧行和最终坐标均为D62 final joint。相对D69，不再全部替换6个旧行；相对D67，没有alpha或连续权重；相对D68，没有center、scale或方向修正。

Stage2-B/Stage2-C注册生命周期是support registry事实，不是query角色Oracle。query只有单一全注册类affine head，逐样本面对全部类。无class ID、scene/receiver、threshold、temperature、offset、query、clean/source或ground分支；D22仍未具正式资格。

## 可观察结果与停止条件

- before必须精确D62；mask为空时final必须精确D62，包括INT8/FP32 state和预测。
- 两折held集合必须exact-once、train/held不交；gate对类标签置换等变，联合mask必须满足全部注册类TP/FP原子安全。
- 真实outer相对D62不得交换A、N、H、J、min-A、min-N或三场景floor；至少严格改善A、F、J或任一floor才可进入第二development seed。
- 若mask活跃但outer负交换，停止生命周期行替换路线，不扫描fold、阈值、权重或温度。若mask全空且精确D62，记录为无新增性能的安全fallback。
- 首seed前只跑receiver`20-1`、seed`713101`、K10/new5、三场景×五fold的105行；不运行125。

## 实现与结果状态

已实现独立core、锁定probe和10项专项测试，不修改D62/D69历史文件。专项10/10、D42–D70完整链345/345通过；完整链用时81.5s。资源审计预留每个target row额外4次inner D62、40次inner component fit以及相应LDA/Fisher/held-score/gate MAC。

实现提交`10536c01`已建立干净worktree，干净D42–D70链345/345再次通过，用时82.8s。真实105行命令、source closure和预期60/30/120/2280调用闭包已在automation report锁定。

真实105/105行已完成。INT8目标B/A/N/H/F/J=92.78/82.22/84.67/82.62/10.56/26.67，min-B/A/N=80.00/53.33/73.33，与D62所有汇总和floor完全相同；混淆为25/8/15，较D62的23/8/15多2个旧→新错误。9折触发联合原子失败精确回退，5折无行接受精确回退，只有`leo_clear_weak/fold1`接受old1一行。该折support-held TP/FP改善没有转化为outer增益。

结论为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`：原子门验证了安全回退，但生命周期旧行替换没有新增性能，并显著增加support拟合计算。停止旧行替换及其fold/阈值/权重/温度扫描；D62仍是联合最强，下一路线必须保留全类joint坐标并同时解决low/rain的old4、old5、new1、new3 floor。完整7候选、3场景、11类、15fold、训练、量化、资源与artifact见同名automation report及`d70_full_performance_summary.json`。

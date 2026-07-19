# D83地面干扰谱精度加载

状态：`PREREGISTERED_LOCAL_VERIFIED_QUERY_NOT_OPENED`。实验ID：`d83_ground_precision_loading_20260720`；时间：2026-07-20 06:15 HKT；操作者：Codex。

目标是修复D82“压小support方差后，LDA逆协方差反而放大干扰方向”的机制错误。D83保留D81的类对称一步Cauchy稳健中心，类内残差不变；随后仅在每个D62 full/block support-only fit的共享协方差中加入rank14地面干扰loading：

```text
tau = trace(Sigma_target_z) / 160
L = (rank * tau / K) * U diag(pi) U^T
precision = inverse(Sigma_target + L)
```

该式使14个地面干扰方向的平均loading恰为target平均方差的`1/K`；K≤2不加载，K1严格回到基线。无超参/强度/rank扫描，新旧类同式，无class ID/role/scene分支，query不更新且额外评分MAC为0。与D80的主要差异是只加载D81筛出的低秩干扰谱，不按ground域自由度强混合完整协方差。

开发单元锁定`rx20-1/seed713101/K10(actual K8)/new5/3场景×5fold`，复用D18 `VALIDATED_ONCE` capsule、runtime authorization和D22只读84-cell int8 ground组件；不重建、不重验数据。ground仍为`UNVERIFIED`，所以只能形成开发诊断证据。成功门：相对D81的B/A/N/H/F/J、全部场景、逐类与mean-row floors、三类混淆均不回退，且A/H/F、rain或新类至少一项严格改善；否则立即判负，不启确认seed/125。

本地Git worktree：`E:\type10-7\code\snapshots\d81wt`；核心SHA256=`953ed053896d189b6022036b2ddcbad8c5c0ac71a88ac8920740b2aababfb31a`；probe SHA256=`e6626edd22a745747ed09a752692c331fdfa42db2fad94e6c337135a59f59f4a`。ssr-gpu环境专项12/12、D62-D83相邻链61/61 PASS，`py_compile`与`git diff --check`PASS。

本地RTX5070Ti执行，不占N607。输出：本报告目录`ground_precision_loading/`；预计105-row日志、receipt、metadata、完整逐类/场景/资源汇总。资源门仍为params≤80k、epochs≤30、steps≤50、state≤256KB、dense query graph=false。

## 完成结果

待完整运行后补充总体、场景、逐类、混淆、量化、训练、资源、缺陷和判定。


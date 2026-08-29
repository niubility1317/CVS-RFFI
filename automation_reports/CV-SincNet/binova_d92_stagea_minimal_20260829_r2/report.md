# BiNOVA-D92阶段A最小实验报告（r2技术修复）

- 状态：`LOCAL_VERIFIED`
- run ID：`binova_d92_stagea_minimal_rx20_1_s713101_20260829_r2`
- r1差异：仅显式指定布尔mask为`torch.bool`；方法、矩阵、门槛、场景、split、seed和资源预算不变。
- 输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/binova_d92_stagea_minimal_rx20_1_s713101_20260829_r2`
- 阶段A通过support-only门槛后自动继续阶段B，否则记录`NOT_RUN_GATE_NOT_MET`。

## 运行结果

- 最终状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- A2/A3/A4训练已计算，但held评估前遇到Tensor→NumPy的`numpy.object_`兼容错误；未写artifact、未打开query/truth、未进入阶段B。
- r2输出保留，显式dtype重建后改用不可覆盖r3。

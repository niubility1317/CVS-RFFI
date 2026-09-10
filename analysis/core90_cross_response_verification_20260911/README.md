# 本地合成验证证据

这些文件验证实现行为，不构成正式CVS性能结果。完整说明见[实现报告](../CORE90_CROSS_RESPONSE_IMPLEMENTATION_VERIFICATION_20260911.md)。

- `matrix.json`：最终版本全部10个变体，真实CUDA主干、label/pseudo及LEO路径；每项2轮。源门阈值仅在fixture内放宽，以触发联合分支。
- `amp.json`：U0/U4_bilinear各4轮16批次，15次有效更新，保留scaler回退。
- `u0_off.json`、`resume.json`：真实U0包关闭和U5续训零容差回归。
- `deployment.json`：辅助状态移除后，真实checkpoint严格加载和输出相等。
- `unit_results.xml`：84项通过，13项按普通套件设置跳过；重计算路径由上述脚本单独执行。
- `final_predictions/`：真实模型和卫星增强下，8条独立合成记录的四场景预测与后置truth评分。
- `runtime_logs/`：最终10个变体实际入口日志。
- `amp_diagnosis/`：修复前失败证据；不能把这些失败计为已完成更新。

来源目录及各项fixture覆盖写在对应JSON中。大体积PKL/checkpoint保留在本地，没有提交。

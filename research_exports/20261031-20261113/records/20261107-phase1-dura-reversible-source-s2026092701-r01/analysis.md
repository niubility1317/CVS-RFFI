# DURA 可逆耦合：源训练完成

在 [计算环境] GPU 0 使用 32768 条合法源记录的三个视图训练四层可逆耦合，DURA 冻结，固定五轮；训练不读取 F、TX 标签、V 或目标。真实 checkpoint 源训练 smoke 通过，五轮优化日志耗时 384.6 秒，源缓存读取 0.648 秒和传输 0.245 秒另计，原 DURA 训练和源缓存构建成本另计。最终模型对源样本的逆变换检查通过；有限精度并非逐值无损。

此前探索过的 672 条源 pilot，每组 48 条，DURA 可能见过相同源域。候选的平均波形误差在 12/12 个受损组低于 DURA，在两个 clean 组均高于 DURA；幅度很小。波形 NMSE 不是 TX 身份准确率，正式七景预测正在运行。

| 源组 | 条数 | DURA NMSE | 可逆候选 NMSE |
| --- | ---: | ---: | ---: |
| RX4_6/clean | 48 | 0.00004552 | 0.00004974 |
| RX4_6/practical_high | 48 | 0.00542070 | 0.00539065 |
| RX4_6/practical_mid | 48 | 0.00802796 | 0.00798849 |
| RX4_6/practical_low_suburban | 48 | 0.00871864 | 0.00868329 |
| RX4_6/practical_high_urban | 48 | 0.00909436 | 0.00906901 |
| RX4_6/practical_mid_urban | 48 | 0.02324280 | 0.02321456 |
| RX4_6/practical_low_urban | 48 | 0.03740450 | 0.03736757 |
| RX8/clean | 48 | 0.00003287 | 0.00004414 |
| RX8/practical_high | 48 | 0.01545082 | 0.01537347 |
| RX8/practical_mid | 48 | 0.01540366 | 0.01533022 |
| RX8/practical_low_suburban | 48 | 0.02475799 | 0.02468340 |
| RX8/practical_high_urban | 48 | 0.01530391 | 0.01523684 |
| RX8/practical_mid_urban | 48 | 0.05165612 | 0.05155327 |
| RX8/practical_low_urban | 48 | 0.10067829 | 0.10060954 |

源摘要、源训练 smoke。

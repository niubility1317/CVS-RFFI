# GroupBarrier support 诊断

已闭合 4 行、160 个 parent。实际 runtime：`9518d46d2f763e5b080b5337e23155c9de883e09`。

本报告只描述 support OOF 与 one-shot proxy，不代表 query 性能，不选参数、不自动晋级。A、B、C 按同 parent、同旧类物理 held 配对；K=1 的准确率均为 N/A，new0 的新类准确率、H 与新旧差距均为 N/A。

R0 独立拟合 B0/C0；R_GROUP_BARRIER_seq 的 B 是实际 Margin B，C 继承该 B，并冻结旧类条件函数。原始同 parent 表与完整 K×新增类数表见 CSV。指标单位为比例，gain/forget 差乘 100 后为百分点。

SUM 计数与耗时、MAX 实际 factor buffer 峰值分开保存。新 Ridge、gate 前向/伴随、准备与实际 B 的成本均保留；训练 events 的累计 audit 不重复计费。CPU 测量和本地 native packet 文件字节不代替星载测量或传输字节；未测量项为 N/A。

barrier 日志与 NPZ 原方程读回单独保存。理论 gap 和 μ×slack 不是精确实测 dual 证书，旧 QP 的独立活跃集、严格互补检查不适用于该 gate。

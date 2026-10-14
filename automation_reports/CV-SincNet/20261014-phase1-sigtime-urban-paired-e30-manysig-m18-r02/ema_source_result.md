# 既有 C4 自身 EMA 的独立源 V 校准结果

三 seed 的冻结 EMA 在源 V 上均能提供大量高精度、高置信软标签。当前低监督覆盖不能只解释为 teacher 低置信；原 batch-neighbor 条件的低通过率是主要可观察限制。该结果支持下一轮检验扩大额外 urban 软监督，尚未证明它提高目标成绩。

| seed | EMA 全 V 准确率 (%) | clean/弱一致率 (%) | confidence≥.95 精度 (%) | 覆盖率 (%) | 同样本弱一致精度 (%) | 覆盖率 (%) |
|---|---:|---:|---:|---:|---:|---:|
| C4-s2026092701 | 97.6296 | 99.7296 | 99.8228 | 91.9519 | 99.8261 | 91.5630 |
| C4-s2026092702 | 97.7667 | 99.7556 | 99.8697 | 90.9667 | 99.8773 | 90.5630 |
| C4-s2026092703 | 97.7222 | 99.7889 | 99.8583 | 91.4889 | 99.8659 | 91.1556 |

每模型完整 V=27000，三模型使用同一组 27000 个源物理 ID 与同一弱 IQ 扰动；不是 81000 个独立物理样本。没有打开 U、读取 query/truth、更新参数或拟合头。EMA 来源是各自 final E30 checkpoint，并通过既有完整 scratch/源契约/预算/final-only 检查。

| row / confidence≥.95 | RX×真实 TX 最低条件精度 (%) | 最少接受样本数 / cell |
|---|---:|---:|
| C4-s2026092701 / clean_confidence | 95.3380 | 429 |
| C4-s2026092701 / same_sample_weak_agreement | 95.0372 | 403 |
| C4-s2026092702 / clean_confidence | 98.3957 | 366 |
| C4-s2026092702 / same_sample_weak_agreement | 98.3146 | 343 |
| C4-s2026092703 / clean_confidence | 96.3592 | 378 |
| C4-s2026092703 / same_sample_weak_agreement | 96.4377 | 364 |

这些分层以源 V 合法真实 TX 定义，统计接受集合中正确预测的比例。它们只作机制观察，不能给 U 或目标设置真实类别门控。总体高精度不代表所有 RX×TX 同样可靠，也不证明隐藏 U 或目标精度。

阈值 .92/.95/.97 均在运行前固定并完整保留，没有按目标成绩选择阈值。弱视图为 source V IQ 加每坐标 std=.01 的固定高斯噪声，seed=2026101501；formal practical IQ、SNR、边界、难样本和原模型结果没有改变。

实际新增网络前向样本量 162768，CPU 墙钟 254.327 秒；新增训练更新与 epoch 为 0。实际代码提交 3c390341f0b0bc65b03413cada2e941138fa1188，命令 python -u -m experiments.cvs_urban_sigtime.ema_source，CWD 为 PROJECT/releases/cvs_sigtime_urban_paired_20261014_r02_ema_vcal，CPU 两线程、CUDA_VISIBLE_DEVICES 为空。

五项聚焦检查通过，一次针对新源诊断代码的独立 P0/P1 审查无阻塞问题；三个真实 EMA source-only checkpoint smoke 通过，tensor equality 确认模型与 buffer 不变。一次归档传输比较、远端编译/preflight 与 PID/CWD/argv/log 增长检查通过。进程已终止，完成证明来自新 source_ema_vcal/results.json，未借用原训练 complete.json。

下一轮保持 SIG_TIME/原内部结构、默认头、旧门控基线与每模型从头 E30。可优先独立检验额外 urban 软监督使用 clean/同物理弱视图可信标签，并保留原 native 旧门控损失；困难 urban 学生不需要先高置信。下一轮新增模型数量尚待答复，因此未启动新训练，没有改变旧矩阵或按目标成绩重新选择 checkpoint。

low urban 60% 目标仍未达到；该诊断不是新的目标准确率结果。

[18 条全部阈值指标](E:/type10-7/local_artifacts/cvs_urban_sigtime_20261014/final/ema_source_vcal_metrics.csv) · [完整源结果及 RX×TX 分层](E:/type10-7/local_artifacts/cvs_urban_sigtime_20261014/final/ema_source_vcal.json)

[源精度—覆盖率图](E:/type10-7/local_artifacts/cvs_urban_sigtime_20261014/final/figures/source_ema_precision_coverage.png) · [固定 .95 阈值三 seed 图](E:/type10-7/local_artifacts/cvs_urban_sigtime_20261014/final/figures/source_ema_threshold095.png)

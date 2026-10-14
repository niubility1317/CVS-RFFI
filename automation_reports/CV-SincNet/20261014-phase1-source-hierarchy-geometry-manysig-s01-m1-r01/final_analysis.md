# 连续层级时序签名：真实源域结果

**VERIFIED，720 个固定源包、135 项统计，0 训练/优化器/分类头更新。** 默认头及完整模型参数和 buffers 逐值不变，保存数组独立重读复算通过。

使用已登记固定首种子 SIG_TIME final50，所有源 RX 均见过；这不是留一RX实验。实际路径为32位置×6特征加全局时间。连续4→2→1组合最大绝对误差：full float64 8.881784197001252e-16，half float64 6.661338147750939e-16，native float32 7.152557373046875e-7，均满足预登记容差。真实source观察 1.240 秒，另有2包零IQ真实checkpoint smoke。

| 错误对照 | 描述子组 | direct RMS | error RMS | error/direct RMS |
|---|---|---:|---:|---:|
| missing_bridges | full28 | 0.76297304 | 0.58309521 | 76.4241% |
| missing_bridges | increment6 | 0.76378562 | 0.89782654 | 117.5495% |
| missing_bridges | time_increment1 | 1.00000000 | 0.09677419 | 9.6774% |
| missing_bridges | feature_area15 | 0.87409741 | 0.54897599 | 62.8049% |
| missing_bridges | time_area6 | 0.23758324 | 0.15992752 | 67.3143% |
| reset_clock | full28 | 0.76297304 | 0.67884388 | 88.9735% |
| reset_clock | increment6 | 0.76378562 | 0.00000000 | 0.0000% |
| reset_clock | time_increment1 | 1.00000000 | 3.09677419 | 309.6774% |
| reset_clock | feature_area15 | 0.87409741 | 0.00000000 | 0.0000% |
| reset_clock | time_area6 | 0.23758324 | 0.74310183 | 312.7754% |
| omitted_cross_area | full28 | 0.76297304 | 0.25584758 | 33.5330% |
| omitted_cross_area | increment6 | 0.76378562 | 0.00000000 | 0.0000% |
| omitted_cross_area | time_increment1 | 1.00000000 | 0.00000000 | 0.0000% |
| omitted_cross_area | feature_area15 | 0.87409741 | 0.32413896 | 37.0827% |
| omitted_cross_area | time_area6 | 0.23758324 | 0.20689475 | 87.0831% |

省略有序cross-area时，feature area 的误差/完整RMS为37.0827%，time area为87.0831%；完整28维为33.5330%。缺少3个跨窗边界时，完整描述子误差为76.4241%；时间增量少3/31。每段分别重置时间后，feature increment/area保持浮点等价，但时间增量与time area明显改变，因此下一步必须单列global-time局部控制，不能把时间坐标变化的影响都归因于层级机制。比例按源样本/对应分量的RMS定义，不能解释为信息量或分类性能百分比。

数学上，拼接遵循 [原论文的Chen identity](https://arxiv.org/pdf/1603.03788)，使用3个边界增量和单一全局时钟；本实现为一阶增量＋二阶反对称面积，并未实现所有阶的完整signature。连续组合与直接路径严格按此截断定义核对；不是近似省略bridge。合成反例显示四个局部描述子相同而整体描述子可不同，但不能据此声称当前整个骨干都没有跨窗表达能力。

下一步固定为单个层级假设：原SIG_TIME、仅全局时钟局部控制、相同容量的大窗口统计及连续层级有序聚合，配同配方旧门控＋cos基线。保持默认头、实际joint160导出、从零≤50轮；不同时叠加取点改变或U权重。是否提升必须等独立固定目标测试。当前无目标前向、无新训练网络、无84%或跨TX能力结论。

运行 20261014-phase1-source-hierarchy-geometry-manysig-s01-m1-r01；实现/config 13728d26334a3523c4225f24a274f410bd462076；实际发布 4cc15c4db59bc444d2952426885a22d6f92dc8fa。唯一直接P0/P1 PASS。普通N607 CPU2、无GPU占用；preflight/remote compile/单archive SHA比对PASS，PID2177144的启动身份及终端产物核实，log204 bytes、failure无。完整135项分层在results.json；原source_geometry.pt及日志留在预登记不可变目录，不上传Git。

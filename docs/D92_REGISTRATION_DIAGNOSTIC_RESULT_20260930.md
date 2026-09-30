# LocalRidge 注册机制结果与优化依据

本次是原 LocalRidge 的完整 support 机制诊断，未训练新方法。主要发现是：在选定的 practical residual 场景中，注册后的旧类下降主要来自新增类竞争；仅冻结旧类参数不足以保证低遗忘。该结论描述固定分数上的错误变化，不能据此断言某个具体模块已经找到了因果根源。

固定旧类数为6，新增类数为0/2/5/10/20，K为每个注册类的物理support数。两个模型seed、两个cohort，各选择两个RX×场景组合，合计160组。rx3选择receiver 19-1，rx1选择20-19；两者均测practical_high和practical_low_urban，不含practical_mid及其他receiver。信道沿用residual/post_sync/noeq、25 MHz。

**三阶段报告边界：** A（地面原模型适应前）为N/A；本诊断B0是仅旧类训练support拟合的原LocalRidge，C0是包含新类训练support后独立重新拟合的原LocalRidge。二者不是继承适应状态的顺序微调。真实适应收益B−A为N/A，不能把这里的B0当作微调提升证据。C0旧类和新类准确率来自统一全类竞争。K=1没有独立物理持出样本，40组均不拟合、不虚构持出性能。下表OOF每个物理held样本只统计一次；K是分折前总support数，实际训练每折少于K。

K=20、新增20类时，OOF旧类从76.250%降到65.104%，总下降11.146个百分点；C0分数固定后仅保留旧类列得到77.083%，因此旧类内部决策变化为−0.833个百分点，新类竞争损失为11.979个百分点，二者相加等于总下降。新类为62.906%，与最终旧类差约2.198个百分点；这个差距小同时伴随旧类大幅下降，不能单独作为成功依据。

K=5时同一规模的竞争损失为15.000个百分点，K=10为9.375个百分点，K=20为11.979个百分点；support增加并不保证有限样本上的注册损失单调下降。该pilot中旧类B0准确率随K从63.333%升至73.542%、76.250%，但加入更多类别后仍有混淆。不能把“相对某基线的提升幅度”与“本方法绝对准确率”混为一谈。proxy每次只用一个anchor，表中的parent K主要改变可用anchor及held集合，不表示每次拟合使用K个样本。

下一版设计应同时改善新旧类别的可分性和support估计的稳定性。仅给旧类加偏置可能把错误转移给新类，必须成对报告B0/C0旧、新及差距；冻结Adapter本身也不能消除新增竞争。当前没有依据宣布某个新PEFT方案优于LocalRidge。方法设计只接收本次完整support证据，不使用历史query结果选参。

原始分数、物理ID和逐阶段成本位于远端run各row的probe目录；独立summary逐条复算分解后生成。训练/评分均未读取query或源域样本，未加载新checkpoint。历史特征缓存沿用已核实的source-only来源。本次新增模型参数与新增星地模型传输为0；研发报告/分数日志不计作部署payload。实际CPU运行墙钟109.111秒，不是星载时延测量。完整资源字段及分层见同run的summary.json和CSV。

证据：[完整汇总](../automation_reports/CV-SincNet/20260930-phase2-d92-registration-diagnostic-m2-r01/results/support_summary/summary.json)、[K×新增类数](../automation_reports/CV-SincNet/20260930-phase2-d92-registration-diagnostic-m2-r01/results/support_summary/by_k_new_count.csv)。运行代码0e6fe9613，分析代码67e452901；此结果不宣称全新独立验证或完整query性能。

旧类内部准确率的净变化较小，并不表示每个旧样本的类别排序完全不变；相互抵消的改善和退化仍可能存在。分解只是固定分数下的计数恒等式，不是某个模块的独立因果效应。

## 完整数值表

完整 160 parent、4 个模型/cohort row 已校验。仅原 LocalRidge B0/C0，无 Adapter。真实 A、微调 B 和适应收益均为 N/A。

信道继承 practical residual / post_sync / noeq，fs=25 MHz；未新生成信道。仅解释显式选择的 RX×场景。

| 诊断 | K | 新类数 | B0旧 | C0旧类列 | C0全类竞争旧 | C0新 | H | 旧类决策变化 | 新类竞争损失 | 总旧损失 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oof | 5 | 0 | 63.333 | 63.333 | 63.333 | N/A | N/A | 0.000 | 0.000 | 0.000 |
| oof | 5 | 2 | 63.333 | 63.333 | 61.250 | 50.000 | 53.522 | 0.000 | 2.083 | 2.083 |
| oof | 5 | 5 | 63.333 | 62.500 | 54.583 | 50.000 | 51.918 | 0.833 | 7.917 | 8.750 |
| oof | 5 | 10 | 63.333 | 63.750 | 53.333 | 48.250 | 50.361 | -0.417 | 10.417 | 10.000 |
| oof | 5 | 20 | 63.333 | 63.750 | 48.750 | 46.250 | 46.539 | -0.417 | 15.000 | 14.583 |
| oof | 10 | 0 | 73.542 | 73.542 | 73.542 | N/A | N/A | 0.000 | 0.000 | 0.000 |
| oof | 10 | 2 | 73.542 | 73.958 | 71.667 | 51.875 | 59.451 | -0.417 | 2.292 | 1.875 |
| oof | 10 | 5 | 73.542 | 73.125 | 67.917 | 56.500 | 61.409 | 0.417 | 5.208 | 5.625 |
| oof | 10 | 10 | 73.542 | 73.542 | 65.625 | 57.875 | 61.456 | 0.000 | 7.917 | 7.917 |
| oof | 10 | 20 | 73.542 | 73.542 | 64.167 | 56.500 | 60.024 | 0.000 | 9.375 | 9.375 |
| oof | 20 | 0 | 76.250 | 76.250 | 76.250 | N/A | N/A | 0.000 | 0.000 | 0.000 |
| oof | 20 | 2 | 76.250 | 76.875 | 74.688 | 53.125 | 61.590 | -0.625 | 2.188 | 1.562 |
| oof | 20 | 5 | 76.250 | 76.875 | 70.104 | 61.250 | 65.085 | -0.625 | 6.771 | 6.146 |
| oof | 20 | 10 | 76.250 | 76.771 | 68.021 | 63.625 | 65.703 | -0.521 | 8.750 | 8.229 |
| oof | 20 | 20 | 76.250 | 77.083 | 65.104 | 62.906 | 63.944 | -0.833 | 11.979 | 11.146 |
| proxy | 5 | 0 | 50.938 | 50.938 | 50.938 | N/A | N/A | 0.000 | 0.000 | 0.000 |
| proxy | 5 | 2 | 50.938 | 51.667 | 49.271 | 33.125 | 33.343 | -0.729 | 2.396 | 1.667 |
| proxy | 5 | 5 | 50.938 | 51.562 | 45.729 | 39.625 | 41.070 | -0.625 | 5.833 | 5.208 |
| proxy | 5 | 10 | 50.938 | 51.146 | 42.292 | 37.438 | 38.783 | -0.208 | 8.854 | 8.646 |
| proxy | 5 | 20 | 50.938 | 51.771 | 38.438 | 33.281 | 34.149 | -0.833 | 13.333 | 12.500 |
| proxy | 10 | 0 | 53.032 | 53.032 | 53.032 | N/A | N/A | 0.000 | 0.000 | 0.000 |
| proxy | 10 | 2 | 53.032 | 54.051 | 51.019 | 25.556 | 29.944 | -1.019 | 3.032 | 2.014 |
| proxy | 10 | 5 | 53.032 | 54.444 | 47.199 | 36.361 | 39.869 | -1.412 | 7.245 | 5.833 |
| proxy | 10 | 10 | 53.032 | 54.352 | 43.773 | 36.931 | 39.069 | -1.319 | 10.579 | 9.259 |
| proxy | 10 | 20 | 53.032 | 54.514 | 39.699 | 32.465 | 34.548 | -1.481 | 14.815 | 13.333 |
| proxy | 20 | 0 | 52.473 | 52.473 | 52.473 | N/A | N/A | 0.000 | 0.000 | 0.000 |
| proxy | 20 | 2 | 52.473 | 52.610 | 49.644 | 28.980 | 33.078 | -0.137 | 2.966 | 2.829 |
| proxy | 20 | 5 | 52.473 | 52.939 | 45.482 | 37.053 | 39.598 | -0.466 | 7.456 | 6.990 |
| proxy | 20 | 10 | 52.473 | 52.917 | 41.634 | 36.089 | 37.686 | -0.444 | 11.283 | 10.839 |
| proxy | 20 | 20 | 52.473 | 53.421 | 38.070 | 32.388 | 34.167 | -0.948 | 15.351 | 14.402 |

准确率为百分数，差值为百分点。恢复比例、绝对新旧差及分场景结果见 CSV；proxy 先均 anchor 后等权 parent。

实测 head fits=3168，factorizations=3168；运行墙钟 109.111 s。
各 head 的 fit 耗时之和 47.865 s，score 耗时之和 148.122 s；并发工作量之和不等于墙钟。

- B0 fits old training support; C0 independently refits all registered training support using the unchanged LocalRidge.
- C0 old-column argmax is an offline fixed-score counterfactual, never a deployed role-conditioned predictor.
- The exact count decomposition separates old-class decision reordering from added new-class competition; it does not isolate individual kernel or coefficient causes.
- OOF pools each physical held ID once before computing accuracy; unequal folds are not equally weighted.
- Proxy first averages all anchors per parent, then weights parents equally; repeated support draws are correlated.
- True K1 is numerical-only. No held accuracy or head fit is fabricated.
- Only the explicitly selected practical residual post_sync/noeq receiver/scene pairs are described; unselected scenes are not measured.
- The ideal 10/1/3 percentage-point directions are descriptive and are not advancement or stopping gates.

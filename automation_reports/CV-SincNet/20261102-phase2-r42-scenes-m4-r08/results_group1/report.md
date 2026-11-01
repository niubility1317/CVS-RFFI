# Phase 2 Group 1 对比结果

本轮固定 R42 与原版 LocalRidge，使用 Tri-SIG-TIME SSL 四个模型种子。8 套 MIX 的全部 16 个 K、N 格、以及 LS/MU/LU 三种单场景 K5/N20 均已完成预测和独立评分。三种单场景是在 MIX 结果公开后继续完成的固定实验，属于已暴露的开发评估，不作为新的独立盲测。

- [MIX 全 K/N：R42 与原版 LocalRidge 同路径及融合比较](../../20261101-phase2-r42-mix16-mgpu-m4-r07/results_mix_kn/report.md)。自然与共同 query 分开；双侧 DURA 的 K=1 四格平均提升不足 5 个百分点。
- [三单场景结果与 A0/B/C、support/query DURA 因子](scene_factor_report.md)，附 [因子数据](factor_stage_summary.csv)、[逐景配对差](scene_paired.csv) 和 [B 后 C 的资源测试](bc_resource_summary.csv)。
- [十个原生外部方法对比](external_report.md)，附 [原生得分、标签预算与实测训练步](native_scores.csv)、[同 raw 输入配对差](native_paired.csv) 和 [源模型成本元数据](native_source_costs.csv)。原生曲线只覆盖 MIX01；DA 原生只报告 B，CIL 原生只报告 C。
- [固定工作点的资源表](../../20261101-phase2-r42-mix16-mgpu-m4-r07/results_mix_kn/resources.csv)。该 profile 的 query 判决使用 CPU 参考实现，不能当成 r07 CUDA 调度的精确延迟；MACs、FLOPs、能耗未测。

原生 DA 与 R42 的 B 阶段目标旧类标签预算相同，但源训练配方和权限可能不同。原生 CIL 的 C 阶段为 0 个旧类目标标签＋N×K 个新类标签，R42/LocalRidge 为 6×K 个旧类＋N×K 个新类标签；二者不构成标签预算等价对比。DURA 与融合路径不混入外部方法的 raw 输入配对。

ISSL 遵照用户指令停止；Tweak、CSIL 没有合法绑定，均为 N/A。Hu 保留原 profile 身份，不重标为另一 scaled 版本。缺项和不适用阶段不填零；完整逐 pool/RX/episode/day/TX 聚合与逐单元对比保留在本机私有分析目录。

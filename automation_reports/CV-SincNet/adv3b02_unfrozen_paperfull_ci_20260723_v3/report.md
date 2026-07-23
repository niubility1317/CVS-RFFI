# ADV3B02可训练骨干CSIL与MoPC-HR全量对比v3

- 类型：`FORMAL_PAPER_METHOD_COMPARISON_BASELINE`
- 状态：`PREREGISTERED_LEGACY_LEO_CACHE_COMPAT_LOCAL_VERIFIED_REMOTE_SMOKE_PENDING`
- ADV3B02不冻结；运行CSIL与MoPC-HR论文完整增量机制。
- Stage2主方法资源和数据协议不约束外部对比方法；唯一强制条件是全部新类support/query叠加LEO星地信道。
- comparison-only legacy loader不要求旧缓存不存在的两个源溯源数组，也不伪造；仍验证17成员、文件SHA、LEO overlay、逐样本IQ/overlay SHA和顺序哈希根。通用Stage2 loader未改变。
- 完整矩阵：100 package、800cell、三场景2400结果行；先执行两方法K1/new2与K20/new20共4cell smoke。
- comparison builder SHA：`5bad3fa662193a2e16deaef036e3bc8d31385f654793549eca773eb17df692f3`；30项focused test、`py_compile`、`git diff --check`通过。
- 远端root：`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_unfrozen_paperfull_ci_20260723_v3`。

完整预登记、方法参数、路径、停止条件和追踪风险见工作区主报告：
`E:\type10-7\automation_reports\CV-SincNet\adv3b02_unfrozen_paperfull_ci_20260723_v3\report.md`。

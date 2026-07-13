# CVS论文级方法对比实验Git交接

- 根报告：`E:\type10-7\automation_reports\CV-SincNet\cvs_publication_comparison_20260713_122049\report.md`
- 追踪表：`analysis/cvs_publication_comparison_traceability_20260713.md`
- 统一协议：`docs/CVS_PUBLICATION_COMPARISON_PROTOCOL_20260713.md`
- 当前阶段：Phase1 seed713101三个baseline仍在训练；Stage2-C K5 seed713101中CSIL和MoPC-HR已完成，Orthogonal仍在运行；Stage2-B三种监督DA的低步数smoke已完成。
- Stage2-C已实现CSIL、MoPC-HR、Orthogonal Incremental统一runner、三种LEO测试、固定query的seeded nested K-shot split、sample score及四层详细统计。正式K5锚点已完成的CSIL/MoPC-HR分别得到`H_old_new=0.1808/0.2459`，单seed且Orthogonal未完成，禁止作正式排序。
- Stage2-B已实现ProtoNet CDA、MRIOR-SDA、DADDA-SDA统一监督runner。每个run只适应一个target receiver，仅有标签target-old support可训练；query只评估。三方法smoke均输出360条score、57条四层明细且全测试星地增强。
- 监督DA聚焦回归`10 passed`；新增runner提交`c157754`。N607同步hash、remote py_compile和dry-run均通过。
- 声明边界：smoke只证明机制与artifact契约；Phase1终局详细后评估、Stage2-C完整K/seed矩阵、Stage2-B五接收机K/seed矩阵和CVS同协议结果仍未完成，因此不构成论文最终性能结论或部署成功证据。

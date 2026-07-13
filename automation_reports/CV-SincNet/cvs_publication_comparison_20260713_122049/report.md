# CVS论文级方法对比实验Git交接

- 根报告：`E:\type10-7\automation_reports\CV-SincNet\cvs_publication_comparison_20260713_122049\report.md`
- 追踪表：`analysis/cvs_publication_comparison_traceability_20260713.md`
- 统一协议：`docs/CVS_PUBLICATION_COMPARISON_PROTOCOL_20260713.md`
- 当前阶段：Phase1 seed713101三个baseline仍在训练；Stage2-C K5 seed713101中CSIL和MoPC-HR已完成，Orthogonal仍在运行；Stage2-B三种监督DA的低步数smoke已完成。
- Stage2-C已实现CSIL、MoPC-HR、Orthogonal Incremental统一runner、三种LEO测试、固定query的seeded nested K-shot split、sample score及四层详细统计。正式K5锚点已完成的CSIL/MoPC-HR分别得到`H_old_new=0.1808/0.2459`，单seed且Orthogonal未完成，禁止作正式排序。
- Stage2-B已实现ProtoNet CDA、MRIOR-SDA、DADDA-SDA统一监督runner。每个run只适应一个target receiver，仅有标签target-old support可训练；query只评估。三方法smoke均输出360条score、57条四层明细且全测试星地增强。
- 监督DA聚焦回归`10 passed`；新增runner提交`c157754`。N607同步hash、remote py_compile和dry-run均通过。
- 声明边界：smoke只证明机制与artifact契约；Phase1终局详细后评估、Stage2-C完整K/seed矩阵、Stage2-B五接收机K/seed矩阵和CVS同协议结果仍未完成，因此不构成论文最终性能结论或部署成功证据。

## CVS同协议入口与训练预算决议（13:50）

- Phase1当前CVS候选不是普通监督基线：`phase1_dgleo_jointp0_leoweak8r2_20260713`以`ADV3B02_CORE90_SOFT_E200`为初始化，使用`rho_label=0.08`有标签与0.72源域无标签，包含三种星地信道训练视图和source-val-only选择。论文表必须显式报告其额外无标签访问预算，不能把它写成仅算法结构差异。
- Phase2提出方法固定为两个与项目谱系一致的入口：Stage2-B使用冻结CVS特征上的`CVS-OPGAC`监督support-only原型高斯校准；Stage2-C使用冻结`ADV3B02_CORE90_SOFT_E200`特征上的`CVS-qKNNV42`。后者参数固定为int8 support code、类内top-1、prototype权重0.45、old anchor0.001、8轮support-clamped label propagation权重0.025；unknown拒识不进入Phase2主线。
- 新增`paper_reproduction/cvs_aligned/cvs_method_runner.py`，强制单target receiver、三种正式LEO缓存、`seeded_nested` K={1,2,5,10,20}、support pool maxK=20后固定query、query标签不训练/不选模、sample score和四层明细、有限support-fit trace。
- Stage2-B与Stage2-C正式矩阵均从3个对比方法扩为“CVS+3个对比方法”，每阶段为`4方法 x 5接收机 x 5K x 5seed=500`行。新dry-run manifest分别生成500行。
- 训练预算采用双层报告：主表使用CVS任务下的common-budget，以控制训练计算量并保证同一数据、K、seed、receiver和query配对；论文原生epoch/batch只作为方法谱系敏感性附表，不作为主表直接混排。该选择避免Orthogonal的100/50 epoch与CSIL/MoPC短训练配置造成计算预算不等，但主表必须标为CVS extension而非论文原始结果。
- 本地验证：`cvs_method_runner.py`与matrix worker py_compile通过；新增CVS runner与matrix聚焦测试`4 passed`。当前尚未生成三场景ADV3B02正式feature cache，也未启动500行矩阵。

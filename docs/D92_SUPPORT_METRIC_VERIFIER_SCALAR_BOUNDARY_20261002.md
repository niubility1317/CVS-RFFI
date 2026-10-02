# SupportMetric验证器标量边界修复

本次纠正验证器内部类型边界，方法、参数、容差、训练状态与原artifact均不变。support analyzer及query scorer都以NumPy float64 eps重建相同的128ε·max(1,|f0|,|ftrial|,|rhs|)，导致JSON float除以tol得到NumPy标量；严格_equal只接受Python int/float，比例1也会拒绝。只对这一内部比值调用float，外部_equal类型规则及错误容差拒绝保持不变。

两个独立r02文件逐字克隆原入口，字节检查证明各只有该一处差异；原不可变release不改。真实非zero K3/new0 producer的完整archive/parent验证、trial_count>0及非零gradient/direction回归通过；同一试验容差与原公式的binary64逐位相等。analyzer12、query局部trial分支6、独立发布11用例共29通过。原43个analyzer fixture采用zero=True，未覆盖此分支，其PASS没有被当成真实运行成功。

新support_analysis_scalar_r02保持完整4/160/1800保存状态闭合、no query/source/fit、CPU2/BLAS2、普通用户和一次调用。原analysis r01 FAILED记录与产物保留，独立输出新目录。当前尚未发布r02或生成完整诊断。query实际runtime c7c持续运行，禁止热改/停止/重启；待全4/2400真实固定预测后，由单独只读结构闭合评分入口纠正原控制器的类型拒绝，保留原状态而不伪造成功，不评分subset。

新方法的方向本身来自完整LocalRidge/偏置/组间竞争对support损失的隐式导数与GGN，物理原型Gram+预测Fisher提供度量；本次修复没有新增参数搜索。理论只约束局部几何，不保证query准确率提升或星载省算力。实测A/B/C、K×新增数量、gain/drop/absGap/H和成本仍待完整分析。

# BranchLocalRidge独立P0/P1正确性审查

2026-09-29；审查者：独立Agent `local_env_reconcile`。审查对象为当前工作树的BranchLocalRidge核心、冻结配置、support缓存入口、运行与汇总流程，以及发布和分析归档。审查者没有实现这些方法或入口文件，仅负责本记录。

**结论：未发现具体P0/P1阻断问题。**这是静态正确性审查结论，不代表真实实验完成、科学假设成立或query性能通过。测试由主Agent独占执行，本审查没有启动另一套Conda或重复测试。

## 依据与核对结果

适用协议为工作区`E:/type10-7/项目.md`的checkpoint继承与Phase2输入、support/query边界，以及`E:/type10-7/tools/optimizer_workflow_contract.md`的八项最小流程；不使用工作树中的旧流程替代当前规则。设计依据为`docs/D92_NEXT_AFTER_ORBIT_20260929.md`。没有读取query结果、benchmark、handoff或实验索引，也没有访问真实IQ、特征、checkpoint或远端。

| 核对项 | 代码依据与结论 |
|---|---|
| 训练折内几何 | `code/cvsrffi/d92_branch_local_ridge.py:113`及`:126`只从传入训练support计算最近异类距离中位数、中心化、s0/sR和gamma；held只在拟合完成后评分。未使用source、query或old/new角色选择参数。 |
| 距离与退化 | 同文件`:66`以QR和非负平方和计算完整交互距离，精确同特征置零；非同特征下溢报错。C=1、全同特征、tau=0等价类核、极小正带宽均有固定路径；无jitter、截谱或候选回退。 |
| 目标与推理 | 同文件`:126`保持物理求和、lambda=1和中心化one-hot目标，Cholesky后两次三角求解；残差分母明确使用谱范数上界。`:218`逐样本使用固定训练状态，输出面对全部注册类；类字典序处理精确平局。 |
| 公平对照与物理拆分 | 同文件`:279`直接复用原branch/interaction公开实现。`:367`和`:418`对三臂采用同一物理train/held集合；按每类物理ID排序分折，并穷尽parent K个固定单样本锚点。真实K1不拟造held结果。 |
| 缓存与模型来源 | `tools/evaluate_d92_branch_local_ridge_probe.py:57`复用已有`load_support`边界检查。`tools/evaluate_d92_branch_support_probe.py:56`起核对p2_min_v1、VALIDATED_ONCE、capsule/split完整矩阵、checkpoint摘要绑定、EXACT_MATCH、scratch继承为空、冻结前target访问为false，以及cache成员和物理ID/标签一致性；本诊断不重新加载模型或生成特征。实际来源结论复用已有缓存证据，不重复builder验证。 |
| 计数与汇总 | `tools/summarize_d92_branch_local_ridge_probe.py:108`重建物理fold/anchor覆盖及三臂指标，`:221`先在parent内平均全部锚点，再等权汇总parent。4800个parent、1200个真实K1、3600个OOF parent、42000个proxy anchor和每臂7879200次proxy held出现次数相符；158400为名义最大分解次数，退化候选减少量单列。 |
| 科学判据 | 同文件`:196`分别检查标准OOF和proxy、parent K5/10/20、两个原基线下的新类/H改善及旧类下降不超过1个百分点；严格旧/新/H共同改善另报。没有把全部K×新增类数为正增设为门槛，没有自动晋级或选臂。 |
| 失败与产物保护 | 核心保留训练ID、scope、parent/train K、fold/trial和已完成stages。入口`:57`与runner`:65`拒绝已有输出；失败记录和已完成日志保留，lane失败不自动重试，也不终止其他任务。 |
| 发布与依赖 | 发布器和分析器使用各自私有transport实例及显式spec。`tools/publish_d92_branch_local_ridge_probe.py:14`与`tools/analyze_d92_branch_local_ridge_probe.py:10`的归档覆盖实际入口及其直接本地导入链；Torch导入留在未调用的export函数中。CPU路径删除旧GPU空闲门槛，仍使用独立run/output及主Agent唯一launch owner。 |

已阅读四个对应测试文件。测试覆盖显式特征/核代数、退化、逐样本不变性、物理与类别置换、原对照等价、训练/持出隔离、失败保留、缓存负测、完整汇总负测、禁止额外分层门槛和隔离目录分析依赖闭包。

主Agent提供的验证结果：首次四文件合并测试终端退出码为1，仅Windows的`longdouble`高精度参考计算失败，其他测试通过；首次失败记录保留。核心Agent将参考计算改为标准库Decimal的80位精度并将相对容差收紧到`1e-12`，没有放宽实现。随后精确距离参考、单类带宽诊断和禁止额外新增类分层门槛三个定点测试通过，退出码0，耗时1.26秒。审查者阅读了Decimal参考计算的修订；执行结果按测试owner的终端证据记录，不声称重新执行完整套件。

本次没有提出需定点复审的P0/P1发现，没有修改方法、配置、入口或发布脚本。远端预检、版本固定、启动与运行读回继续由主Agent按既有授权和八项最小流程完成。

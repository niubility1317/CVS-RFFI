# 单视图 REPL：U 的星地输入与恢复输出

两行单种子源域证伪；方法/预算/seed/数据/环境/CWD/命令/路径/GPU/owner/停止规则的唯一矩阵为 [experiment.json](experiment.json)。Git commit 随发布 runtime 记录。原完整 PAMF、一个 SIG_TIME、joint160 原默认6类头；各自从已核实同契约 scratch singleSIG_TIME ordinaryPL E20 独立+10+20，累计13L+37LU/E50/11100，不继承此前按目标分数挑选的变体。原三个角色学生和已完成六基线保持。

## 为什么增加这两行

旧 AllURDrop 的目标 raw practical 有正收益，但同G下降；恢复增强也多次出现源G提高、原始泛化退步。本次不复用那些目标分数调整参数，只将已有冻结的正负路线作为背景。当前角色研究的L-only源结果也显示，G训练本身不能保证原始IQ保持。旧 U-only 混合 pair 的 physical 半实际是 raw，ordinary 半是 G，因此半监督没有同时覆盖未恢复星地 H 和恢复 G。新两行只改变这个缺口，再独立比较低置信同包关系。

| 方法 | L | U混合physical半 | U混合ordinary半 | 新低置信损失 |
| --- | --- | --- | --- | --- |
| 既有REPL_G_U，仅复用参照 | 原真标签/As/ordinary/raw | raw | Ga/Gb | 无 |
| REPL_GH_U | 同上 | 原生T1 practical H | Ga/Gb | 无 |
| REPL_GH_U_Pair | 同上 | 同上 | 同上 | 仅raw95拒绝包的Ga/Gb detached互概率KL |

两行各seed2026092701，不是两个/三个种子。L的As、ordinary/raw、所有真TX及浅/中/joint跨RX保持；不对L套伪标签门控。U原始IQ当前学生产生.95硬伪标签，仅接受包参与原分类和跨RX关系。RX/day可见，U TX始终隐藏；低置信包不能因为RX已知就获得类别关系。

Ga/Gb继续使用相邻不同source practical景与独立随机实现，按物理ID/累计epoch/role/a/b固定；只有mixed ordinary半被替换。physical半恢复为原生T1 source residual/post_sync/noEQ practical，源session/RX/day保持。原六个U前向和分类剂量不变；L每update768包，U6×全批，原raw对和ordinary对、浅/中/joint跨RX不变。每学生新增G1701000包（stagewarm另4），新增未恢复U H850500包、承载两次既有F前向；没有额外F前向，配对行也没有额外G。H模拟额外耗时须实测，不声称相同总算力。

## 低置信同包项

缓存诊断在已见L的raw95拒绝子集显示，G高置信一致也会共同错误，不能直接放开硬标签；G-G detached概率的单位特征方向冲突比raw-soft少，但不保证骨干/AdamW或目标收益。来源见[完整缓存结果](../20261026-analysis-PAMF-SIG1-rejected-pairs-m1-r01/report.md)。

Pair行使用两个已有G logits，同一物理包，无硬TX和新跨RX正对：0.5×[KL(stop(pb)||pa)+KL(stop(pa)||pb)]，仅乘原始置信<.95的mask，再以整个原G半批平均。权重1.2=6×.25/1.25，匹配原mixed KL系数；不按少量拒绝包单独归一化放大。两侧targets各自detach，无熵最小化/锐化；一致的共同错误没有该项纠正信号。新项加入原U pending_loss，一起接受原U/L cap1，不绕过保护；0新forward/头/EMA/噪声。

## 源域冻结与后续

真实E20+同G的源L128/U256联合backward smoke先行，核实H/G输入差异、隐藏TX、原默认头、Pair对sig/head/body的真实非零有限梯度、G参数/buffers不变。PASS立即两行独立训练；任一系统技术失败停止新任务派发、保留健康已有子进程和所有产物，无自动重跑或其他owner动作。全球CUDA+pending每卡<2/free>=10000MiB才派发，本run允许GPU0..7。

训练结束每行自动完整源V27000×8评测，0V状态更新。复用既有PL-X3、REPL-SIG1、NoStyleNoRXE三份源评测，不重训/重新预测。原raw/H3各不得低于max参照超过.1pp，即−27/−81正确数；G3至少+243/81000（+.3pp）。两行全部完成后通过者按G/raw/H、并列优先无Pair选定，无通过不扩种子/目标。若此前角色家族也有通过者，在任何新目标访问前，按相同G/raw/H及较少G成本比较全部已完成source-pass候选，再冻结单一方法。严禁逐行目标分数反馈。

正式目标仍是同一方法三F种子、每seed原F0 168000物理包，raw/同原G clean+六practical，全部prediction固定后独立truth-last。绝对门槛raw clean≥85.981746%、raw六景>72.096197%、同G六景>78.074636%保持，不能将源域验证或两变体当作达标。当前本run仅源证伪；target/seed扩展只在固定源门槛通过后登记。

9项 focused检查和编译通过，唯一独立P0/P1审查发现的smoke/训练入口问题已修复并定点复核PASS，见 [local_verification.json](local_verification.json)。实际发布与运行证据随读回追加，不把提交返回值当作运行成功。

# R11：源选择的原生噪声软 MAP 正式确认

本轮继续完整恢复目标：TX 无标签训练的独立恢复器输入当前受损 IQ，输出同一份 IQ，全部 FFT、PA、时域分支由冻结原 F 图导出。用户最新验收为 low urban 准确率至少 70%，其余五个受损场景分别与同一原 F/seed 的原始 clean 准确率相差不超过 5 个百分点；恢复后的 clean 损伤另报，不降低 clean 参照来获得达标。

候选固定为 R10 的 native_map_prior 最后 E20/4940 权重。选择依据仅为完整1800源V的七视图：平均六场景 NMSE=0.02760034，原R03=0.02836140；原F50低城区82.1667%（原R03=79.8889%），clean下降0.0556pp，六TX低城区均改善，最差变化约+0.6494pp。R10的uncertain主候选仍保持原身份验收失败；R11是新登记的源域候选选择，不能追认R10已晋级。

源选择沿用已冻结的相对R03波形与身份要求：平均六场景NMSE不高于R03，clean NMSE≤0.02；源低城区accuracy至少+0.5pp，clean下降≤0.2pp，任一TX低城区变化≥−1pp。波形A不使用F/TX；身份B使用sourceV标签，不能把整个选择流程称为完全无标签。恢复器训练仍使用完整6300L/56700U的IQ、0TX标签、0F损失、0target，训练与源选择权限分开报告。

R10的direct/physical初值偏差保留；native/uncertain初值相同。R11不重新训练或应用后续seed修复，保持原native完整状态。它是完整恢复流程与R03的冻结性能比较，不能把两者差异全部归因于物理算子：native组件累计E60、R03累计E40，训练混合分别15%clean/85%residual和15%clean/60%residual/25%full。

固定测试原168000条目标物理记录、clean和六practical。高/中/低郊区及高/中/低城区全部单列，不增加随机信道副本或重合成。输入复用R05/reference原字节，原F50 CROSS_RX-s2026092701/defaulthead冻结；原R09三种完整读出仅作已冻结比较，不重新运行旧恢复器。所有新七视图IQ/logits/features/方差输出完整、有限、物理ID一致之后，独立CPU scorer才连接truth，报告accuracy、Macro-F1、repair/damage、每TX/RX/day、波形与配对clean特征及用户阈值差距。单seed、同模拟族，不是独立盲测或在轨证据。

本机实现与13项聚焦检查；复用R10 native候选既有独立P0/P1审查，不新增重复candidate/runner审查。真实2条L_s、原native和原F50的CPU no-query smoke是launcher第一步，PASS立即继续。N607允许池GPU0/1/5/7，优先空卡，最多一条本run GPU预测worker，每卡最多一条本任务worker；共享capacity4锁从容量检查保持到CUDA PID可见，不与子进程重复加锁。计入其他owner cap2且空闲≥12GB，不改健康任务。

run/release不可覆盖，唯一launch owner。来源/权限/形状/非有限/碰撞/无法合法预测的技术故障保存产物并停止本row，无性能停止、自动重试或目标反馈。实际commit、命令、环境、路径和预期产物以experiment.json为准；本轮新训练、适配、头拟合、信道生成均为0。

|ID|要求|状态|证据|
|---|---|---|---|
|N11-01|仅根据完整源V选择固定native，保留R10拒绝结论|implemented|源权限/预算/最差TX负测通过，待真实父元数据|
|N11-02|真实权重和原F50/defaulthead，0新增训练|implemented|严格加载器复用，待launcher真实源smoke|
|N11-03|完整七视图，预测完成后独立truth-last|implemented|复用R10已执行预测/scorer和覆盖负测，待正式产物|
|N11-04|urban与郊区各三档及逐TX/RX损伤|implemented|七场景/840源分层的复用通路，待新目标同row结果|
|N11-05|lowurban≥70，其余五场景距原clean≤5pp|verified|逐场景、原clean参照、边界值和降clean不得放宽的测试通过|
|N11-06|GPU共享容量、唯一owner、不可覆盖root|implemented|复用共享锁，待N607 preflight/post-state|
|N11-07|跨seed、独立物理族、逐细节真实可辨识性|deferred|本轮单seed冻结确认不能证明全部细节真实|

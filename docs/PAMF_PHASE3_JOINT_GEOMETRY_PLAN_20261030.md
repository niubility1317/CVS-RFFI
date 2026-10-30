# PAMF 旧类与未知类联合邻域诊断

protected160 源侧三组已完成。主 R1 保住 novel 三设备内部的跨 RX/day 几何（76.571429%，原 F0 为76.603175%），但源 R_U 只有9.269841%，正确known条件AUROC .852538351低于2pp损伤＋RU90的必要界.880215754。当前分数只整体移阈值不足；novel内部可分也不能证明它们与旧六类分开。本轮固定诊断两者的联合邻域，不训练新状态。

固定 run 为 `20261030-phase3-star-joint-geometry-s2026092701-r01`；唯一比较为 original F0 unit160 和主 protected160 R1。R1父为 `20261030-phase3-star-protected-open-s2026092701-r01`、code7dd90671968ddf58aab5a5b302b4d56371c8b9c9、2220新parts／native0，原F0 E50/11100、Tri-SIG-TIME SSL/defaulthead不变。低源拒识不是读取冻结模型诊断的额外门槛，也不晋级目标测试。

输入全部复用同 G10MIX01/P1：L6300的原fused160及四贡献缓存，和该R1已经导出的 novel3150原／新unit160。novel逐值复用、不重新前向、不读取旧source score来生成特征。L六旧类各1050，novel三个设备各1050，合计9450既有物理记录、每设备每源RX/day70；训练已见五RX（1-19、18-2、19-2、2-19、3-19）及三日期2021-03-08/15/23，仍不是未见RX/TX盲测。

只严格加载实际R1 checkpoint的新parts，核对原F0／DURA祖先、旧source数据句柄、训练配置及其source-only完成记录，不加载原F0/G权重。真实adapter checkpoint先执行两条合成特征无query smoke，PASS立即继续；之后只在L缓存上计算6300条adapter映射。原F0基线由同L raw fused160单位化；不执行真实或合成F/G前向、不生成新IQ。合成特征是接口算术，不当新增星地物理样本。

联合特征NPZ只含opaque IDs、公共RX/day、source角色pool（0=L，1=novel）及两种unit160，不含L真类、novel真TX或未知K。L先6300、novel后3150、全体ID互斥；角色来自既有采集划分，并非模型识别出的known/unknown。不能将本诊断说成无标签拒识或匿名注册。

独立CPU分别为两种表示固定以下候选集合，均排除自身、按FP64 cosine与opaque-ID稳定ties：

- Union-reference：全部9450条作为参考，分别all-other、cross-RX、cross-day、cross-RX-and-day四条件。
- Novel-reference：仅既有novel角色3150条作为参考，四条件相同，作为先前内部几何的严格匹配控制。

全部16套近邻索引在任何真实设备标签读取前固定。这里的novel-reference依靠benchmark角色组池，不能用于实际部署或声称系统已经找出未知集合。选择不使用真实TX、未知类数、按类配额或预测身份路由。只在索引固定后读取公开L标签和唯一novel私有truth，评分九设备逐类、known／novel角色和RX/day结果；输出联合9×9混淆、novel最近邻落known比例、正确未知同TX比例、落其他novel比例，以及相同novel query从旧novel-only到union-reference的配对变化。

整体、逐真TX、逐RX/day均使用清楚的分母和TX宏平均。Cosine类内外差与TX/RX/day方差仅为描述性诊断，非因果分解；联合候选数改变本身会改变邻域，不把指标变化直接归为训练效果。原/新feature使用同一bank、ID及mask；旧L参与过父模型和adapter训练，必须披露其reference身份，不能称known部分独立测试。

本轮不拟合聚类、分类器、阈值、prototype、域适应或注册状态，不消费V/OE/Us/target U/Q、正式RX24-18新数据。source标签只评分，不更新判决参数；source novel不会转为训练。全部联合近邻和角色混淆仅是同池参考诊断，不是未来query推理、R_U、A_N或Phase3成功。主R1不能被控制替换，source-only diagnostic／eligiblefalse，完整目标保持ACTIVE。

查重结论：此前local-support已测试L/OE 8近邻、pooled_relative和跨RX中位相对距离；conditional-support已测试默认预测身份条件的L子bank，主R_U分别13.111111%与25.111111%，均不晋级，且source逐行bank不能部署。本轮不重试这些拒识分数、不重调42组保护，也不把pooled控制的较高AUROC当新主模型。联合邻域结果先定位新信息是否需要，再独立登记下一学习机制。

执行仅需一个GPU adapter导出worker，实际free≥4096MiB、每卡≤2新增own，其他用户保持；随后CUDAempty／BLAS1 CPU scorer。按现有最小流程使用一个代码commit、聚焦负测、唯一直接P0/P1审查、最小预登记、单preflight／archive比较／compile和实际PID/CWD/argv/log核验。缺合法父或非有限／碰撞时保留、不自动重试；unknown状态先核原PID/root。预期产物为smoke/union NPZ/export complete、独立score results/complete及运行观察；不增加成员hash／签章／批准链。

# 单接收锚点条件传递：源域预登记

LOCAL_VERIFIED，尚未发布。源依据：TX×RX交互占组变化的主要部分；共享加性R预测主要改善分类头不敏感方向，完整软度量也未通过固定源推进规则。本轮检验依赖输入身份状态的接收作用是否能预测这些交互，不重复前述全局预测或物理特征读出。

固定三个原始纯CE E200父，仅用现有L_s6300和V27000缓存，不加载E40派生父，无新增身份前向/更新，无target。父已见所有源身份和接收条件，本轮cross-TX只针对辅助预测器，类别RX索引只覆盖源RX，不宣称未知RX推理能力。

沿原source-relation-split-v1物理ID哈希，将每个70包L_s TX/day/RX cell分成不相交35+35。每个折留两TX和一天，拟合只用其余4TX及其余2日。每个有向RX对有16条拟合观察：anchor A作输入，target为dest B−anchor B；反向交换半包再一条。独立输入/目标半包用于避免把anchor测量噪声的自包含当作可预测位移，独立性解释仍要求同组包误差条件，分开ID不自动保证统计独立。

四固定方法：zero；shared_rx（同拟合集TX/day均衡平均RX组差×.4）；conditional_joint（主候选，锚点mean joint160→完整位移）；conditional_state（40维物理state→完整位移，解释性对照）。对每有向RX对拟合带截距ridge，输入减拟合均值，并除sqrt(mean||X−meanX||²)，ridge=.1，输出整体乘.4；输入缩放仅一个标量，joint坐标共同旋转保持结果。零输入散度时条件部分精确为零，退化为相同加性预测，不利用验证挑尺度/秩。线性部分秩最多15，由拟合样本数决定，不手动按预测成绩选rank。

预测时只给定held L_s某一RX的A半包均值，不读取其他held RX的输入或目标，也不读取V输入。用该同一锚点预测所有5RX势，其中anchor势=0；任意g,h预测为p_g−p_h，因此同一锚点内环路一致。五个锚点全部固定评价，不挑anchor、不跨锚点平均来暗中引入多个RX输入。不同锚点的预测可以不一致，不能混称单一全局势。

同一组预测分别与held L_s B半包和独立V300包组均值核对；V只提供诊断终点，模型和预测在连接V终点前固定。V标签仅用于已声明的源分层统计及评分，不参与输入、标准化、fit或可靠度。L_s A/B仍同一合法角色，未创建V_cal/V_select。每种子18个held TX/day块×5锚点×10RX对=900次关系观察/角色；反复观察相同cell不算900个独立组。

主指标除完整160误差外，必须核对实际归一化并中心化cosine头的logit差误差，以及相对锚点P0最大非真类的固定竞争margin位移误差。统一原头scale30。margin竞争类在锚点输入上固定，预测器不消费y或rival；所有6类logit差同时报告，避免选择某个竞争类掩盖其他错误。几何目标不执行单位化、无IQ反事实，也不称分类准确率。记录预测最大组差、所有anchor/fold；若以后转训练仍需单位球可行性和身份保持检查。

推进规则预先固定：每seed的conditional_joint在V完整位移、中心化logit位移、固定竞争margin三个MSE上都严格优于shared_rx和zero，并且L_s B完整位移优于shared_rx。只有主候选可推进，其他对照不补位。全部报告，不为低效选择性重跑。本轮仅源诊断，不自动target；通过也只允许下一机制设计/核验，不意味着84%达成。

|ID|报告章节|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|T01|九步骤四|条件作用解释全局跨TX失效|transport_audit.py|verified|held TX/day误差|新条件算子研究|
|T02|六/九|预测输入不得含待预测位移|transport_checks.py|verified|改held destination不改变锚点预测；V不进入fit|只给一个RX输入|
|T03|六|组势环路与完整160目标|transport_audit.py|verified|环路和共同旋转|不直接输出训练目标|
|T04|三/九|作用必须进入身份决策|transport_audit.py|verified|logit和margin误差|预先固定primary|
|T05|执行|三seed源契约与不可覆盖发布|transport_publish.py|verified|本地检查/独立P0P1/远端读回|CPU2无target|

8项定向检查通过；独立P0/P1审查通过，数据/预测隔离、数学公式和发布路径无阻断问题。

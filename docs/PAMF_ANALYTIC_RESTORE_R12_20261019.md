# R12：PAMF确定性输入统计的TX无标签恢复

原完整目标保持：独立恢复网络输入当前received IQ，输出同一份IQ供PAMF全部时频/PA分支计算，lowurban accuracy至少70%，另外五个受损场景分别距同一原F/seed原始clean≤5pp，clean自身损伤单列。N607为训练默认，允许多GPU，标签权限和真实输入对应的TX身份保真先于外观/平均波形指标。

研究依据只取既有源证据和用户六类网络报告。R10源验证中，uncertain平均NMSE0.02747268低于native的0.02760034，但低城区身份81.3889%低于82.1667%，TX14-10相对native下降2.6578pp。该源矛盾在R11目标测试前已存在：更低平均波形误差不必然保持PAMF关心的输入细节。本轮不根据R11目标分组调参，不建立任何TX/RX/场景专属分支。

两组从同一R10 native_map_prior原末轮完整状态独立分叉。只继续训练原prior，coarse和作用估计器冻结；重置AdamW，新增20轮/4940步，batch256、LR3e-5预热247步后cosine至3e-6、WD1e-4、clip1、FP32。两个新增prior初值来自相同权重，不再依赖父构造的随机数顺序。原prior已20轮，本次后累计40；coarse20+action20+prior40=组件E80，不能写成总20轮。原F50历史E50/11100不更新，默认头不拟合或替换。

|行|目标|F训练信号|新增预算|
|---|---|---|---|
|pair_control|原IQ NMSE+0.1 FFT logpower L1|无F权重/输出/TX标签|E20/4940|
|analytic_views|相同pair项+1.0×四组归一化Huber的等权平均|无F权重/输出/TX标签|E20/4940|

四组直接复用原PAMF参数无关计算：32带×4通道正负频带logpower/logratio/asymmetry；4延迟×(1,3,5)阶复PA lift；7个频谱/圆性统计；公共L-STF参考响应的28个非显式CFO坐标。没有classifier/projector/可训练提取器；原PA内部clip2只用于同一数学基函数，不clip恢复网络输出。前端逐张量与原网络一致、零/正常输入有限梯度、clean目标停止梯度和参数数目0均有测试。它保持观测统计，并不把这些统计称为已辨识的TX硬件参数，RX/信道仍是混杂因素。

先在固定8192条合法L/U clean上计算四组每通道RMS，floor0.01，共63个source拟合标量；0optimizer/标签/F/V/target读取，额外L/U曝光单列。两组共用归一化来源；control不加载/执行无用解析损失分支。训练完整6300L/56700U，复用R07已有20轮received缓存：15%clean/85%residual，物理顺序seed2026101923，两组相同；这是重复原缓存的新优化阶段，不是新增噪声或新增物理样本。每组L126000/U1134000曝光，source V不更新任何持久状态。

固定末轮后立即生成原1800sourceV的七视图IQ/features/logits/方差，再由独立CPU scorer连接V标签，默认F50/原头；只在评估阶段加载F。原R10 source预测作固定输入/父参考，不重跑。source资格仅analytic主候选：平均六场景NMSE相对同预算control和父都≤1.05倍，cleanNMSE≤0.02；低/中城区accuracy相对control和父均≥+0.5pp；clean相对原输入下降≤0.1pp；低城区任一TX相对父下降≥−1pp，其他五场景accuracy相对父变化≥−0.5pp。波形A无F/TX；身份B使用sourceV标签，整套选择不是完全无标签，但恢复训练0TX/F。

全部源条件通过后自动对固定pair_control+analytic_views执行原168000物理记录×clean/三郊区/三城区；任何条件失败则本轮0target，保留完整负结果，不调阈值/扩轮救跑。所有七视图预测完整有限且IDs匹配后，独立CPU scorer才连接truth。原F50/defaulthead冻结，用户70%/5pp标准逐场景计算，逐TX/RX/day和repair/damage、波形、频谱、配对clean特征全部报告。目标结果不回流训练或选择。

真实2条L_s、原native和原F50的no-query smoke是launcher第一步，额外检查解析梯度到达prior而冻结父无梯度，PASS立即继续。新checkpoint独立schema保存完整coarse/action/prior、原R10来源及新增预算；加载严格验证，复用原预测/评分函数时只注入新schema加载器，不改变旧默认行为。

N607池GPU0/1/5/7，最多两个本run GPU worker、每卡一个，优先空卡，计入全部owner cap2且空闲≥12GB；共享capacity4锁保持至子CUDA PID可见，子进程不重锁。唯一owner、不可覆盖run/release；非法来源/权限、碰撞、非有限或无法合法预测时保留产物、停失败row、暂停未启动row，健康row继续，无自动重试或性能止损。科学协议、原物理角色和数据缓存不变，不重新验证或添加hash/receipt链。

|ID|要求|状态|证据|
|---|---|---|---|
|A12-01|复用原PAMF参数无关前端，0F训练|verified|逐张量匹配、0参数、梯度分离/有限性检查通过|
|A12-02|全L/U、共同父、同预算、冻结coarse/action|implemented|独立checkpoint全状态往返和权限负测，待实际训练曝光|
|A12-03|8192训练源、63RMS、0V/F/TX|implemented|权限/标量形状/floor负测通过，待真实校准|
|A12-04|新schema正确用于原七视图预测/独立评分|verified|自定义加载器全七视图接口检查，旧默认未改|
|A12-05|源固定阈值，未合格禁止目标|verified|身份/波形/最差TX和角色负测通过|
|A12-06|郊区urban各三档、clean、用户70%/5pp|implemented|复用已执行scorer与用户阈值函数，待本轮结果|
|A12-07|共享容量/不可覆盖/单owner/健康任务保护|implemented|待一次P0/P1与N607实际post-state|
|A12-08|多seed、独立物理族、逐细节真实|deferred|本轮是单seed源机制和条件性冻结确认|

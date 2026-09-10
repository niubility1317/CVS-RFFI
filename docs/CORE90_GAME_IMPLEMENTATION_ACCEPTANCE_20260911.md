# CORE90博弈追踪实施与验收

日期：2026-09-11。用户授权：按计划实现、检查不合理与未实现项、验收后发布实验，并特别要求落实原始报告注意事项。

## 实施范围与科学边界

新入口为`code/SSDG/train_core90_game.py`，独立包为`code/cvsrffi/game_tracking/`，未改变其他在跑实验的入口。历史模型取自Git`766c2ea8`，历史损失、阶段和参数取自20260701 ADV3B02快照及CORE90 launcher。新代码不继承旧checkpoint。

保留lite_d双分支、共享前端、160维身份/域特征和CORE90损失。9原生域合成验收模型有1048855参数；正式source RX×day域数由实际数据确定，不能把9域当正式数据声明。

历史数据比例`.10/.70/.20`和`joint_safe`选模不符合本次契约。新实验采用L/U/V=`.07/.63/.30`、固定source RX/day、独立训练seed与split seed、scratch-only、固定final E200。U不向训练器提供TX标签或TX元数据；V仅在最终四场景预测中只读使用；不加载target进行在线校准、控制或选模。

历史快照虽配置concat，但原训练块缺少对应实际路径。新入口实际执行clean+satellite拼接、卫星仅TX CE、E80启用和原固定场景阶段；这属于明确修复，不能称为历史逐位复现。历史checkpoint及历史分数不构成匹配基线。

## 原报告注意事项落实

1. **域头学得差不等于RX信息消失**：复制在线adv_head进行源训练池fit/monitor恢复，输出G_lag与S_domain；低频独立RX线性/MLP探针与原生RX×day任务分别命名。恢复头不回写在线头。单域、塌缩和无效梯度不解释成健康。
2. **真实梯度归属**：按参数对象去重共享前端；域保留头和对抗头明确区分。legacy_weighted保持GRL=1及外部有效权重；separate_head_scale独立编码器符号与头尺度，不把AdamW下调lr与缩放梯度混为等价。
3. **完整虚拟状态**：预测/校正使用同一StepContext及随机状态；临时参数、optimizer矩/step、BN和RNG回滚。只正式推进一次optimizer/EMA/prototype。BN提交原点完整closure的一次统计，不将额外U前向误删。
4. **算法严格命名**：EG、Heun、Optimistic、交替和head-lookahead分开；AdamW均标记所实现的隔离预测变体，不套用SGD双线性收缩结论。非有限主更新整次拒绝，已成功头追赶单独计数。
5. **高阶场不穿过GRL再求Hessian**：完整CORE90有符号字段在隔离副本上构造，包含当前aux/FISHR/卫星/U目标；动态detach目标的响应独立处理。算子失败记invalid，旧CE-only函数仍明确标子系统。
6. **源域信号与课程**：独立TX读出、跨RX中心间隔及clean/LEO一致性共同控制。阈值由至少3次合法source训练池审计校准后冻结。缺失/非有限consistency不能升级；审计自然过期不执行动作，但也不破坏跨审计确认序列。
7. **预算与反馈**：额外head/field/校正比例有窗口限制，审计与高阶耗时预检并按实际计费；超支保留记录。next_audit_interval实际改变审计频率。审计耗时使用预测准入，不能保证单次审计绝不超时。时间预算在epoch边界停止，实际超出量必须报告。
8. **分支与隐式响应**：CORE90的z_dom不是身份响应分支，融合接口标记NOT_APPLICABLE_NO_RESPONSE_FUSION。隐式追踪采用实际编码器位移，对最终线性域头作条件凸近似；阻尼CG、残差、拒绝及source monitor接受检查均记录。这不是整个非线性域头的精确最优反应。
9. **成本对照与证伪**：支持离线跨seed donor日程、固定均匀时间和随机置换，严格保留动作计数多重集合。recipient实际拒绝、HVP和wall time另报，不宣称动作计数相同就等于实际同成本。未完成donor不能生成未来日程。未发现滞后、普通更新足够和状态策略无收益均是允许结论。

## 逐项实现追踪

G=`code/cvsrffi/game_tracking/`。下列“验收”指实现或功能正确性，不代表E200性能完成。

|ID|实际路径|验收与限制|
|---|---|---|
|R01|G/legacy、config、parameter_roles|历史来源清单、共享参数去重、实际参数数及阶段loss|
|R02|G/data、config、runtime|scratch/源目标RX冲突负测；U隐藏边界；epoch恢复严格契约比较|
|R03|G/field、step_context|显式符号及两种尺度与实际一阶GRL梯度比对|
|R04|G/state、solvers、step_context|随机重放、buffer/矩回滚、单次持久提交|
|R05|G/solvers、runtime|同步、交替、固定k及不同lr配置；实际head追赶入口测试|
|R06|G/solvers|双线性SGD-EG与AdamW状态迁移；真实E41/E131目标|
|R07|G/solvers|Heun解析倍率与实际E80卫星目标|
|R08|G/solvers、runtime|历史保存；权重/课程/伪标签活动集合改变时重置。U阶段保守逐batch重置|
|R09|G/solvers|head-lookahead独立标签与参数块验收|
|R10|G/source_audit、data|整个TX/RX/day容器fit/monitor不交；审计与无审计训练精确一致|
|R11|G/source_audit|归一化CE指标、先验、平衡准确率及无效情况|
|R12|G/source_audit|独立线性/MLP探针及固定优化预算；自然run是否执行由日志判断|
|R13|G/gradient_audit、runtime|同一身份参数集的online/recovered方向；TX CE与完整labeled clean非对抗项分别命名|
|R14|G/field、jacobian_audit|实际完整目标显式场及JVP/有限差分；限定末端身份块+域头，非全网矩阵|
|R15|G/controller、runtime|当前版本detached特征只用于头追赶，实际提交次数验收|
|R16|G/response_tracking、field|线性解、负曲率/不收敛拒绝；实际编码器位移的条件最终线性层补偿|
|R17|G/capability|fit侧类中心、独立TX读出、一致性、塌缩负测|
|R18|G/capability|条件分支注册/门控测试；CORE90融合不适用|
|R19|G/curriculum|确认/滞回/冷却/限幅及一致性负测；真实入口课程事件|
|R20|G/controller、runtime|真实入口受控信号触发CATCHUP、CORRECT、HOLD；自然激活另查日志|
|R21|G/budget、runtime|窗口计数及预算耗尽回退；实际审计/HVP时间与预估超支分开|
|R22|G/coverage_audit、runtime|合法L_s覆盖、TX×RX/day关联；缺失SNR标N/A，不推断U真值|
|R23|scripts/build_core90_game_matrix、build_core90_game_replay|完整菜单、初始22row配置、其他seed冻结日程及拒绝未完成donor|
|R24|scripts/analyze_core90_game|完整JSONL及四场景/RX评分，缺失pending、坏尾行incomplete，实际资源和配对差异|
|R25|G/runtime、scripts/core90_game_evaluate|连续/恢复model、EMA、optimizer、proto及全部RNG精确一致；deployment预测精确一致|
|R26|SSDG/train_core90_game、scripts/dispatch_core90_game|CLI→更新→日志→checkpoint→四场景预测→独立评分闭合；GPU UUID及每卡2进程队列|

R13的完整非对抗参考是当前labeled clean目标，不包含该参考批次以外的卫星/U目标；完整实际训练目标高阶场属于R14。R19当前只控制LEO场景/概率一个轴，不在同一实验改变辅助loss权重或把E80悄悄提前。R24报告receiver/day分组及seed配对，不将重叠IQ窗口作为独立重复；两seed探索不能支撑稳定置信区间或科学晋级。

## 已解决的实现缺陷

独立P0/P1审查发现并修复：U元数据权限、虚拟BN提交、完整resume状态、头追赶grad=None语义、EMA在响应补偿之后更新、一致性未参与课程条件、审计预算未准入、稀疏审计未接入、CUDA数字枚举绑定风险。随后补充自然审计过期导致控制连续确认永不达标的问题，保留历史且禁止过期动作。

## 验证与发布记录

本地环境：已验证`C:/Users/lh594/.conda/envs/ssr-gpu/python.exe`，PyTorch2.10+cu128。测试使用真实CORE90结构以及专门解析模型，未并发Conda包装。

已完成合成源数据E1闭环：1次主更新接受、final checkpoint、clean及三个LEO场景各108条预测，共432条；全部预测文件关闭后才连接truth评分。该结果仅功能证据，不报告其准确率为研究收益。

合并验证102项全部通过：activation1、analysis8、audit/control21、field16、integration8、matrix10、replay17、resume2、solvers19。只有历史autocast弃用警告。相关Python编译通过。

真实骨干高阶验收发现并处理有限差分跨ReLU区域：h=.001跨1处，误差77.5124；按预登记规则缩小至h=.00025后跨区0，误差.00990065<.1，valid=true；没有丢弃loss或放宽容差。隐式补偿5次HVP、残差8.04e-7，source monitor CE从2.195294降至2.194922并接受。这些是合成数据数值/功能验收，不是正式研究收益。[逐项产物摘要](evidence/core90_game_local_acceptance_20260911.json)保留成本、失败试次、最终场检查及预测闭合信息。

独立审查原问题已定点复核；GPU UUID绑定按建议修正。Git OID及N607独立读回在发布后补充。

初始发布矩阵是B0/B1/B2/B4/B5/B6/S1/S3/S4/C1/C2×seed392002/392003，共22行，每行E200，scratch-only。它是source开发探索，不立即派发全部调优菜单；B3调优、离线donor成本对照和进阶H1后续按冻结假设执行，不能拿首轮target结果选择。

队列只利用空位并保留其他健康实验；每卡最多2个训练进程，启动期尚未创建CUDA上下文的PID仍占位。实际源数据scratch checkpoint smoke通过后才启动矩阵，失败保留产物且不盲目重试。运行完成还需E200与四场景真实数据评分，启动日志不能替代这些证据。

## 首次发布技术故障与修复

代码`1c495c1741c73736bc5dc1e972f3433602b10885`已推送且独立比对远端OID一致。N607首次release为`releases/core90_game_20260911_1c495c17`，run-root为`runs/core90_game_source_20260911_1c495c17`。传输SHA256、编译及真实源数据scratch checkpoint重载均通过；真实source有6个注册TX。

训练初始化失败：远端PyTorch为2.1.0+cu121，缺少新版`torch.amp.GradScaler`。完整读取22份日志确认全部相同异常，均未产生有效训练；队列active=0、failed=22，dispatcher独立进程查询不存在。旧目录和checkpoint/log全部保留。原队列状态COMPLETE仅表示队列结束，不是训练成功。

本地新增兼容工厂，旧版使用`torch.cuda.amp.GradScaler`，新版路径保留。缺失统一AMP工厂的回归测试由失败转为通过，连同连续/恢复及审计非侵入性两项复测共3项通过。累计独立测试用例103项；本次仅重跑受影响测试。发布smoke增加真实source完整E1目标及一次optimizer更新；两项预测前失败后停止后续派发并保存pending，避免扩大同类故障。定点独立P0/P1复查无新问题。该smoke不冒充远端AMP/EG/后期伪标签的完整验收。

修复后使用全新release/run-root，参数矩阵、seed、E200预算和source-only科学边界不变；不复用失败root或旧checkpoint。

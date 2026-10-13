# 六方向报告的探索与验证追踪

来源：用户附件 `26998ca7-fee0-41eb-b0ab-237c74c443b9/已粘贴的文本.txt`，2026-10-13；报告中的旧提交 a41c3423 只用于历史定位，当前工作树 cd995bb6 为实施起点。

用户已直接确认新训练仍每模型从头30轮，覆盖报告末尾E20/E20+20的预算建议。三primary 2026092701/02/03、普通cos头、160维身份、无显式CFO、U TX隐藏、V只健康、训练无卫星增强、84%原目标保持。历史30/E20产物均保留；不得静默切回含显式CFO的原E20。

先按报告对全部六个假设执行固定源机制诊断。固定同契约CE30父模型仅作诊断，0身份更新；每方向的数学性质、合法输入、对照、源适用条件和真实运行证据分别记录。适用条件不满足时保留可证伪的负结果并解释，不能盲目加大强度或堆叠六种机制。源V不作为微小性能排名门槛；训练配方、矩阵和源条件在目标访问前冻结。

|ID|报告位置|验收要求|目标模块/产物|状态|验证|边界|
|---|---|---|---|---|---|---|
|C1|全局/8|只合法L/U/V；U TX隐藏；checkpoint角色与完整来源匹配|design/common/登记|verified|r03完整18cells/各源方向JSON＋已有聚焦检查；仅组件/源诊断证据|复用已验证数据，不新增hash/ledger|
|C2|全局/8|独立消融、随机及成本对照、三primary、从头30|登记/训练/同row分析|verified|E30采样四组12rows完整日志79920updates/2016000原始cleanpredictions/216same-row指标独立重算|不把sameepoch称samecompute|
|C3|全局/8|冻结后逐row自动预测与独立truth-last；全部失败保留|dispatch/predict/score|verified|E30采样四组12rows完整日志79920updates/2016000原始cleanpredictions/216same-row指标独立重算|不按target调参/排名/重跑|
|C4|全局|普通cos160可供Phase2重新拟合；辅助训练状态弃掉|model/query检查|verified|r03完整18cells/各源方向JSON＋已有聚焦检查；仅组件/源诊断证据|当前Phase1 clean不作部署声明|
|M1|2.1/2.2|q定义及其跨RX可比较性诊断；承认received-IQ含RX|matching.py/源分组|verified|r03完整18cells/各源方向JSON＋已有聚焦检查；仅组件/源诊断证据|不称理想激励/同波形配对|
|M2|2.2|固定参考或stopgrad表示；cost含q及参考距离|matching.py|verified|r03完整18cells/各源方向JSON＋已有聚焦检查；仅组件/源诊断证据|当前表示不得靠改变matcher直接降低loss|
|M3|2.2|partial P非负、行/列caps、固定非零rho|matching.py|verified|r03完整18cells/各源方向JSON＋已有聚焦检查；仅组件/源诊断证据|严谨求解，不用完整匹配冒充部分匹配|
|M4|2.3|加权相对hinge；同TX跨RX正例及可比异TX负例|matching.py/objective|deferred|完整30轮身份训练/部署连接有意暂缓：源条件未满足|达到margin后不继续压缩|
|M5|2.4/8|随机/状态近邻/部分传输同loss/正例质量/成本；TX/RX/状态覆盖|源诊断/训练三组|verified|r03完整18cells/各源方向JSON＋已有聚焦检查；仅组件/源诊断证据|困难模式排除不能冒充解耦|
|P1|3.1|S2无损拆分、逆、1采样交换加carry|sampling.py|verified|r03完整18cells/各源方向JSON＋已有聚焦检查；仅组件/源诊断证据|采样位置不是载波相位|
|P2|3.2|共享处理严格尊重交换/carry；不接任意phase1x1|sampling.py/model|verified|r03完整18cells/各源方向JSON＋已有聚焦检查；仅组件/源诊断证据|有限窗口GN边界需明示|
|P3|3.3|无循环拼缝小偏移；共同内容区域、累计stride对齐后逐层差/间隔变化|sampling.py/源诊断|verified|r03完整18cells/各源方向JSON＋已有聚焦检查；仅组件/源诊断证据|不把正常位置移动算失稳|
|P4|3.2/3.3|只改源证据定位层；普通/归一化/低通/多相对照|model/训练矩阵|verified|E30采样四组12rows完整日志79920updates/2016000原始cleanpredictions/216same-row指标独立重算|低通可删TX细节，局部性质非整网不变|
|U1|4.2/4.4|独立source批次真实RX与随机分组更新迁移G矩阵，全部RX轮换|update.py/源诊断|verified|r03完整18cells/各源方向JSON＋已有聚焦检查；仅组件/源诊断证据|不永久留出RX不训练|
|U2|4.2/4.3|小参数块单步virtual update与完整可微outer|update.py/objective|verified|r03完整18cells/各源方向JSON＋已有聚焦检查；仅组件/源诊断证据|no_grad复制诊断不能冒充训练目标|
|U3|4.3|近似版本必须注明忽略项；部署无innerloop|update.py/query|deferred|完整30轮身份训练/部署连接有意暂缓：源条件未满足|训练计算单列|
|U4|4.4/8|真实RX冲突稳定超随机才进入完整训练；相同额外forward/backward控制|源条件/训练矩阵|verified|r03完整18cells/各源方向JSON＋已有聚焦检查；仅组件/源诊断证据|未改善不能默认权重不足|
|T1|5.1/5.3|共同A_r/b_r不接TX标签；heldTX拟合排除和真实独立端点|transform.py/源诊断|verified|r03完整18cells/各源方向JSON＋已有聚焦检查；仅组件/源诊断证据|随机包不是逐波形端点|
|T2|5.2|T_rr=I、逆、跨边composition；同时检查非平凡端点预测|transform.py|verified|r03完整18cells/各源方向JSON＋已有聚焦检查；仅组件/源诊断证据|identity也满足循环，不算成功|
|T3|5.3|限定rank/近单位范数/condition；六TX均值不足任意160x160|transform.py|verified|r03完整18cells/各源方向JSON＋已有聚焦检查；仅组件/源诊断证据|小参数不证明不拟合身份|
|T4|5.4|每TX平移/共享低秩/等幅随机；仅训练增强原cos，部署弃算子|objective/训练矩阵/query|deferred|完整30轮身份训练/部署连接有意暂缓：源条件未满足|未知RX不可查表inverse|
|D1|6.1/6.2|同中层实值中心化Ctau及antisymmetric低秩读取|directed.py|verified|r03完整18cells/各源方向JSON＋已有聚焦检查；仅组件/源诊断证据|已有complex crosspath lag不是本次实现|
|D2|6.1/6.2|固定序列倒序transpose/sign；tau0=0；低秩公式因子2|directed.py|verified|r03完整18cells/各源方向JSON＋已有聚焦检查；仅组件/源诊断证据|不承诺原IQ倒序整网性质|
|D3|6.3/6.4|固定embedding增量判别、错误补救/新增错误跨RX独立包重复|directed.py/源诊断|verified|r03完整18cells/各源方向JSON＋已有聚焦检查；仅组件/源诊断证据|RX/信道也有记忆，非天然TX不变量|
|D4|6.4|同容量静态二阶与打乱有向对照；打乱保留边缘与C0|directed.py/训练矩阵|deferred|完整30轮身份训练/部署连接有意暂缓：源条件未满足|收益不依赖时序不能归因有向动态|
|D5|6.4|160维小残差；仅末层零初始化、上游正常初始化，后续梯度恢复|model|deferred|独立残差组件数学/梯度已测；源条件不满足，未连接完整身份训练|额外部署参数/计算实测|
|S1|7前提/7.3|U与L候选TX集合一致，真值隐藏；先构造集合后用合法L检查|sets.py/源诊断|verified|r03完整18cells/各源方向JSON＋已有聚焦检查；仅组件/源诊断证据|不看U隐藏标签调整|
|S2|7.2|精确logsumexp集合loss、singleton CE、全集loss/grad0|sets.py|verified|r03完整18cells/各源方向JSON＋已有聚焦检查；仅组件/源诊断证据|集合内比例可能改变|
|S3|7.3|保守可信判断集合，不盲目取交集；coverage与size同报|sets.py/源分组|verified|r03完整18cells/各源方向JSON＋已有聚焦检查；仅组件/源诊断证据|source覆盖非target保证|
|S4|7.4|hard/soft/set相同U/teacher/流/mask/预算；含模糊但小集合|objective/训练矩阵|deferred|完整30轮身份训练/部署连接有意暂缓：源条件未满足|只高置信singleton不检验该假设|
|A1|8|六假设各有真实源机制结果、适用/失败条件及后续说明|analysis/report|verified|r03完整18cells/各源方向JSON＋已有聚焦检查；仅组件/源诊断证据|不重命名已存在机制|
|A2|8|报告实际训练/aux/forward/backward/参数/时间成本和负结果|logs/同row分析|verified|E30采样四组12rows完整日志79920updates/2016000原始cleanpredictions/216same-row指标独立重算|84%仍须独立实际三seed验证|

当前32项pending，0verified/deferred/rejected/blocked。源码只读审计已定位实质缺口，尚不等于实际实现或正式性能；后续按实际证据更新每项。

实施状态更新：六方向 source-only 模块、固定 matched 控制与 source 判据已落地；全部本地聚焦数学/负例/合成入口检查 PASS，真实随机 native 架构的 sampling 替换与 identity-only functional 更新接口 PASS。一次独立 P0/P1 PASS。当前实际 source 18 cells 与三 checkpoint smoke 尚未执行；正式 scratch30 对照和目标测试仍待源条件，不把组件通过改写为性能通过。32 项验收继续等待各自真实证据，完整结果将逐项更新。

## 2026-10-13 接续实施：采样结构独立训练

P4/C2/C3：准备四组（原池化、逐点 GN、固定低通＋逐点 GN、保留两相位＋carry＋逐点 GN）×三主种子从头 30 轮纯 CE，6660 次更新，普通 cos160、无显式 CFO、无 U/教师/EMA/卫星分支，固定末轮后自动 clean prediction→独立 truth-last。不改目标分数驱动的强度，部署成本与结果分别报告。来源是当前六方向固定源诊断；采样结构第一种子适用，三种子完整适用性仍待 r03，不据此宣称性能提升。

当前验收状态（覆盖旧32项pending声明）：{"verified": 22, "implemented": 4, "deferred": 5, "pending": 1, "rejected": 0, "blocked": 0}。verified限该项的数学/组件/固定源诊断证据；不代表84%或全方向端到端训练完成。最高风险仍是采样改变的实际目标识别收益未知。

2026-10-13最新验收：26verified、6deferred、0pending/rejected/blocked。verified限定各项适用范围；五方向完整身份训练及有向残差正式连接因原源条件未满足而deferred，不称全方向正式性能已验证。采样四结构12条E30训练及clean测试已完成，但84%仍未到达。

当前用户预算覆盖：后续epochs由Agent自主选择，原则上每model不超过50；在发布前固定同预算对照。原30轮实验不追溯改写。

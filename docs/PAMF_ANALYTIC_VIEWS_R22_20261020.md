# R22：固定解析观测量的配对恢复约束

状态：ANALYZED / SOURCE_NOT_QUALIFIED。本轮没有新正式目标集成绩，持续目标尚未证实。 原无标签、F冻结及70%/5pp验收定义保持。

持续验收使用原冻结 F50 seed2026092701/defaulthead：正式 low urban ≥70%；其他五个受损场景各自准确率 ≥79.6761904762%，即距原始 raw clean 84.6761904762% 不超过 5 个百分点。restored clean 损伤单列；本轮源域成绩不替代正式测试。

假设只依据源证据：R15 的通用 log-power 辅助损失没有优于 R16；R17 的三步不共享先验优于这两个源参考；R19 的源 low/mid urban 为88.2778%/92.5556%。R20 同批源诊断显示真值路径与自由运行差距；R21 直接终点目标相对路径目标使源 low/mid 增加7.0556/3.9444 pp、mean6 NMSE下降19.5375%，但仍未超过 R17/R19，且训练调用量更高。这些事实不支持继续只根据速度路径损失选网络。

历史 R12 已验证过四组解析损失，不能遗漏其负结果或把本轮写成首次尝试。R12 从单步 R10 prior续训、同时保留0.1 FFT项、使用1.0解析权重及63个每通道RMS尺度；解析源low urban82.9444%与同预算pair相同，mid urban89.9444%比pair低0.3889 pp，身份验收未通过。本轮在R17三步scratch结构内做新的匹配对照，按坐标源变化标准差缩放，并固定较小0.01权重、无FFT辅助项；不含PA polynomial lift或DAC统计。R22与R12之间改变了多个条件，不把两轮直接差值归因为单一因素。[R12完整记录](../automation_reports/CV-SincNet/20261019-phase1-restoration-analytic-views-selfsup-s2026092701-m2-r12/report.md)。代码与测试使用独立`pamf_analytic_unrolled_restore`名称，保留历史`pamf_analytic_restore`。

已有 R14 oracle 诊断保持原结论：已知作用＋模拟器白输入方差代理的源平均 NMSE 相对 native下降15.4053%，low urban仅+0.5556 pp、mid urban−0.0556 pp，未支持优先提升作用估计。本轮不重复该矩阵，不把 oracle 分数当全模型理论上界。新假设是：IQ 平方误差之外，对原接收观测中的固定解析量施加配对约束可能保留更多差异；这些量含 RX/信道混杂，不能预先称为 TX 专属指纹。

两行 `iq_control`、`analytic_views` 均使用 R17 三步不共享 Core9、每步两次软 CG、coarse 锚定及相同完整初值，只继承 R07 冻结 coarse/估计作用/原生方差代理。新 prior 从头 E20/4940，不加载 R17/R19/R20/R21 的训练后 prior。两行完整部署图相同、同参数量（2507526可训练／4665073总参数）；不新增分支输出、场景路由或新分类头。

训练均为全部合法6300L/56700U、每行20轮、batch256、L126000/U1134000曝光，R07原20轮缓存、15%clean/85%residual、data2026102001/cache2026101912/model2026092701。AdamW2e-4至2e-5、warm247、WD1e-4、clip1、FP32，固定末轮。组件 E60 与原 F50 监督 E50 单列。TX标签、F权重／分类／嵌入输出、target 和源V持久拟合均为0。允许的 clean→received 模拟配对必须披露，名称使用“TX无标签模拟自监督”，不声称无clean或整条识别链无监督。

`iq_control` 只计算原逐包 IQ NMSE，不执行无用解析分支。`analytic_views` 计算同一 IQ NMSE＋0.01乘以一个解析描述量损失。描述量分为三个组，组内按坐标平均、组间等权：

- 原 raw-FFT 镜像压缩量：32频带的正／负频功率log1p、对数功率比、非对称性，复用原lite_d实际配置、数学定义及epsilon1e-8；不是重复加入与时域L2等价的完整复FFT L2。源类通用默认64带不等于本F所用lite_d的32带。
- 原谱PA统计与圆度：edge ratio、regrowth ratio、spectral kurtosis、abs(E[z²])/(E[abs(z)²]+epsilon)，不使用可学习Sinc或PA投影。
- 公共L-STF解析参考响应：原29维响应去除显式CFO坐标27，保留28维；固定80:160窗口、20点周期、4周期均值、320延迟网格、12个占用tone、相同归一化与软界限。不使用训练好的参考投影或分类器。

这些量必须由独立参数为空的数学实现生成，训练模块不构造或加载 F。与原数学实现的一致性仅在聚焦检查中验证，不给训练注入 F 权重或预测。原本存在的解析量截断只属于该数学变换，不裁剪恢复 IQ。梯度按原离散延迟路由分段传播；不宣称整个硬选择可微。所有源验证和正式推理的输出均只依赖当前 received；clean 只进入训练目标，不进入恢复前向。

解析坐标尺度在训练前固定：只读取原源训练 clean 缓存前8192条及其合法角色，不读取TX/V/F/target，以FP64累计各坐标标准差并使用max(std,0.05)。解析差值除以此尺度后使用beta=1的SmoothL1，三个组等权。尺度仅用于训练损失，无推理依赖；8192条额外无标签统计曝光、实际L/U组成与0优化器更新单列，不混入训练更新曝光。权重0.01、尺度下限0.05及组权重不依源V或目标反馈修改。

实际实现须验证：完整新初值和初始部署输出相等；三步输出与各解析组至最终IQ的梯度有限；IQ项和0.01解析项的输出梯度量级分别报告，不以loss数值代替梯度；冻结上下文无梯度；逐包/合批一致；仅IQ行不执行解析损失。应覆盖零/近零IQ、近零谱点、已知相位/增益、变长或非有限输入拒绝、尺度角色/数量及checkpoint schema隔离。只做与本行为直接相关的聚焦检查和一次独立P0/P1，不增加额外审批。

源资格保持 R21 五固定参考native/R15/R16/R17/R19：对每个参考mean6 NMSE≤1.05倍、clean NMSE≤.02、low/mid urban各≥+.5 pp、相对原clean准确率下降≤.1 pp、low urban最差TX下降≥−1 pp、其他五场景最差变化≥−.5 pp。两行原1800条×七视图预测固定后独立CPU连接源truth，均完成后按low/mid均值、最差TX、最差RX、名称选一个；不合格则0target。

若选定则自动原168000物理记录×七视图、原F50/defaulthead预测，全部固定后独立truth-last评分；完整报告郊区/urban逐场景、TX/RX/day、repair/damage、波形与配对clean特征、实际成本。R17/R19缓存正式结果只在新评分完成后描述比较；不依任何旧目标分数设计、选模或调参。已暴露基准不称新盲测。配对身份CPU诊断复用既定工具，不新增部署决定或拟合。

N607普通账户、新不可覆盖root/release、原R21所属进程全部退出后发布；池0/1/5/7、最多两条本任务GPUworker、每卡一条，全owner cap2/free12GB，共享capacity4锁到CUDA PID可见。真实2Ls/checkpoint/noquery入口PASS后立即训练与自动测试。非法来源、输出冲突、非有限或预测不完整时保留产物、停止失败行并暂停pending推进；不救跑、不重启健康任务、不动其他owner。

|ID|要求|状态|
|---|---|---|
|A22-01|独立固定解析量与原数学一致，0训练好F信息|verified：本地聚焦检查|
|A22-02|匹配完整初值/部署、仅改变配对损失|verified：本地聚焦检查|
|A22-03|全LU、训练尺度权限、0TX/F/target、固定预算|verified|
|A22-04|五源参考及固定排序、选定行隔离|verified：本地聚焦检查|
|A22-05|真实checkpoint入口、发布、资源与负测|verified：真实入口与运行后证据|
|A22-06|完整源七视图、梯度和成本证据|verified|
|A22-07|条件正式七视图、配对身份及Git交付|verified：源资格失败，按登记保持 0 target；正式配对身份诊断不适用|
|A22-08|多seed、独立物理族与逐细节身份归属证明|deferred|

当前 7 verified、0 pending、1 deferred、0 blocked；另1条REJECTED_EXTRA_GATE已拒绝且未新增门槛。36项聚焦检查、编译及一次独立P0/P1 PASS；真实源checkpoint入口、全LU训练、全部条件评价及退出均核实。

非阻断记录：REJECTED_EXTRA_GATE——通用TDD技能的全仓库测试要求由项目明确的聚焦负测协议覆盖，未新增全面测试门槛。R12模块、原测试与历史产物保持原样；新模块和测试独立命名。

实际结果：本轮没有新正式目标集成绩，持续目标尚未证实。最高剩余风险为单seed跨RX泛化及逐细节身份真实性。[完整报告](../automation_reports/CV-SincNet/20261020-phase1-restoration-analytic-unrolled-selfsup-s2026092701-m2-r22/report.md)。

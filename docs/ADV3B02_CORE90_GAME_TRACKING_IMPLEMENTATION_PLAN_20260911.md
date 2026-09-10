# ADV3B02 CORE90源域审计、对手追踪与课程控制实现计划

日期：2026-09-11。状态：PLAN_ONLY。本次完成报告到代码的实施设计，不修改训练代码，不启动实验。

## 1.目标与依据

在历史ADV3B02 CORE90的方法结构、损失组成和训练阶段之上，完整建设“源域诊断→有限域头追赶→选择性预测—校正→能力驱动课程”的训练系统。先保持网络、损失和课程固定，逐层证明新增机制的作用。进阶Jacobian诊断、Optimistic更新和隐式响应追踪也纳入实现范围，但独立开关、后置验证，不默认为最终组合。

原始设计：用户附件`pasted-text.txt`，章节1—10。方法基础：历史`code/snapshots/phase1_adv3_mechanism32_queue_20260701/{train_ssdg.py,losses.py,launch_phase1_adv3_mechanism32_queue_20260701.sh}`；结构参照[CORE90技术报告](CVS_ADV3B02_QKNNV42_TECHNICAL_REPORT_20260709.md)。当前科学权限以工作区`E:/type10-7/项目.md`为准。

本计划要求完整覆盖设计中的可执行模块；双线性推导作为解析验收，局部最优反应假设作为理论边界。不会将“探针不可读出”写成互信息为零，也不会将一般深网训练写成已有收敛定理的直接实例。

## 2.已核实的基线与必须显式处理的差异

|项目|核实结果|实施决定|
|---|---|---|
|历史骨干|技术报告记载`lite_d`双分支，`z_id/z_dom=160`，身份`feat_joint`、域`feat_imp`，共享`sinc/hf`前端|保留结构；以历史版本解析配置及参数别名核对为基线恢复首项，不能拿当前默认值替代历史|
|历史核心机制|TX/CosFace、域保留、身份域对抗、正交/跨域项、FISHR、原型、紧致/尾部几何、proxy unknown、soft mixup、source episode、EMA伪标签|全部进入固定损失表；记录阶段有效权重，不把当前A1/ECRS/DAOT新增项混入|
|CORE90标识|`proxy_core_quantile=.90`、`proxy_accept_quantile=.85`、`proxy_vaccept_cvar_alpha=.30`、core权重`.45`|不是90%准确率，也不是只保留一个同名checkpoint|
|训练|历史launcher为E200，label130+pseudo70，seed392002；AdamW默认lr=.0002、weight_decay=.0001|作为机制对照初始配置，随机seed与数据划分seed分开管理|
|对抗梯度|历史训练`grl_lambda=1.0`，总损失乘`cur_w[adv]`，launcher基础`lambda_adv=.35`；全模型单AdamW|历史兼容模式保留编码器和域头同时被外部权重缩放的语义；归一化域头CE是单列对照|
|域保留与域对抗|当前模型`dom_head(z_dom)`；`adv_head(GRL(z_id))`；可选`tx_adv_head(GRL(z_dom))`|恢复测试针对`adv_head`；不得误用`dom_head`准确率；可选反向TX对抗须核对历史是否激活|
|增强|历史launcher清楚指定clean+satellite、卫星仅TX CE，权重.68、一致性0、E80计入；E1/41/91切换三LEO弱场景概率|固定课程对照完整保留；能力课程单列方法，显式披露对固定日程的修改|
|数据与选模|历史launcher为`.10/.70/.20`、`joint_safe`、周期测试；当前协议为`.07/.63/.30`及source-only|正式新实验继承方法但按当前契约从零训练，预登记final E200；历史数值仅背景，不作为匹配性能基线|
|旧权重|历史`joint_safe`选模路径消费测试统计|不作为正式新run初始化。若后来指定其他来源，逐一核对完整训练契约和祖先；不因可加载而放行|
|工作区|当前Git承载面有多项他人暂存/未暂存修改；读取时HEAD=`0f91ac936c922af39b56cbd2da1791dfe1591520`|本次只提交本计划；代码实施使用独立、核实基线的工作树，保留现有修改|

完整历史模型文件未独立保存在20260701训练快照中；技术报告自身也披露模型结构证据来自Git谱系。实施阶段需恢复可核实的模型版本及依赖；本次不声称已完成历史逐位复现。

## 3.参数和梯度场：先把参与者定义正确

新增`code/cvsrffi/game_tracking/`包。`parameter_roles.py`按参数对象身份去重，列出共享前端、身份专属层、TX头、域保留专属层、`dom_head`、`adv_head`及其他实际激活头。共享参数只允许一个optimizer owner，但可接收多个损失贡献。

`field.py`显式输出每组梯度。普通历史模式先与原始GRL一阶结果比对；高阶计算用不经过GRL的普通可微前向构造有符号梯度场。编码器上的对抗项是负RX梯度，`adv_head`上的更新是正RX梯度；域保留CE对域分支与其头是合作项。共享前端还要累加其余损失，不能把整个模型机械写成两玩家零和博弈。

先提供`legacy_weighted`尺度策略：保留当前阶段有效外部权重对双方的作用。另提供`separate_head_scale`实验策略，独立设置编码器反向系数、域头CE尺度及双方lr；不要用调整lr声称与AdamW梯度缩放严格等价。无对抗对照将该对抗场关闭，域保留分支仍按原定义训练。

验收：参数组无遗漏/重复、共享参数只更新一次；逐损失梯度符号正确；冻结头权重仍可对输入求导；`detach`仅用于头追赶；关闭新模块时固定随机状态的一步损失、梯度、参数及持久状态与匹配基线一致。

## 4.批次事务与完整求解器

`step_context.py`一次构建训练上下文：物理样本ID、clean/LEO张量、MixStyle配对与随机量、Dropout随机状态、伪标签和mask、EMA教师快照、原型读取状态、所有loss权重、课程状态。两次场求值不得另取batch、换增强或重算另一套伪标签。

`state.py`管理临时参数、AdamW一阶/二阶矩和step、BN buffers、AMP scaler、RNG及共享权重别名。虚拟求值不提交EMA、prototype、GroupDRO或其他loss内部状态；这些状态由无副作用的候选更新返回，在成功正式step后提交一次。BN采用预先明确的“原点单次统计提交、其他求值使用隔离buffer”策略；记录该随机有状态实现选择，不将它描述成唯一标准EG。

求解器菜单：

|模式|算法定义|关键验收|
|---|---|---|
|普通同步|在原点一次求值并正式更新|历史兼容回归|
|交替/固定k|编码器不动，域头更新k次，再用新头更新其他组|明确k含基础更新；避免域头被额外重复step|
|完整SGD-EG|预测`w_tilde=w-eta F(w)`，从原点用`F(w_tilde)`更新|整个相关场参与预测与校正；不是两次连续step|
|Heun/RK2|同预测点，从原点用两次场的平均更新|不得与EG混名|
|AdamW-EG变体|以旧optimizer状态的隔离副本生成预测位置；在预测点求梯度，再从原参数及原optimizer状态执行一次正式AdamW|正式矩/step/weight decay只推进一次；明确标为选定的AdamW外推变体，SGD收缩结论不自动适用|
|AdamW-Heun变体|隔离预测后合并两次原始梯度，仅一次正式AdamW|不称为对完整Adam动力系统的严格RK积分|
|Optimistic|`2F_t-F_prev`，首次使用普通场|课程/对抗权重/解冻/伪标签集合显著变化时重置历史；保存重置原因与历史|
|head-lookahead|只预测域头的分块近似|独立命名、预算统计；不作为完整EG交付|

纯双线性FP64验收：同步半径平方倍率`1+eta²`，EG倍率`1-eta²+eta⁴`，Heun倍率`1+eta⁴/4`；交替更新行列式1且在指定步长区间呈有界振荡。AdamW另按手算状态迁移验收。

任一预测/校正出现非有限值时，整次耦合主更新不提交，恢复临时状态并记录具体阶段；AMP溢出可降低scale，但不得留下部分玩家参数已更新。已独立成功的头追赶动作单独计数，不能伪称整个迭代零更新。

## 5.源域审计与可辨识性诊断

`source_audit.py`将合法训练池中的采集组确定性划为probe-fit和probe-monitor角色，二者不得共享原始记录/相邻重叠窗口；各source RX仍参与主训练。这是探针内部隔离，不宣称monitor对主干训练完全未见。划分固定、披露样本量，不新增永久保留RX。

当前协议禁止在V上反向传播或更新持久模型状态。因此探针拟合绝不使用V；在线控制信号也来自上述训练池审计。V保持单一、只读，供离线source校准/选择，不能拆成新的V角色，也不能用V拟合branch读出头。没有采集组元数据时标明独立性未证实，先落实可核实分组，不能用随机窗口划分冒充独立证据。

在固定编码器及固定参考增强上：复制当前`adv_head`，以统一初始化optimizer和固定预算拟合，随后在独立probe-monitor同时评价在线头与恢复头。探针副本不回写在线头。低频加独立初始化线性头和受控MLP，排查容量/优化不足。

输出公式与附件一致：

```text
G_lag = max(CE_online - CE_recovered, 0) / (H(pi) + eps)
S_rx  = max(H(pi) - CE_recovered, 0) / (H(pi) + eps)
c_IA  = dot(g_id, g_adv) / (norm(g_id)*norm(g_adv) + eps)
rho_AI = norm(g_adv) / (norm(g_id) + eps)
```

同时输出未截断CE差、普通/平衡准确率、RX先验熵、多数类准确率、先验采样准确率及有效RX数。若只有单RX、样本不足、梯度范数近零，结果标记无效而非健康。历史域标签若为RX×day，必须分别报告原生domain任务与RX探针任务，不能把domain类数当RX数。

`gradient_audit.py`在同一参数集合/同一固定批次比较在线头与恢复头对编码器的对抗梯度cosine、差异范数，以及身份梯度冲突和范数比。身份CE与完整非对抗目标分别记录，避免把全部辅助梯度叫作身份CE。禁止把所有冲突直接投影删除。

`coverage_audit.py`报告合法可见的TX×RX覆盖、day、SNR（缺失则N/A）和clean/LEO分配关联。TX相关统计仅取L_s或合法只读V，不读取U_s隐藏TX。低频条件RX探针可用于诊断，但不暗中替换现有无条件对抗目标。

## 6.有限追赶、选择性校正与计算预算

`controller.py`只消费带时间戳、编码器版本和有效性标记的源域审计，输出`NORMAL/CATCHUP/CORRECT/HOLD_CURRICULUM`及理由。审计过期或无效时不解释为低滞后，采用预登记普通更新并暂停升级。

|状态|动作|
|---|---|
|G_lag高且恢复后仍有RX读出能力，身份保持|有限头追赶；不直接提高对抗强度|
|域头基本跟上但固定参考方向持续失衡|在预算内触发完整EG；梯度转向本身不冒充旋转证明|
|G_lag和S_rx低、身份保持|普通更新，降低审计频率至预登记下限|
|身份下降/间隔恶化|暂停课程升级，记录梯度与覆盖诊断；不以低性能停训练|
|身份分支/拟融合分支能力不足|延迟对应课程动作，不到epoch强开|

头追赶采用当前编码器版本的detached特征缓存，仅更新在线`adv_head`，k有上限；缓存编码器版本变化即失效。probe-fit的恢复优化与在线catch-up是两个独立动作，避免把monitor拟合进线上域头。

`budget.py`显式限制窗口内额外头step、额外场求值、审计耗时和校正比例。阈值、窗口、滞回、冷却和预算由source基线重复波动校准后冻结；报告未给定的数值不编成“理论最优”。预算耗尽回到普通更新，保留触发被抑制原因。

逐动作记录状态前后、实际k、触发原因、预算、编码器版本、预计/实际成本及是否成功提交。一次迭代明确先审计决策、再有限追赶、再选择主step，最后更新课程；若组合触发，所有额外成本均计入。

## 7.能力驱动课程与分支条件接口

`capability.py`实现独立TX读出、跨RX类中心间隔、低分位/最弱RX对间隔、clean/LEO一致性和身份可辨识联合检查。读出拟合与监测分离；类中心只由合法拟合侧L_s建立，不从U_s真值建中心，不能用待测样本自身充当参考中心。

`curriculum.py`实现状态机：连续窗口确认、分离进入/退出阈值、最大状态增量、冷却、能力不足的hold与诊断；不强制完成课程。课程变化后清空过时Optimistic方向及特征缓存，更新审计版本。

固定课程模式保留CORE90全部启用时刻。能力课程先只控制现有LEO严重度/场景采样概率，保留clean+satellite CE-only语义；随后独立消融控制现有辅助项权重/启用，避免同轮改动多个轴。E80卫星CE启用是否延后必须作为明确课程参数及消融差异，不可静默变化。

附件中的“响应分支独立能力→融合”是条件接口。CORE90的域保留分支不是等待进入TX融合的身份响应分支，不应为了凑模块强行将z_dom接入TX。实现可注册的分支能力与融合门控接口，并用合成双分支验收；CORE90配置标记`NOT_APPLICABLE_NO_RESPONSE_FUSION`。未来确有响应分支的独立方法再接入，并单独验证部署图变化。

## 8.进阶模块也纳入交付

`jacobian_audit.py`在显式有符号场上，用JVP/VJP计算局部`||(J-J^T)v||/(||Jv||+||J^Tv||+eps)`，限定末端身份块和域头、记录参数尺度/方向seed，不构建全网矩阵。用解析双线性和有限差分核对；含FISHR等高阶项时明确阶数/算子支持，不能静默丢失项后声称全场诊断。

`response_tracking.py`实现域头空间的阻尼Hessian-vector求解：

```text
(H_phi_phi + damping*I) delta_phi = -H_phi_theta delta_theta
```

从带正则线性头开始，再支持受控非线性头。使用迭代线性求解，不显式求逆；记录阻尼、迭代数、残差、负曲率/未收敛和预算。CG只用于满足正定条件的路径；否则增加预登记阻尼或回退普通有限头更新，不能宣称已完成有效响应补偿。`delta_theta`来自实际编码器/共享参数更新，且补偿后在合法monitor验证响应方向。

它是可选进阶实验，与同成本额外头训练比较；不展开多层内循环来改变编码器目标，不把隐式公式本身称为创新。

## 9.逐项追踪表

所有代码目标均为拟新增或拟接入路径，非已实现声明。简称G=`code/cvsrffi/game_tracking/`；T=`code/tests/test_game_tracking_*.py`。

|ID|报告章节|要求|目标文件|状态|验收|
|---|---|---|---|---|---|
|R01|1、10.1|CORE90配置和参数梯度归属|G/parameter_roles.py、configs/core90_game_legacy.json|pending|配置差异、共享参数去重、逐loss梯度|
|R02|8.2|当前协议、scratch、source-only及谱系|SSDG/train_ssdg.py、新launcher|pending|禁止旧权重、V/U/target权限负测|
|R03|10.1—10.2|GRL/外部系数与冻结导数|G/field.py、Tfield|pending|符号、尺度、detach/冻结反例|
|R04|10.3—10.4|完整虚拟状态与固定批次事务|G/state.py、step_context.py|pending|状态回滚、随机固定、一次提交|
|R05|2、3.4|普通、不同lr、固定k、交替|G/solvers.py|pending|历史兼容、解析交替、计数|
|R06|3.1、7.2|完整EG|G/solvers.py|pending|解析收缩、原点校正、全组覆盖|
|R07|3.2|Heun/RK2|G/solvers.py|pending|两梯度平均及半径倍率|
|R08|3.3|Optimistic与事件重置|G/solvers.py|pending|历史保存、课程/伪标签变化重置|
|R09|7.2|分块预测近似明确命名|G/solvers.py|pending|只动头、独立配置和标签|
|R10|4.1—4.3、8.2|采集组隔离及恢复探针|G/source_audit.py|pending|fit/monitor组不交、无主模型写回|
|R11|4.3|G_lag、S_rx与先验|G/source_audit.py|pending|解析CE例子、单RX/缺样本无效|
|R12|4.3|独立初始化线性/MLP低频审计|G/source_audit.py|pending|统一预算、容量及方差记录|
|R13|4.4|梯度冲突及恢复前后方向|G/gradient_audit.py|pending|固定参数/批次、零范数处理|
|R14|4.5、10.5|显式场Jacobian局部审计|G/jacobian_audit.py|pending|JVP/VJP对有限差分与解析场|
|R15|5.1—5.2、7.2|有限catch-up与新鲜缓存|G/controller.py、state.py|pending|编码器冻结、k封顶、版本失效|
|R16|5.3—5.4|阻尼隐式响应追踪|G/response_tracking.py|pending|线性解、残差/负曲率处理、成本匹配|
|R17|6.4|独立身份能力、跨RX间隔、一致性|G/capability.py|pending|类中心排除泄漏、塌缩反例|
|R18|6.4、7.1|条件分支能力/融合接口|G/capability.py、curriculum.py|pending|合成分支测试；CORE90无响应融合标N/A|
|R19|6.1—6.5|有滞回/限幅/冷却的课程|G/curriculum.py|pending|抖动序列、长期未达标、事件失效|
|R20|7.1—7.3|状态触发动作与无效信号处理|G/controller.py|pending|各状态可达、实际执行计数|
|R21|7.4、9.3|预算控制与全成本|G/budget.py|pending|开销计入、耗尽回退、计数一致|
|R22|8.1|覆盖/先验/条件RX诊断|G/coverage_audit.py|pending|仅合法TX、缺项N/A、无目标替换|
|R23|9.1—9.2|完整对照、随机/固定/跨seed重放|configs及scripts/build_core90_game_matrix.py|pending|匹配动作预算、计划可重放|
|R24|9.3—9.4|任务/机制/资源及分组不确定性|scripts/analyze_core90_game.py|pending|完整日志、三预算口径、采集组统计|
|R25|10及7.4|断点恢复与部署导出|G/state.py、post_stage_common.py|pending|连续/恢复一致、部署输出一致|
|R26|全报告|入口集成与真实路径验收|SSDG/train_ssdg.py、T、launcher|pending|CLI→场→动作→日志→checkpoint闭合|

计数：pending=26，implemented=0，verified=0，deferred=0，rejected=0，blocked=0。进阶模块没有从计划删除；条件分支接口须实现，但CORE90不强加新的融合架构。未启动实验不等于代码实现被阻塞。

## 10.实施顺序与实验矩阵

1. **基线恢复与事务基础**：R01—R04、R25。提取显式历史配置和损失场，完成关闭新模块的等价验证。历史结构缺证据部分在此落实；不能冒用当前演化版本。
2. **只诊断**：R10—R13、R17、R22。固定网络、loss、课程。先验证“在线低准确率但恢复后重新可读出RX”及其梯度方向变化。
3. **固定求解器对照**：R05—R09。先解析模型，再CORE90真实前向/反向；SGD严格公式与AdamW变体分开命名。
4. **有限追赶和选择性校正**：R15、R20—R21。先只自适应头step，再状态EG；要求优于相同动作预算的固定/随机调度。
5. **能力课程**：R18—R19。先单轴LEO课程，后现有辅助项；不把域保留分支变成身份融合分支。
6. **进阶及完整交付**：R14、R16、R23—R26。补齐局部Jacobian、响应追踪、全部矩阵、resume及分析产物。

|行|设置|回答的问题|
|---|---|---|
|B0|CORE90匹配scratch固定课程|当前协议下方法基线|
|B1|B0+诊断，控制关闭|诊断开销及非侵入性|
|B2|无身份域对抗，其他损失不变|是否需要这一对抗项|
|B3|普通GRL的source调优lr/对抗尺度|是否只是原基线未调好|
|B4|固定k及交替，source调优更新比例|是否只是头多训练|
|B5|完整EG|固定方向校正价值|
|B6|Heun/RK2|相关求解器对照|
|B7|Optimistic|低开销替代|
|B8|head-lookahead|完整与分块近似差异|
|S1|自适应catch-up，课程固定|追踪诊断的作用|
|S2|固定/随机比例EG，课程固定|校正计算量对照|
|S3|状态EG，课程固定|同预算下触发时机作用|
|S4|追赶+状态EG，课程固定|两类动作组合|
|C1|能力课程+普通更新|课程单独贡献|
|C2|S4+能力课程|完整主框架|
|C3|固定/随机动作、同预算|状态反馈是否必要|
|C4|其他source seed生成日程→新seed重放|实时反馈与固定好日程的差异|
|H1|局部响应补偿及同成本多头step|进阶机制的额外价值|

此表是最终支持的实验菜单，不是立即全量派发命令。先小规模配对source探索，再按冻结假设做多seed确认。调优预算对各基线同等披露；不能把target评估后的胜出行拿来重新训练。跨seed日程只来自source开发过程；固定/随机动作匹配总量且声明时间分布差异，不能偷用本seed未来诊断。

三种口径分别报告：同主更新/样本预算（额外前向读取另计）、同实测计算/时间预算、充分训练E200结果。若E200未充分收敛，明确限制；更长预算仅在后续训练授权内预登记，不能凭target结果延长。达到预冻结source能力标准的时间作为第四项效率指标。

## 11.验证、产物与完成标准

实施验证分为解析求解器、状态事务、权限负测、真实CORE90集成四层。项目代码测试使用串行`conda run -n ssr-gpu ...`并核对解释器。关注AMP溢出、共享参数、BN/MixStyle/Dropout、EMA/伪标签、原型提交、控制事件及resume；文档阶段不执行GPU测试。

实际训练链：`CLI/config → resolved config → parameter roles → StepContext → audit/controller → solver commit → counters/state → checkpoint → analysis`。配置存在或loss非零不足以验收，日志须证明catch-up/correction/课程事件实际发生。自然训练未触发的模块标为已实现但该run未激活，集成测试以可控状态覆盖路径。

交付文件：`resolved_config.json`、`parameter_roles.json`、`logs.jsonl`、`game_audit.jsonl`、`game_actions.jsonl`、`curriculum_events.jsonl`、`resource_summary.json`、final checkpoint及controller/optimizer恢复状态、独立预测和评分、同row分析报告。沿用现有run目录，不增加签名或receipt链。

任务指标包含clean、三个`leo_*_weak`各自accuracy/macro、LEO均值与最低场景、最弱RX和关键TX对；机制指标包含G_lag/S_rx、探针容量、梯度关系、状态动作前后变化；资源包括总时间、达到能力时间、主干前后向次数、额外域头step、审计成本、峰值显存、失败/跳步。统计以receiver、采集会话/物理记录为合理独立单位，报告配对seed差异及不确定性，不将重叠IQ窗口当独立重复。

source开发对照与锁定target确认分开；冻结候选后再固定prediction，由独立scorer连接truth，target结果不回流选择、调度或重跑。单seed、曲线更平滑或训练完成均不构成科学晋级。

最高实施风险是完整预测—校正的共享参数和持久状态一致性；其次是探针数据独立性、实际domain标签含义，以及把源域目标冲突误诊为对手滞后。最高研究风险是匹配预算后收益消失。三种可证伪结论均应保留：未发现滞后、调好普通更新已足够、状态策略未优于固定/随机重放。

## 12.本次计划验证记录与参考

已执行：读取附件全部章节；核对项目协议及最小流程；读取历史launcher、训练loss/optimizer/选模入口、当前双分支关键forward、历史技术报告；执行Git状态与HEAD查询。发现并保留既有修改。本次未执行训练、性能测试或远端操作。

网页核对确认域对抗博弈与Runge–Kutta已有先例：[Acuna等，Domain Adversarial Training: A Game Perspective](https://arxiv.org/abs/2202.05352)。本计划的Heun是附件指定算法对照，不声称精确复现该论文所有求解器/实验。

官方接口参考：[PyTorch functional_call](https://docs.pytorch.org/docs/2.14/generated/torch.func.functional_call.html)。实施必须以ssr-gpu实际PyTorch版本验证接口、共享参数和buffer行为；网页版本存在不代表本地已安装对应版本。

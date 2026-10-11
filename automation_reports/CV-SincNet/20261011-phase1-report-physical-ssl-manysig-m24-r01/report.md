# 无显式CFO的物理半监督与多解耦优化

当前为设计/实现阶段，尚未发布。目标沿用clean三seed均值≥84%、同seed原纯CE＋旧门控＋cos E200、二阶段40epoch/8880。报告末尾修订优先，用户最新强权重优化要求覆盖报告首版较小LR建议。

首批预定8变体×3seed=24行：C0匹配改参考特征后的L-only CE；C1软KL SSL；C2 C1＋DL已知线性动作；C3 C1＋DR条件教师；C4 C1＋DQ重复残差/可靠性；C6 C1＋非线性记忆身份支路；C7 C1＋unit-aware Sinc频带优化；C8 C2＋DQ＋unit-aware的固定组合。C8仅检验预定组合，不冒充报告要求的源侧选中C5。C5等待C1/C2/C3/C4三seed源结果，再按预登记源规则纳入有效模块，所有输入禁止target结果。图/低秩交互/Fishr按报告的后续顺序保留。

学生E40统一测试，不选EMA。BL=BU=128；前888步只用L身份CE（物理变体在此阶段只用L；U从step889起参与），完成step888后克隆适应后的EMA=.99，step889开始U，lambdaU随后888步升至1，soft KL分母固定BU，tau=.95，identity weight与physical quality weight分开。无U entropy压缩或真实TX检查；不继续旧parent硬标签、强一致及稳定mask配方。新U采样按15源RX×日期与opaqueUID平衡覆盖。

主普通权重encoder/head峰值1e-4/3e-4，保持所有身份权重可训练，warmup与cos固定预算；unit-aware仅对应C7/C8，freq峰值3e-6（25MHz下coordinate步长75Hz上限，native舍入另报），频率不做weight decay。当前F21源码用于基础设施参考，不复用其旧状态/参考缓存/动作网络。

新参考保留29维兼容投影，但原CFO位置固定0，去旋转改为各重复周期受限unit-modulus对齐，不能声称完全CFO不变；原IQ通路保留。辅助状态不含相邻相关real/imag；使用相关幅值/包络/重复质量。新投影语义、作用监督与条件统计全部在当前合法源数据中重新适配。

高风险未完成点：新参考输入改变后，教师、R统计、动作目标必须使用同一版本表示；所有预测入口必须安装同样的前端定义。对应代码、实际参数及checkpoint来源要在resolved与评分provenance中一致记录。完整事项与边界见traceability.md。

已接入真实训练与预测入口，当前本地验证通过，未发布。动作头读取原始中间349维与变化349维；DQ预测留出周期相对模板残差24维与12维对角误差尺度。任务头从零训练，不沿用旧作用网络。DL/DQ每4次身份更新执行一次，各最多L32+U32；单位频带优化仅C7/C8，所有学生普通权重强LR。

DR使用共享TX原型＋加性RX/日期偏移及50%共享方差收缩；保存L/U均值、软有效样本量、固定L先验及alpha。每次刷新固定一个教师坐标快照，避免动态编码器表示与旧统计混用。条件版本在V按50%、90%、95%、100%覆盖对比普通EMA；这些V结果可供预登记C5源选择，禁止目标反馈。

本轮C5规则已在任何新目标分数读取前固定：三seed源分数0.5×V总体＋0.5×最差源RX，模块均值相对C1正增益且任一seed下降不超过0.002；C2/C4还要求三seed独立V物理MSE均低于零基准，C3要求三seedV的90%覆盖错误率均低于EMA。纳入所有通过模块，否则C1。C5只冻结选择说明，后续使用独立run执行；C8仍是固定探索组合。

源码中data.py明确按TX/RX/day/eq分组、连续sig索引切分，当前合法源角色复用；不称随机划分，也不从sig索引推断绝对时间漂移。辅助网络只能读取作用实际参数和物理观测，U真实TX始终不提供。

完整24row配置、六类seed记录和逐行输出已写入experiment.json。训练日志含各损失/权重、每888步分量梯度、普通权重移动、native频带实际变化、U访问/接受/翻转与RX×日期覆盖。测试保存argmax和log概率，固定后独立评分accuracy/F1/NLL及parent fix/break。历史父预测只有argmax，parent NLL记N/A。

发布状态 **VERIFIED**：实际执行提交 `dfaf47493ac65cf98b96b81dc622d229b103139f`，N607 dispatcher PID 4066843，24/24新row实际进程及resolved config核对通过，8卡各3进程；三份真实E200父权重与24个native图预检PASS。预检在丢弃的内存副本上执行24次更新，正式预算仍每row8880。训练与测试尚未全部完成，clean均值84目标未验证。

最近实际进展：3个C0已完成40epoch/8880并完成自动clean评分；其余21个SSL变体处于第15至22epoch，teacher/U已正常启动，未发现所属row技术错误。当前不读取C24测试数值改变任何训练。

Git旧结果归档的原始resources.json为181.85MB，首次push被GitHub100MB限制拒绝；失败本地commit保留，原始文件不移动/覆盖/删除。另建干净分支`codex/report-physical-ssl-clean-delivery-20261011`，以现有共享OID为父提交，完整评分与数值成本保留，重复role IDs使用源契约路径引用，独立远端OID读回VERIFIED。后续正式交付使用新干净worktree，原实验执行commit不变。

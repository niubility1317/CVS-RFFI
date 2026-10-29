# G10 源端域残差投影证伪

本次不追加目标 Q 调阈值。上一轮全星地单 F0 的 RMD 未能区分未知，且没有形成有效匿名原型。本候选检验：源 L 类内接收机／日期的聚合残差是否掩盖身份几何。可逆线性缩放后重新拟合全协方差可能抵消变换，因此采用真正的低秩方向去除，并以类均值子空间保护和匹配控制检验。

唯一主候选为 P1/rxday/RMD。固定比较 control、RX-only、RX×day，各自 MD/RMD，两条 P0/P1 路径；control 需与原同输入 RMD 数值一致。五个单位分支的 class-conditioned 域均值残差先正交于六类 between-class 均值张成空间，再取累计90%残差能量且最多16方向，投影去除后单位化。零残差使用 identity。均值、协方差、分数标准化均只拟合 L；原 F0 默认六类身份头不更新、不替换。此保护仅保留源类均值空间，不保证未知身份未被损伤，后者正是待证问题。

固定原 F0 E50/11100（232993参数）与 DURA v1/R69（266951参数），来源沿用已核实原 Main 六类 source contract；无 E51、OE 增量、目标适配祖先。原五 RX、三日、L6300/V27000 物理身份和角色不变；从原 raw ManySig 按原 metadata 选包，经同 G10 MIX01 once-RMS 生成模型输入。已在上一轮生成的 known G10 是同 seed/physical ID 的确定性重建，不是新增独立观测。

另建 source heldout novel3150 的 G10 输入：复用 `20261029-phase3-source-oe-data-r01` 的实际 parent complete/report/novel mapping，仅从原 Full15 ZIP member 精确取原索引，旧 normalized IQ 和旧 truth 不作为输入，不重采样、不重验 Full。三 TX 为19-15/19-16/7-1，各1050，均在原source RX/day，但未进原六类F0训练；它们有历史源开发及开发目标暴露，不是新 TX 盲测。正式四未知与 RX24-18 不访问。新 received 与 opaque IDs/公共 RX/day metadata 隔离于 private TX/scenario truth；predictor 不读取 truth。

source 仅消费 L6300＋V27000＋novel3150＝36450 个已有物理记录，两路径72900条 F 前向、36450条 G 前向，合成smoke另计；神经训练0更新。source L全部状态冻结后，V仅做42组已知风险的全局τ，再读取 novel public预测。全部path/variant/score预测完成后，独立source scorer才连接 novel truth。源已知 V是校准集，经验保护不等于独立泛化证据。

主候选晋级必须通过全部42组已知保护，以及源 novel总体、5RX、6scene的逐TX宏拒识≥90%；RX×scene未知只诊断。缺类不能认证，通过也只代表source eligibility，不代表跨RX Phase3成功。control/rx/MD是固定消融，不能按评分替换主候选。主候选失败即NO_PROMOTION，不访问target、不调参或扩seed；通过后才另行冻结小规模目标四态检验。联合指标按 `docs/PAMF_PHASE3_METRICS_20261029.md`，新类注册指标在本源端阶段为N/A。

执行使用唯一 owner、不可覆盖root。本地CPU数据run `20261029-phase3-star-source-novel-g10-r01` 完成后，N607模型run `20261029-phase3-star-nuisance-s2026092701-r01` 单GPU worker。最多新增2/card且启动查实际余量，保持其他owner进程。配置、代码commit、精确命令/CWD、输出、资源与技术停止规则在各run最小登记补齐，不另设设计审批。数据builder无模型/目标读；模型run第一步真实checkpoint合成星地无query smoke，PASS即继续；技术错误保留partial，科学负结果保持完整评分。

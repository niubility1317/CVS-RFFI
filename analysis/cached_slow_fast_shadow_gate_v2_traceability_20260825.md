# Cached Slow-Fast影子状态与门控V2需求追踪

来源：用户指导报告《对提交8c6ddd3a的优化建议》；实施设计：`docs/CVS_SLOW_FAST_SHADOW_GATE_V2_DESIGN_20260825.md`。

|ID|来源章节|验收要求|目标文件|状态|验证／说明|
|---|---|---|---|---|---|
|V2-01|§3、§17A1|truth-blind输出全部固定lambda影子状态|runner/scorer/config|pending|状态输出、query隔离和truth-last测试|
|V2-02|§4.1|用MacroCE、class CVaR和移动风险替代离散三重一票否决|selection|pending|手算风险与回退边界测试|
|V2-03|§4.2|K10分层5／5双折重复cross-fit；K5保持类平衡；K1回退|selection|pending|fold逐类计数和物理行互斥测试|
|V2-04|§4.3|从合格候选中选择最小风险lambda，平局选较小强度|selection|pending|非单调风险fixture|
|V2-05|§4.4、§18|保存每lambda完整CE、margin、move、flip和拒绝原因|selection/receipt|pending|audit schema测试|
|V2-06|§5.1|统一Phase1.5、Phase2 update与gate的logit scale|objectives/phase15/selection|pending|非8.0scale传播测试|
|V2-07|§5.1|prediction明确输出raw cosine|runner/receipt|pending|score_type和数值测试|
|V2-08|§5.2|FAST_LOWRANK使用零中心有符号门控|adapter/bundle|pending|零输入关闭、正负方向测试|
|V2-09|§5.3|COMMON_SHIFT只用rho表达lambda强度|adapter/selection|pending|0／0.5／1强度手算测试|
|V2-10|§5.4|拆分尝试与提交更新量、fold拟合次数|selection/receipt|pending|回退仍报告计算量测试|
|V2-11|§5.5|trust_radius强制显式传入并记录|selection/runner|pending|遗漏参数失败测试|
|V2-12|§10、§17A2-A3|比较J={1,3,5,10}与步长倍率={0.5,1,2,4}|runner/config/scorer|pending|truth-blind影子网格测试|
|V2-13|§13|核验checkpoint、160维特征、class mapping与原型预测一致性|smoke/report|pending|真实checkpoint无query数值核验|
|V2-14|§13|保存逐文件／代码／feature SHA并作为发布条件|无|rejected|`REJECTED_EXTRA_GATE`：Git提交和一次release SHA已固定实现；改用直接数值核验|
|V2-15|§6、§7|receiver rank4与LEO rank4慢基、paired reduced-rank operator|phase15/bundle|deferred|仅当P0证明非零状态无query上界后进入P1|
|V2-16|§8|球面切空间与robust公共残差|phase15/selection|deferred|P1条件项|
|V2-17|§9|类别梯度共识／可靠性加权|phase15/selection|deferred|P1条件项|
|V2-18|§11|receiver-held-out元训练与support-only gate学习|phase15|deferred|P0显示support/query泛化失效后进入P1|
|V2-19|§12|clean identity loss及新的pair/floor/trust权重|phase15|deferred|P1条件项|
|V2-20|§14|区分经验几何中心与决策原型|cache/bundle|deferred|P1条件项|
|V2-21|§15、§17C|前移time／freq／fusion Adapter|新中间层实现|deferred|P1仍无有效上界后才进入P2|
|V2-22|§17、§19|按P0→P1→P2条件顺序发布实验并形成详细报告|report|pending|N607 prediction、独立scorer和结论|

当前统计：pending=14，verified=0，deferred=7，rejected=1，blocked=0。最高风险是影子状态数量增加后，runner必须只前向一次query并保持所有状态truth-blind、不可根据评分反馈选择或重跑。

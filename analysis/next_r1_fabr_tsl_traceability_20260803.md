# NEXT-R1 FABR-TSL需求追踪

阶段：`DESIGN_FROZEN / IMPLEMENTATION_PENDING / NO_NEW_PERFORMANCE_RESULT`

|ID|来源要求|冻结实现目标|状态|验证/证据|
|---|---|---|---|---|
|NR1-01|只开发1条原理不同的联合候选|FABR参数空间局部残差＋TSL类对称头|design-frozen|目标§0.1；D130两条特征变换已关闭|
|NR1-02|同checkpoint/received-IQ、42fold、K1/K5|84个candidate-row，每row六逻辑臂|design-frozen|`R0Q/R0F/R0L/R1Q/R1F/R1L`|
|NR1-03|DA不全参更新、不写checkpoint|单block rank2 functional override`phi'=phi0+Ba`|design-frozen|实现与真实checkpoint smoke待完成|
|NR1-04|选层不得预设浅层或读取target|Phase1-only S/F/C/T确定规则，fold排除held receiver/class|design-frozen|资产构建测试待完成|
|NR1-05|Fisher锚必须与量化基一致|封存反量化`B`上的完整`K=B^TFB`|design-frozen|PSD、条件数和量化误差负测待完成|
|NR1-06|K1/K5可辨识|K1只类间项；K5 compactness用physical-LOO|design-frozen|self-leak和K1类内方差负测待完成|
|NR1-07|query零fit/update/selection|`a`仅由当前全部registered support等权闭式求解|design-frozen|query变异不改变state测试待完成|
|NR1-08|精简D92且不分old/new|TSL使用全类pool EB对角方差与单公共`eta`|design-frozen|class permutation/role禁入测试待完成|
|NR1-09|K1不得伪造头部收益|同表示F/L逐logitalias Q|design-frozen|逐logit/hash receipt待完成|
|NR1-10|尾部保护不得成为support调参gate|Phase1封存`rho_h`；单连续Frobenius投影，无网格/正确率选择|design-frozen|距离上界与no-fallback测试待完成|
|NR1-11|单一轻量部署头|`INT8 W[C,160]+FP16 scale/intercept`|design-frozen|量化、字节、MAC receipt待完成|
|NR1-12|分别证明DA/Lite/联合|`R1Q-R0Q`、`R0L-R0F`、`R1L-R1F`|design-frozen|完整score后同row判定|
|NR1-13|负结果立即关闭|任一K5主比较不增H/总正确或伤retained/held/floor即关闭|design-frozen|不得调block/rank/rho/先验|
|NR1-14|正结果最小晋级|G0真实588→一次fresh63→单seed Target25|design-frozen|不运行125、不重复历史矩阵|
|NR1-15|发布前最小gate|聚焦协议负测、真实checkpoint无query smoke、独立P0/P1、Git、preflight|pending|不重验VALIDATED_ONCE数据、不建通用平台|
|NR1-16|模型分工|Sol集成/分析；Terra科学实现/复核；冻结机械runner默认Luna|verified|目标§8与实时AGENTS一致|

## 独立可行性复核（20行内）

1.结论：初审与落盘核对发现的P1均已按唯一公式修正；冻结文本为`P0=0、P1=0`。
2.Phase2只用当前support；query零fit/update/selection，无role/quota/clean/source运行时访问。
3.FABR为单block rank2功能式覆盖，不写checkpoint。
4.量化后完整`K=B^TFB`进入闭式求解与信赖域。
5.K1只用全类间项；K5紧致项必须physical-LOO。
6.K1的F/L逐logitalias同表示Q，头部不声明K1增益。
7.K5先验`v0/nu0/rho_h`在target前共同封存。
8.TSL不分old/new，不读取F臂，不按support正确率选择超参。
9.落盘核对曾发现`eps_F`、反量化选层和`mu/e/v0`解码仍不唯一。
10.目标§0.1现已冻结`eps_F`、实际反量化`B`、physical-LOO残差、log-variance解码、spherical reference、EB diagonal head和单Frobenius投影。
11.`D=0`、非有限或量化后仅噪声变化均拒绝，不fallback。
12.信赖域只限制logit扰动，不宣称保证真实floor。
13.84行六臂完整封存后一次score；任一主比较失败即关闭。
14.通过后保持method lock依次进入G0、fresh63、Target25。

最高风险：Phase1资产构建、真实functional override和量化后完整Fisher几何尚未实现；在真实checkpoint无query smoke完成前只有冻结设计，没有新性能证据。

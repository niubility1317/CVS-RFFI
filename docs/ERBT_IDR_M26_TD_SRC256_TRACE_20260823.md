# ERBT-IDR M2.6 TD-SRC256设计实现追踪

日期：2026-08-23

设计计划：`docs/superpowers/plans/2026-08-23-erbt-idr-m26-td-src256.md`

|ID|来源|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|M26-01|目标域偏移|只用六个目标旧类support与Phase1旧类聚合锚点估计域状态|`stage2_m26_td_src256.py`|verified|合成共享偏移、旧类隔离和集成测试通过|新类和query不得参与|
|M26-02|Phase1知识边界|只持久化checkpoint绑定的int8旧类256维聚合中心|`stage2_m26_spectral_anchor.py`、builder|verified|混合导出只筛选source；target值变化时量化锚点逐元素不变；manifest checkpoint和D19类映射逐项绑定；不可覆盖发布和严格加载测试通过|component ID绑定量化状态，不得保存样本或raw IQ|
|M26-03|去RF32|正式候选只消费identity160＋FFT96真实IF256|M26模块、row executor|verified|receipt断言`rf32_consumed=false`，相邻M2.4/M2.5回归通过|禁止零填充或RF代理|
|M26-04|FFT改进|CEP96正交分解及MGD96去趋势/斜率/镜像不对称|`stage2_m26_spectral_anchor.py`|verified|维度、有限、单位范数和局部/镜像扰动测试通过|两者均为同一FFT96确定性视图|
|M26-05|中心职责|support中心只估计类内状态，decision中心单独构造|`stage2_m26_td_src256.py`|verified|状态结构及T1–T5拟合测试通过|避免M2.3中心混用|
|M26-06|非抵消传输|域偏移进入辅助类别分数，不同时平移support/query|`stage2_m26_td_src256.py`|verified|identity、CEP和MGD共享偏移均可检测|不会数学抵消|
|M26-07|安全接口|D92主分数不变，仅低margin加入不超过alpha的残差|M26模块、row executor|verified|高margin逐位回退及逐logit上界测试通过|alpha网格固定|
|M26-08|支持选择|强度只由support留一old/new双边无害证据选择|`stage2_m26_td_src256.py`|verified|K1五臂严格B0，K≥2执行support-LOO|query不参与选择|
|M26-09|消融归因|冻结B0、T1–T5六臂同row矩阵|runner、scorer、summarizer|verified|screen=24行、full125=750行常量测试通过|T1身份、T2/T3 CEP、T4/T5 MGD|
|M26-10|协议闭合|复用VALIDATED_ONCE，prediction truth-unopened，独立scorer truth-last|row executor、runner、scorer|verified|真实checkpoint无query smoke通过；24/24 prediction闭合后truth-last评分，89份score且状态PASS|query输入和拟合计数均为0|
|M26-11|完整分析|输出总体及全部预登记切片、偏移诊断和资源|summarizer、正式报告|verified|总体、K/new、receiver、seed、scene、四状态、old/new、class、margin、中心角距、help/harm、F_within/F_std、域偏移、LOO、门控和资源均已生成|正式裁决`SCREEN_NEGATIVE_NO_FULL125`|
|M26-12|版本发布|本地验证、一次P0/P1审查、commit、push、远端OID回读|Git及正式报告|in_progress|实现与预登记提交均已推送并核对；最终结果提交待完成|只stage本轮文件|
|M26-13|screen证据边界|4 identity screen不得标为full125|M2.6 summarizer、汇总和报告|verified|新增screen/full125证据边界回归测试；正式汇总`matrix.matrix_kind=screen`|仅修正标签和裁决，所有实验数值数组保持不变|

## 当前反向审计

- verified：12
- implemented：0
- deferred：0
- rejected：0
- blocked：0
- in_progress：1

本地M2.6聚焦测试18项及M2.4/M2.5/M2.6相邻回归25项通过。首次独立审查为P0=0、P1=2；两个P1均已按原问题定点修复，允许的一次定点复审确认原P1-1/P1-2关闭，终态P0=0、P1=0、READY。

正式run为`erbt_idr_m26_td_src256_repr_screen_20260823_v1`。Phase1锚点只消费2400条source行，target/query均为0，component ID为`236823c57210de9a58a47ee1868d27a907368864f8ad9f710c2eda0c51258da5`。真实checkpoint无query smoke、远端编译和不可覆盖路径检查均通过；24行prediction完成后才连接truth，最终状态`ANALYZED`。

screen结果：T3相对B0的`ΔH=−0.000929`、help/harm=17/13；T5为`−0.000729`、15/12。T4 MGD96获得唯一正H差`+0.000242`，但远低于`+0.002`门槛。故不启动完整125，M2.6不晋级。该结果验证“目标域状态确实进入决策”，同时否定“support类内LOO可靠度足以决定独立目标域残差”的假设。下一候选应把目标域和复频域特征改为M2.5 B3的共识/否决器，而非替换或平移D92稳健中心。

# CORE90锚定几何重构要求追踪

日期：2026-09-11。来源：[用户设计报告全文](core90_anchored_geometry_design_source_20260911.md)；实施说明：[设计实现计划](core90_anchored_geometry_implementation_plan_20260911.md)。代码基线`7fd06087`。

本表状态针对**未来实现**，不是把计划写完记成代码已验证。42项`pending`、6项`deferred`；0项`implemented`、0项`verified`、0项`rejected`、0项`blocked`。计划本身已完成；以下项目尚未进行生产代码修改。

`deferred`均有报告规定的分阶段原因；不因缺少后续实验数据阻止当前source主线实现。source章节编号来自原报告，所列文件均相对当前隔离Git承载面。

|ID|来源章节|具体要求|目标文件/任务|状态|拟采用验证|备注|
|---|---|---|---|---|---|---|
|R01|2.1、8|冻结原H0完整推理含义|`anchored_cache.py`，T1|pending|真实H0完整/fast/identity三方前向对照|不只保留weight|
|R02|2.1|baseline不含标签margin|`anchored_cache.py`，T1|pending|修改监督label不改变s0；labels=None与CosFace原推理对照|margin logits只能用于监督支路|
|R03|3.1|L_s、U_s、V角色不串用|`anchored_cache.py`，T1/T7|pending|训练拒绝V/U_s/target角色，标签只在合法层存在|当前G不使用U_s|
|R04|3.2|一次提取绑定来源的缓存|`anchored_cache.py`，T1|pending|checkpoint/contract/preprocess/view不匹配拒绝命中|哈希仅缓存身份，不新增验证链|
|R05|3.2|eval/no_grad与普通训练tensor|`anchored_cache.py`，T1|pending|cache重载参与weight反传；BN/参数不更新|不把eval当作关闭autograd|
|R06|3.2–3.3|physical ID与view唯一性和权重|`anchored_cache.py`、`anchored_fit.py`，T1/T3|pending|同样本多view总权重不变；所有view同fold|view不是shot|
|R07|2.2|共享低秩正定M，r=4/cond≤2|`anchored_geometry.py`，T2|pending|谱/正交/稠密参考/数值边界|允许正负log特征值修正|
|R08|2.2–2.3|a=0退化H0且可学习|`anchored_geometry.py`，T2|pending|H0等价；非退化数据b梯度非零|不使用全零AAᵀ死初始化|
|R09|2.2|保持角度范数不变性|`anchored_geometry.py`，T2|pending|正缩放、多幅度、eps边界测试|eps区不声称严格尺度不变|
|R10|2.2–2.3|W/tau冻结、无类bias/自由协方差|`anchored_geometry.py`，T2|pending|trainable参数白名单与step后不变|首轮仅B/b共644参数|
|R11|9.1|低秩公式与类别投影预计算|`anchored_geometry.py`，T2/T9|pending|dense参考/梯度；推理分配和计时|不每样本生成M|
|R12|3.3|匹配clean/LEO配对拟合|`anchored_fit.py`，T3|pending|相同物理/增强seed、多view权重、四场景生成记录|clean/LEO各0.5是建议起点|
|R13|3.3|180physical/360feature/35batch|`anchored_fit.py`，T3|pending|全量无重复无遗漏；4RX/3RX折同比例|公平报告step而非仅epoch|
|R14|4.1–4.2|分类、正确样本keep、metric正则|`anchored_fit.py`，T3|pending|错误H0样本无keep梯度、截断/无正确样本边界|不对全部样本做教师KL|
|R15|4.3、10.2|损失贡献看加权梯度|`anchored_fit.py`，T3/T9|pending|分别求梯度范数/夹角不增加step|loss负或大不等于梯度主导|
|R16|3.5|预算与收敛分开|`anchored_fit.py`，T3|pending|固定上限、稳定窗口、converged/budget_exhausted输出|只用source内部诊断|
|R17|3.4|五折source RX头部OOF|`anchored_crossfit.py`，T4|pending|fit/heldout RX与physical集合互斥|骨干见过所有source RX|
|R18|3.1、3.4、10.3|门控训练/评估诚实隔离|`anchored_crossfit.py`，T4|pending|嵌套外5/内4折；预处理/阈值也隔离|设计必要展开，不用OOF门控训练集自评|
|R19|3.4–3.5|最终5RX重拟合与预算冻结|`anchored_crossfit.py`，T4|pending|final专家来源、预算、seed；3RX训练集合去重|head seed不等于backbone seed|
|R20|5.1|先固定概率混合5动作|`anchored_fusion.py`，T5|pending|alpha端点、概率归一化、全动作审计|不是独立似然乘积|
|R21|5.2|监督实际混合的rescue/harm|`anchored_fusion.py`，T5|pending|三类示例混合选第三种决定|不能只判两专家谁正确|
|R22|5.2–5.3|小共享utility门控|`anchored_fusion.py`，T5|pending|成本敏感utility、类别置换、输入禁用字段|lambda_H=2为计划起点|
|R23|5.3|不足收益时回alpha0、专家detach|`anchored_fusion.py`，T5|pending|全非正utility回退、并列小动作优先、无专家梯度|低quality不自动采用G|
|R24|5.4|可计算H0保护边界|`anchored_fusion.py`，T5|pending|每类边界、严格不等式、浮点/并列、实际argmax|保护H0决定不保证其正确|
|R25|5.2、5.4–5.5|utility动作与部署动作一致|`realize_actions()`，T5|pending|保护裁剪后的概率用于训练label和推理|不以原alpha训练再部署降档|
|R26|5.5|固定内部尺度后校准最终输出|`anchored_calibration.py`，T6|pending|冻结后禁止改T0/TG/专家；正TF不改唯一argmax|V不训练门控|
|R27|5.5、10.1|所有候选匹配最终校准|`anchored_calibration.py`，T6|pending|同V、同physical权重、同TF约束|H0/cosine/G/fusion都覆盖|
|R28|10.3|部署选择与置信排序分开|`anchored_calibration.py`、reporting，T6/T9|pending|实际覆盖/ties/有效准确率、两条曲线单独输出|target阈值不回写|
|R29|2.1、8|严格冻结骨干与alpha0回原H0|`anchored_pipeline.py`，T7|pending|前后state相等、alpha0直接锚点、导出重载|共享骨干改动不允许继续称严格回退|
|R30|3.1–3.2、10.3|缓存、训练态与部署态分离|`anchored_pipeline.py`，T7|pending|禁止cache/label/RX/ID/数据路径；stage声明|不自动授予Phase2运行权限|
|R31|10.1、10.3|独立source协调器与候选冻结|新CLI/config，T7|pending|字段全消费、存在输出拒绝、阶段不隐式target|不复用旧自动预测全链|
|R32|10.3、项目协议|预测只读、逐样本全类竞争|`anchored_pipeline.py`，T7|pending|query批次/顺序不变、无参数/统计更新、truth分离|实现能力不等于确认数据已经可用|
|R33|6.1|固定坐标尺度与观测语义|`partial_evidence_fit.py`，T8|pending|mask不改保留坐标；命名块schema|IQ缺失不冒充joint局部mask|
|R34|6.2、4.3|简化E分阶段拟合共享协方差|`partial_evidence_fit.py`，T8|pending|无state/support/class covariance；观测模型单独拟合|不继续堆联合density总loss|
|R35|6.2、9.2|复用正确边缘化与低秩NLL|`partial_gaussian_head.py`，T8|pending|低秩/稠密对照；仅一次solve；全缺失|共享logdet不是类判别新信息|
|R36|6.3|可靠性与新增身份价值分开|`pairwise_evidence.py`＋reporting，T8|pending|Schur/J全部类对；对应source跨RX风险|诊断不替代实验，不伪造Bayes去重|
|R37|6.4|pattern而非只按维数校准|`mask_pattern_calibration.py`，T8|pending|相同维数不同模式、匹配V视图、physical支持计数|旧v1状态不静默重解释|
|R38|6.4|未知模式defer/缺失不等于unknown|`mask_pattern_calibration.py`，T8|pending|未覆盖/少支持/全缺失；正确拒绝计错|首版不放行粗组fallback|
|R39|10.1|A0–A6/P1及匹配普通角度对照|config/CLI/report，T9|pending|矩阵ID和开关一一对应；端点复用不重复训练|C_angle与A1不同|
|R40|9.3、10.2|成本拆分和所有类流量诊断|reporting/profile，T9|pending|各阶段计时/显存；所有类precision/recall/流入错误|不对历史类3设置专属规则|
|R41|10.2|六类指定报告文件和必要补充|reporting，T9|pending|计数/分组/梯度/geometry/cache/协议完整|全部实际预算和缺项如实记录|
|R42|10.3|净收益→分组/seed→等覆盖→成本判定|report，T9|pending|完整source输出与分层verdict|单head seed或0.1pp不自动晋级|
|R43|2.3|有限类别方向修正|计划第8节|deferred|以后检查共同角度上限与M/W分开消融|先证明共享M有效|
|R44|7.1|共享响应→类响应，均值/协方差分开|计划第8节，未来conditional响应扩展|deferred|source跨RX收益、RX/TX关联、均值-only/cov-only|首轮G/E均不加入|
|R45|7.1、9.2|相关状态误差残差与低秩传播|计划第8节|deferred|配对delta残差偏置/方向/协方差；JL_e低秩|不默认z/e误差独立|
|R46|7.2|中心support后验及受限斜率|计划第8节，未来registration扩展|deferred|均值/收缩/后验；物理shot、秩/条件数/状态覆盖|先明确source注册，再合法target support|
|R47|1、8、9.3|联合骨干/H5等复杂分支|计划第1、8节|deferred|新骨干缓存/误差/先验/V全部重核；额外H0成本|报告明确首轮不执行，不是漏项|
|R48|10.3|独立确认数据及完整骨干多seed|计划第1.1、9.2、12节|deferred|更独立目标的实际契约与全部继承来源；truth-last|当前请求只做计划，数据未指定；旧target不洗白|

## 本次计划自查

- 原报告第1–10章及最终推荐都有对应任务/边界，无条目被静默替换为历史H3/H4命名。
- 五项实施补充已显式标为计划决定：eps边界、嵌套门控评价、保护后动作utility、具体默认预算/超参、source-only默认协调器。
- 最高实现风险：margin、温度、保护裁剪或OOF处理导致utility与部署实际动作不一致。
- 最高科学风险：把已观察target启发的开发研究当作新的干净确认。
- 未来代码完成前不得把上述pending记为verified；本次仅文档覆盖、链接、编码与Git交付检查通过。

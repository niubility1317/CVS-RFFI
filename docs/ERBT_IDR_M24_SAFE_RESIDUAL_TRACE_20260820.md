# ERBT-IDR M2.4／F1-SafeResidual实现追踪

日期：2026-08-20

设计输入：`D92 E1／ERBT-IDR M2.3 RFGuard全面复盘与优化方案`。实现基线：`ed890015ddb9968663609727c28fdb4d749d4334`。实施分支：`work/m24-safe-residual`。本轮复用既有`p2_min_v1`、`VALIDATED_ONCE`、`capsule_id`、`split_id`和M2.3 overlay，不重验数据。

状态词：`pending`、`local_verified`、`verified`、`rejected_by_evidence`。`local_verified`表示代码与本地测试已经闭合，但仍需真实N607 artifact确认。

|ID|设计要求|实现落点|验证|状态|
|---|---|---|---|---|
|M24-01|物理256维F1：`normalize([normalize(id);4 normalize(fft)])`，随后复用冻结F1对角度量、逐样本归一化和量化头|`m24_features.py`、`m24_compiler.py`|真实K10缓存660条query零差异；紧凑态7677B|verified|
|M24-02|D1不得包含quality、prior、uncertainty、nuisance、RF-lite和二次单边加权|D1配置锁和fit审计|D1审计、系数维度与禁用项测试；60行正式prediction保持D1锁|verified|
|M24-03|support center、decision center、covariance center分离|`m24_center.py`|三个中心独立变化测试|local_verified|
|M24-04|固定floor改为相对trace jitter且保持PSD|`m24_covariance.py`|缩放等变、最小特征值和退化输入测试|local_verified|
|M24-05|quality权重与uniform混合并满足`ESS>=max(3,K/2)`|`m24_quality.py`|极端quality、K1/K2/K5/K10边界测试|local_verified|
|M24-06|D3中心quality与D4协方差quality分开|D3、D4独立arm|同输入下仅目标路径变化测试|local_verified|
|M24-07|D5只启用IF残差可靠性|`m24_quality.py`|RF质量不影响D5测试|local_verified|
|M24-08|旧类prior默认关闭；仅support-LOO门控；K1强制关闭|`m24_prior_transport.py`|K1关闭、help/harm门控和新类无prior测试|local_verified|
|M24-09|首轮uncertainty完全关闭；D8仅归一化且封顶|`m24_uncertainty.py`|D1零惩罚、D8尺度有限测试|local_verified|
|M24-10|RF-lite只作为后置残差，`alpha<=0.1`|`m24_rf_residual.py`|系数上界、零交叉块测试|local_verified|
|M24-11|RF全局no-harm先行；仅K10允许类门并层级收缩|`m24_rf_residual.py`|K5全局门、K10类门、K1/K2关闭测试|local_verified|
|M24-12|整候选相对精确F1安全选择；任一模块可完整回退|M2.4编译器|人工伤害例完整回退测试|local_verified|
|M24-13|K1/K2/K5/K10保守分段|M2.4配置解析|四个K分支行为测试|local_verified|
|M24-14|量化增加margin归一化误差P50/P95/P99/max和阈值比例|`m24_compiler.py`|手算margin fixture测试|local_verified|
|M24-15|持久态只保留量化仿射头、registry、schema和摘要|`M24InferenceState`|禁止保留workspace／FP32旁路与字节审计测试|local_verified|
|M24-16|瞬时注册态独立计量|`M24RegistrationWorkspace`和resource receipt|三种字节口径测试|local_verified|
|M24-17|D0–D10完整因果臂且同row同seed|M2.4 row executor／suite|历史两条件×11臂×3场景真实N607诊断闭合；复盘后另增D1-REFIT证据臂|verified|
|M24-18|四状态显式命名与query独立全类argmax|row receipt／prediction artifact|22个诊断臂及60个扩展行truth-last评分通过|verified|
|M24-19|canonical manifest、依赖、非覆盖和提交绑定预检|M2.4 preflight|错误manifest、已存在输出和错误commit测试|local_verified|
|M24-20|truth-last评分复用同row配对诊断|M2.4 scorer|诊断22行与扩展60行均在prediction闭合后独立评分|verified|
|M24-21|旧M2.3和D92默认路径不变|全新opt-in入口|M2.3及相邻回归最终47项通过|verified|
|M24-22|D1扩展覆盖5个receiver、3个真实method seed和4个K／新类条件|base-cache-only D1矩阵runner／scorer／汇总器|60/60行、180场景、零parity分歧、truth-last评分PASS|verified|

## 冻结实验顺序

1. 诊断run先执行receiver=`3-19`的`K1/new20`和`K10/new5`，每条row执行D0–D10和三个`leo_*_weak`场景。
2. D1必须先通过物理256维F1等价性；若失败，只修复D1几何并用新run ID重跑，不继续叠加模块。
3. truth-last评分后，最多选择2–3个候选进入扩展筛选。候选必须相对D1没有明确同row伤害，并满足其声明模块的支持侧安全门。
4. 扩展筛选目标为5个receiver、至少3个method seed、至少3个新类draw和`K1/new20`、`K5/new20`、`K10/new20`、`K10/new5`四种条件；只使用已存在且合法的`VALIDATED_ONCE`cache组合。
5. 技术停止仅限协议／query越界、错误row或checkout、输出覆盖、非PSD、无prediction闭合、scorer连接错误或重复确定性执行故障。性能差只触发保留负结果和候选淘汰。

## 证据边界

诊断与扩展筛选均为研发证据。除非后续完成预注册的独立确认矩阵，否则不得表述为fresh confirmation、Phase3开放世界能力或星载部署结论。

## D1纠错追踪

v2的K1 D1通过，但K10 D1出现14/660个预测差异，触发预登记硬停止且未打开truth。问题不是physical256定义本身，而是旧实现把冻结对角度量后的逐样本归一化近似成固定support中位数bias缩放。纠错后推理态显式持久化256维冻结log-diag，先执行逐样本变换，再使用由历史F1量化状态解码并按相同两块语义重编译的紧凑头。真实K10 cache的三个场景合计660条query全部与历史F1一致，状态7677B，满足不超过历史F1 1.25倍的预设资源界限。

## 最终实验判定

- 诊断run：`erbt_idr_m24_safe_residual_diagnostic_20260820_v3`，22个臂结果完成；D1在K1与K10分别达到1560/1560和660/660逐query一致。
- 模块判定：D2–D10未晋级。K1无独立输出差异，K10在三个场景均触发support-harm整候选回退。
- 扩展run：`erbt_idr_m24_d1_expanded_20260820_v1`，5个receiver×3个method seed×4条件=60行、180场景；60行评分PASS且总预测分歧为0。
- 科学判定：历史D1只可作为`M24-D1-COMPILE-PARITY`部署编译基线；尚未证明IF256独立support refit等价，也未证明附加模块带来性能增益。

# ERBT-IDR M2.9 TASR48实现追踪

日期：2026-08-25

设计来源：用户提供的《FFT96替代与TASR48设计报告》。实现基线：`codex/m28-local-conformal-risk-20260823`提交`5d50b480`。本轮只实现报告“近期最值得实现”和第一阶段权重消融，不改变既有B3局部残差、RF可靠性、分类头家族或数据划分。

|ID|来源章节|要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|M29-01|结论、§6.1|主候选使用`identity160+TASR48`，辅助权重`alpha=1`，总维数208|`code/cvsrffi/stage2_m29_tasr.py`|pending|聚焦单测、真实cache row smoke|不得以288维零填充状态冒充208维部署状态|
|M29-02|§2.1|从Phase1 source-only的接收机×类别FFT96中心构造类别中心化扰动协方差，固定`rank=8`|同上、`code/scripts/build_m29_phase1_tasr_bundle.py`|pending|接收机置换、target污染负测、特征值/正交性测试|真实Phase1导出含2400条source、6类、7个接收机|
|M29-03|§2.1|bundle封存`mu_G_sp/U_sp/lambda_sp/scaler`并绑定checkpoint|同上|pending|不可覆盖发布、严格成员加载、checkpoint错配负测|复用M2.6的int8聚合知识边界；不保存样本或raw IQ|
|M29-04|§2.2|仅用类别等权support估计目标公共频谱偏移，只保留Phase1扰动子空间并按`lambda/(lambda+tau)`收缩|`code/cvsrffi/stage2_m29_tasr.py`|pending|类重复不改变结果、query不可达、K1有限值测试|REG0只用旧类support；REG1使用全部已注册类support|
|M29-05|§2.3|9点对称平滑；16维残差均值、16维残差RMS、8维一阶差分RMS、8维二阶差分RMS|同上|pending|手算fixture、形状、有限值、确定性测试|固定48维，不使用PCA或密集投影|
|M29-06|§2.3|Phase1冻结中位数/MAD标定后L2归一化|同上|pending|常数维保护、单位范数测试|MAD使用正下限，query不更新标定量|
|M29-07|§6.2|保留D92 full/block support-only LOO融合，首轮不改其他模块|同上|pending|与原D92审计字段、LOO权重和query_rows_used=0核对|运行时使用真实208维与`block_sizes=(160,48)`|
|M29-08|§8第一阶段|同row比较FFT96 `alpha=4/1/0.5`和identity-only|`code/cvsrffi/stage2_m29_row_executor.py`、runner|pending|四臂prediction闭合与same-row评分|用于分离权重问题，不在query选择alpha|
|M29-09|§8第二阶段|加入TASR48 `alpha=1`作为首个替代表示|同上|pending|与四个权重臂同一support/query/cache比较|本轮不加入SR32/TASR32/DSQ/Gabor|
|M29-10|§9|报告old/new/H/F、floor、receiver/scene、条件数、full/block LOO、时延、注册时间、state bytes|scorer、summarizer、正式report|pending|机器可读汇总字段检查|量化一致率沿用D92编译收据；端到端时延不可用时明确写N/A|
|M29-11|权限边界|`p2_min_v1`、`VALIDATED_ONCE`、匹配capsule/split；query只读、truth-last评分|row executor、runner、scorer|pending|协议负测、真实checkpoint无query smoke|方法变化不触发数据重验|
|M29-12|§4|DSQ-Lite48仅在固定前导契约满足时比较|无|deferred|现有输入契约审查|当前导出是任意received-IQ特征cache，未证明固定LTF/重复符号契约|
|M29-13|§5|复数Gabor/Sinc-PCEN48作为第二代Phase1重训路线|无|deferred|不适用|会改变Phase1训练与前端，不属于本轮最小可归因实验|
|M29-14|§6.3|把频谱扰动能量加入support稳健权重|无|deferred|不适用|设计明确要求第二轮再做，首轮先隔离TASR输入收益|
|M29-15|§3|用`J_j`筛选32/48个坐标|无|deferred|不适用|首版固定池化，避免同时改变表示与特征选择|

## 首轮最小实验

- receiver：`3-19`、`8-8`；method seed：`7282101`。
- 条件：`K1/new20`、`K5/new20`、`K10/new5`。
- 臂：FFT96-alpha4、FFT96-alpha1、FFT96-alpha0.5、identity-only、TASR48-alpha1。
- 共6个配对输入身份、30个方法row、90个`leo_*_weak`场景单元。
- 晋级门槛：TASR48相对最佳冻结FFT权重满足`Delta H>=0.002`、`N_help>N_harm`、`Delta min-old>=-0.005`、`Delta min-new>=-0.005`，且完整部署state bytes下降。未通过则`SCREEN_NEGATIVE_NO_FULL125`，不启动完整125。

## 反向审计

完成实现后逐项核对：所有`implemented`项必须有可达生产入口和验证；任何未到`verified`的条目都不得写成完整落实。额外seal、成员哈希、重复数据验证或第二轮模块均记为`REJECTED_EXTRA_GATE`，不得阻塞本轮最小发布。

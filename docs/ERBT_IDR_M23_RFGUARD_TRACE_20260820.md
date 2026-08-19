# ERBT-IDR M2.3／D92 E1-RFGuard实现追踪

日期：2026-08-20

基线锚点：`045c5433ec3778ab56e325333cefa9d75304b160`。实施分支：`work/cvs-active`。本追踪只约束M2.3代码、聚焦测试和首轮同row实验，不改变`p2_min_v1`、capsule、split或既有D92 E0默认路径。

状态词：`pending`、`verified`、`deferred_by_evidence`、`rejected_by_protocol`。

|ID|指导要求|实现落点|验证|状态|
|---|---|---|---|---|
|M23-01|identity160、FFT96、RF独立归一化和独立权重|RFGuard特征模块；IF主头固定`β_id=1,β_fft=4`，RF分类权重不再与FFT等能量|块范数和能量单测|verified|
|M23-02|RF-lite使用物理稳健的8–12维统计|10维非圆度、四阶累积量、幅度分位比和相关幅值|尺度不变、CFO旋转稳健和有限值单测|verified|
|M23-03|RF-quality只作为support可靠性|质量分支输出`q∈[0.1,1]`，并与IF残差可靠性组成support权重，不进入query状态更新|异常IQ降权、fit签名与row receipt单测|verified|
|M23-04|正式兑现降维|M2.3 IF及K≤2状态使用256维；启用RF-lite时使用266维；F1原生`P2-A1`保留其严格对照身份并单独报告资源口径|state、MAC和F3块尺寸单测|verified|
|M23-05|真实域×类INT8知识|从base cache已绑定的`int8_domain_class_center_lowrank_residual_radius_v2`重建瞬时域×类中心；持久cache只复制INT8/FP16方向组件|registry、dtype和只读性单测|verified|
|M23-06|类LOO交互流形|共享域偏移子空间加逐旧类leave-one-class-out交互子空间；逐类投影经固定收缩后进入transported prior并冻结在Stage2-B状态|逐类本类排除、非零交互偏移、中心变化与registry绑定单测|verified|
|M23-07|保留地面流形外共享偏移|共享偏移分解为流形内分量和固定收缩的流形外分量|非零流形外合成例单测|verified|
|M23-08|Stage2-B域状态冻结并复用于Stage2-C|先由旧类support构造不可变domain state；注册新类时只复用，不重估|对象摘要严格相等与row闭合单测|verified|
|M23-09|模块二直接输出中心、中心不确定性、样本权重和域协方差|`M23CenterEstimate`显式返回四类对象|形状、有限性、任务权重和新类target-only单测|verified|
|M23-10|禁止统一平移support间接改中心|LDA使用显式center override；原始support只用于残差/协方差|输入不变与center override审计单测|verified|
|M23-11|旧类地面先验与target support后验融合|仅旧类identity160使用transported prior；FFT/RF-lite和全部新类保持target-only|旧/新先验掩码单测|verified|
|M23-12|中心不确定性进入LDA截距|加入`-0.5 tr(Σ^-1V_c)`类级惩罚|高不确定类截距下降单测|verified|
|M23-13|PSD目标域nuisance covariance|PSD加法和正jitter；不再自由仿射扭曲support|nuisance与总协方差最小特征值单测|verified|
|M23-14|旧/新任务等权协方差|旧类、新类按类内可靠性归一后固定`0.5/0.5`合成|显式任务权重与support权重和单测|verified|
|M23-15|RF块强收缩且跨块软关闭|K5只允许RF-lite对角；K10默认对角且跨块为0|RF协方差和cross-block零单测|verified|
|M23-16|K1/K2独立保守统计区间|K1为IF prototype/对角头，K2为IF对角任务头；两者RF分类门恒为0且不做LOO|K1/K2分支单测|verified|
|M23-17|IF-base／RF-lite-diag安全LOO|K5聚合全support的help/harm后只作一次全局门控和回退；K10才执行类别级门控；均含复杂度惩罚|K5全类gate严格相等、K10门范围、强复杂度回退和no-harm审计单测|verified|
|M23-18|F3紧凑量化|256/266维头使用双层INT8残差、FP16块scale和bias；量化前FP32参数只在fit中瞬时用于support一致性审计，不进入持久状态|无FP32旁路、support一致率、compiled bytes和完整retained-state bytes单测|verified|
|M23-19|预测翻转归因|独立truth-last分析输出`N_help/N_harm`、错误迁移、McNemar和类簇bootstrap，并按scene/K/receiver/role/class分层|纯函数单测通过；真实scorer待prediction|deferred_by_evidence|
|M23-20|完整因果臂F0–F5|F0原生`P2-FULL`、F1原生`P2-A1`、F2低权重RF32、F3 RF-quality、F4 RF-lite-diag、F5安全门控|目录、配置哈希与合成row闭合单测|verified|
|M23-21|四状态命名|输出显式标记`DA0_REG0/DA1_REG0/DA0_REG1/DA1_REG1`；REG0新类指标写`N/A_UNREGISTERED`|四状态主效应和difference-in-differences单测|verified|
|M23-22|query边界|fit API不接收query；每条query独立全注册类argmax；无配额、重排或状态更新|fit签名和truth-unopened artifact单测|verified|
|M23-23|旧D92默认行为不变|M2.3独立opt-in入口和cache schema；旧v2 cache与P2-FULL路径保持原样|既有量化/cache/row/scorer相邻回归48项|verified|

## 首轮真实实验

- run ID：`erbt_idr_m23_rfguard_targetscreen_20260820_v1`。
- row 1：receiver=`3-19`，method seed=`7282101`，`K1/new20`。
- row 2：receiver=`3-19`，method seed=`7282101`，`K10/new5`。
- 每条row执行F0–F5，固定support seed=`7282201`、query seed=`7282301`、draw seed=`7282401`和三个`leo_*_weak`场景。
- 首轮只用于可证伪筛选。若F5相对F1的同row`H`、`min-old`或`min-new`出现明确伤害，保留完整负结果并停止扩矩阵；低性能不触发技术停止。
- 技术停止仅限协议/query越界、错误输入或checkout、输出覆盖、非PSD、无prediction闭合、scorer连接错误或重复确定性执行故障。

## 证据边界

M2.3完成代码和单测只证明实现闭合；真实scorer返回前状态为`NO_PERFORMANCE_RESULT`。两条首轮row即使为正，也只构成小矩阵研发证据，不自动成为fresh confirmation、完整125晋级或星载部署结论。

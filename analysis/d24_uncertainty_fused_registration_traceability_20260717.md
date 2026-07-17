# D24不确定度融合旧类与独立新类注册追踪

## 方法锁

D24在同一归一化160维ADV3B02身份空间中维护三类信息：

1. Phase1旧类地面多域锚：保持既有int8量化和只读不可变，只提供稳定身份先验。
2. Phase2目标旧类prototype：由当前receiver的合法`LEO_weak` old support生成并以FP32保存，只负责域校正。
3. Phase2目标新类prototype：只由对应new support生成并以FP32追加，禁止接触地面旧类锚。

query只接收冻结状态并对全部注册类逐样本评分，不参与prototype、半径、融合权重、格式、阈值、候选或回退选择。

## 数学定义

所有输入先做`L2`归一化，余弦距离为`d(a,b)=1-a^Tb`。

对旧类`c`，从int8多域锚解量化得到`A_c={a_dc}`：

- 地面中心：`g_c=norm(mean_d a_dc)`。
- 地面不确定度：`r_g,c=Q90_d d(a_dc,g_c)`。
- 目标中心：`t_c=norm(mean_i z_ci)`。
- 目标不确定度：`r_t,c=Q90_i d(z_ci,t_c)`；K=1时使用method-lock常量`r0`。
- precision：`w_g,c=1/max(r_g,c,r_min)^2`，`w_t,c=K/max(r_t,c,r_min)^2`。
- 目标权重：`lambda_c=w_t,c/(w_g,c+w_t,c)`。
- 融合旧类中心：`p_c=norm((1-lambda_c)g_c+lambda_c t_c)`。
- 融合半径：`r_c=sqrt(1/(w_g,c+w_t,c))+lambda_c(1-lambda_c)d(g_c,t_c)`。

对新类`n`：`p_n=t_n`、`r_n=r_t,n`，不做ground融合。Stage2-C只追加新类FP32 prototype/radius/count，不改旧类target prototype、lambda、radius、class handle或score列。

类间几何门为：

`d(p_i,p_j)>r_i+r_j+m_sep, forall i!=j`。

该门只使用support-derived状态。若违反，D24候选失败并记录冲突对、gap和角色隔离审计；首版不得通过query调半径或用类别配额修正。后续若需优化，只允许support-only、逐类局部且不回写已冻结旧类状态的机制。

## 追踪矩阵

|ID|Requirement|Target files|Status|Verification|Notes|
|---|---|---|---|---|---|
|D24-01|int8 ground旧类锚只读，target旧/新prototype均FP32且同一160D空间|`code/cvsrffi/stage2_uncertainty_proto_fusion.py`|verified|D24独立11项与CIAF联合23项PASS|API不持久化sample feature|
|D24-02|ground radius由多域int8锚估计，target radius只由对应LEO_weak support估计|同上|verified|radius/K1 r0公式测试PASS|正式v2 ground cell radius仍待共同bundle重建|
|D24-03|旧类按inverse uncertainty融合，新类完全独立注册|同上|verified|closed-form与target-only append测试PASS|new suffix不读取ground类锚|
|D24-04|Stage2-C append-only，old target/fusion bytes与raw score列bitwise unchanged|同上；runner|implemented|模块prefix/hash/score测试PASS|runner artifact证据仍pending|
|D24-05|全类满足中心距离大于半径和加margin；失败列出collision pairs|同上；audit|verified|全pair geometry/collision测试PASS|首版为cosine门；球面角门列入D25扩展|
|D24-06|作为独立D24候选进入15fold development support筛选|`code/scripts/run_d19_support_only_ciaf.py`|pending|runner tests+N607 log|与Z0/B3同row比较|
|D24-07|资源报告ground int8、target FP32、融合元数据、MAC、延迟、scratch|runner；resource audit|implemented|模块状态/MAC/scratch测试PASS|实测延迟与runner总资源仍pending|
|D24-08|保持single-LEO、clean/source不可达、query-only-test及全Oracle禁令|runner；tests；report|implemented|模块public fit API无query/role/quota/source/clean输入|runner闭包仍pending|
|D24-09|K10开发锁参后覆盖K1/K5，不从隐藏K10状态裁剪|runner/matrix|pending|K1/K5 capsule tests|每K独立从可达support建state|
|D24-10|完整日志、逐类floor、Git提交与N607报告|active report|pending|full-log analysis|正式125前不作query性能声明|

## 与D23压缩路线的关系

D23保留为prototype存储格式ablation；D24是用户最新指定的主机制。首个D24实验固定target prototype为FP32，以隔离不确定度融合和半径几何本身的效果。若D24通过support floor门，再用D23比较FP16/INT8是否在不损失floor的前提下获得更优状态/延迟Pareto。

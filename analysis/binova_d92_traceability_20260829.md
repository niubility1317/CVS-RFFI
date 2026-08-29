# BiNOVA-D92设计实现追踪表

设计来源：用户提供的两阶段BiNOVA-D92报告；协议收紧依据：`项目.md`、`AGENTS.md`及[设计规格](../docs/superpowers/specs/2026-08-29-binova-d92-design.md)。

|ID|来源章节|可验收要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|BNV-01|总流程|阶段A仅旧类support，阶段B使用old/new support且冻结`phi_D`|`stage2_binova_da.py`、`stage2_binova_reg.py`|pending|聚焦测试|无query训练入口|
|BNV-02|3.2|按类聚合后几何中位数形成类均衡`c_D`|`stage2_binova_features.py`|pending|置换/重复样本测试|所有类同公式|
|BNV-03|3.3|rank-16 late-time样本相关非线性残差|`stage2_binova_da.py`|pending|零初始化与非仿射行为测试|使用pooled late-time摘要|
|BNV-04|3.3|rank-32 identity条件残差并保持160维|`stage2_binova_da.py`|pending|形状/梯度测试|最终供D92使用|
|BNV-05|4|实现affine-leak软惩罚，不做硬投影|`stage2_binova_da.py`|pending|仿射/非仿射对照测试|A4将权重置0|
|BNV-06|5|旧类4+2角色轮换且每类次数均衡|`stage2_binova_da.py`|pending|轮换覆盖测试|不使用真实新类|
|BNV-07|5–6|五fold、每类8fit+2held且物理ID互斥|`stage2_binova_da.py`|pending|fold互斥测试|K10第一版|
|BNV-08|5–6|阶段A原型CE、SupCon、pseudo-D92、连续遗忘、信任域损失|`stage2_binova_da.py`|pending|逐损失有限值/反传测试|未定义physical-view随机损失不实现|
|BNV-09|6|固定D-A/D-B/D-C权重，不做大网格|`stage2_binova_da.py`|pending|配置测试|实现名DA_PLAIN/DA_PSEUDO/DA_STRONG|
|BNV-10|7|持久化`phi_D`，不持久化临时分类头|`stage2_binova_da.py`|pending|state字段测试|无target head|
|BNV-11|9|`phi_R`在256维identity+FFT联合空间零初始化|`stage2_binova_reg.py`|pending|逐行identity测试|RF32关闭|
|BNV-12|9|`q_i`由cross-fit D92距离、margin和熵生成|`stage2_binova_d92.py`、`stage2_binova_reg.py`|pending|held行移除测试|无query角色|
|BNV-13|10|可微task-balanced shrinkage D92经Cholesky反传|`stage2_binova_d92.py`|pending|梯度/正定测试|OAS解析代理，最终头仍为精确D92|
|BNV-14|11|旧/新CE、双向侵入、遗忘、拓扑损失|`stage2_binova_reg.py`|pending|手算fixture测试|全部support-only|
|BNV-15|12|old/new梯度冲突时投影新类梯度|`stage2_binova_reg.py`|pending|反向向量测试|不修改`phi_D`|
|BNV-16|13|S0/S1/S2分别重拟合精确D92并support-only回退|`stage2_binova_lifecycle.py`|pending|三状态回退测试|query打开前冻结|
|BNV-17|四状态规则|输出`DA0_REG0/DA1_REG0/DA0_REG1/DA1_REG1`及S2附加态|`stage2_binova_lifecycle.py`|pending|状态键测试|REG0新类指标为N/A|
|BNV-18|协议|固定received IQ、support/query ID互斥、query无标签/角色/更新入口|`stage2_binova_features.py`、runner|pending|协议负测|复用VALIDATED_ONCE句柄|
|BNV-19|16|阶段A A0–A4与阶段B B0–B3最小矩阵及自动继续门槛|runner、计划、报告|pending|inspect-plan测试|B4/B5后置|
|BNV-20|交付|本地验证、独立P0/P1、Git发布、N607 prediction、独立评分和报告|报告/产物|pending|OID/PID/artifact/scorer读回|按最高已证状态报告|

## 明确优化或不实现项

|ID|原报告内容|状态|原因|
|---|---|---|---|
|BNV-X1|未定义的随机`physical-view loss`|rejected|会产生不可审计的第二view语义；第一版只用同一received IQ确定性数学表征|
|BNV-X2|阶段B允许`phi_D`以`phi_R`的0.05倍学习率继续更新|rejected|会把实际新类身份混入长期域状态；本实现完全冻结`phi_D`|
|BNV-X3|用最终query指标执行S0/S1/S2回退|rejected|违反query只测试；回退仅使用support cross-fit|
|BNV-X4|第一版直接运行完整A/B消融和多seed|deferred|最小流程先单seed关键矩阵，达门槛后再扩展|

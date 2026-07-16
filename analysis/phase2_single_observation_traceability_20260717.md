# Phase2单物理样本单LEO接收观测追踪表

日期：2026-07-17

状态：协议先行修订

## 1. 变更原因

旧D1/D3路线把同一个clean/raw物理IQ样本分别叠加`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`，再把三份观测用于Phase2适配或评估。该构造不符合单颗卫星接收机的实际可达信息：一次接收只能得到一个既定LEO星地信道下的IQ观测，不能同时取得同一发射波形在三种场景下的平行观测。

本次先修改`项目.md`，再据此审计和修复cache builder、sealed package、validator、runner、实验矩阵与报告。旧结果只保留为历史诊断，不得继续用于候选选择、超参数锁定、正式排名或性能声明。

## 2. 需求到实现追踪

|ID|协议需求|`项目.md`落点|实现/验证状态|
|---|---|---|---|
|SO-01|一个clean/raw物理IQ样本在进入Phase2前必须且只能叠加一次，并恰好叠加一个`leo_*_weak`场景，形成唯一接收IQ观测|第7.1.1节|本次落盘|
|SO-02|同一matched receiver×seed×K×new规模下，三个场景的support∪query全角色物理样本并集两两不交，同场景support/query也不交|第7.1.1节、第10.3.1节|本次落盘；builder/validator待修|
|SO-03|Phase2密封包必须逐样本记录不可重命名的pre-overlay稳定根ID、唯一scenario、恰好一个satellite seed值和overlay provenance|第7.1.1节|本次落盘；package schema待修|
|SO-04|禁止从同一clean/raw样本派生多场景、多信道或多子样本用于Phase2训练、适配、注册、校准、选择或评估|第7.1.1节|本次落盘；历史D1/D3需降级|
|SO-05|允许的多view只能由已接收的固定LEO_weak IQ在Phase2内执行接收侧信道均衡、增强、变换或表征提取；不得重新访问clean/raw、叠加另一LEO状态或恢复另一物理观测|第7.1.1节、第8节、第10.3.1节|本次落盘；首个合法view待实现|
|SO-06|同一接收IQ派生的计算view不得计作独立support或增加K；只有support view可参与拟合，query view只能用于当前样本推理且不得更新任何状态|第7.1.1节、第10.3.1节|本次落盘；计数与query-fit审计待修|
|SO-07|历史D1/D3因确认使用同一物理样本的三种LEO观测，统一标记`PROTOCOL_INVALID_FOR_PHASE2_SINGLE_OBSERVATION`|第7.1.1节、第12节|本次落盘；报告待更新|
|SO-08|极轻型正式上限调整为adapter参数不超过80,000、适配不超过30epoch、持久状态不超过256KB、无dense query图|第10.3.1节|本次落盘；资源validator待修|
|SO-09|三场景仍需完整覆盖，但它们是三个独立接收样本单元，不是同一物理样本的三份view|第8节、第10.3.1节|本次落盘；矩阵生成器待修|

## 3. 合法与非法view边界

合法：

- K-shot由K个互不重复的独立物理样本构成；例如K5是5份独立接收IQ，不是一个样本的5个副本。
- 从唯一已接收LEO_weak IQ执行信道均衡、时频增强、接收侧变换，或计算固定时域、频域、时频和RF统计表征。
- 对唯一已接收IQ执行预登记、接收侧可计算的变换，并将多个分支用于同一个物理样本的联合表征、一致性约束或轻量适配。
- 所有分支共享同一个`physical_sample_id`、scenario、satellite seed和support/query角色；K只计一次。
- 每个派生view记录`parent_received_iq_sha256`、`operator_id`和`view_seed`；operator不得调用LEO channel simulator或创建新的overlay provenance。
- 只有support派生view可以参与适配或状态更新；query派生view只服务当前query的逐样本推理。

非法：

- 从同一clean/raw IQ分别叠加三种LEO场景。
- 从同一clean/raw IQ生成多份带不同LEO状态或不同信道随机性的子样本。
- 把同一接收IQ的计算分支当作多个独立support或独立query，从而放大K或样本数。
- 在Phase2中重新打开clean/raw IQ、叠加另一LEO状态、逆推出clean参考或生成另一物理接收观测。

## 4. 旧结果边界

D1和D3的已保存score、loss trace、逐类结果与资源数据继续保留，作为“旧三场景同源多观测构造为何会产生表观性能”的历史诊断证据。但它们不再满足当前正式Phase2单观测协议，不能用于：

- 选择下一candidate或超参数；
- 证明K10/K5/K1性能；
- 形成125确认矩阵；
- 与identity-only单qKNN做正式Pareto排名；
- 声明Stage2-B/C部署性能或floor达标。

## 5. 后续验证

1. 审计现有cache中的`physical_sample_id`跨scenario重用。
2. 修复稳定根ID，使其绑定overlay前dataset artifact、member/split和original record index或等价不可重命名lineage token。
3. 修复离线builder，使物理样本先分配唯一scenario，再执行一次LEO_weak overlay。
4. 在pre-open validator中检查三场景support∪query全角色物理ID并集互斥、同场景support/query不重叠及每个样本唯一overlay provenance。
5. 对接收后view核验父IQ SHA、operator和view seed，并阻断query view拟合。
6. 修复runner，使计算view只能关联同一sealed IQ并按一个样本计数。
7. 先运行单receiver、单development seed、K10、5/10/20真实new TX验证，再决定是否扩展K1/K5和正式125矩阵。

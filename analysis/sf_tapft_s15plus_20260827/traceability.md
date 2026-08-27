# SF-TAPFT S15+设计追踪

|ID|来源章节|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|S15P-01|四|300步专用cosine：reference=300、warmup=0.10|adapt、配置、测试|implemented|RED/GREEN+OOF|F1单变量|
|S15P-02|五|cross-fitted OOF全局温度校准|adapt、bundle、prediction、测试|implemented|argmax不变、NLL非增|F2；温度进入严格bundle回读|
|S15P-03|六|冻结embedding的head-only预训练，再联合head+norm|adapt、配置、测试|implemented|昂贵backward计数下降|F3；前60步复用冻结embedding|
|S15P-04|十|使用长程M02 cross-fitted logits做OOFKD0.1|adapt、artifact、测试|deferred|每个样本teacher未见自身|缺严格绑定的长M02 OOF teacher logits，不以source teacher替代|
|S15P-05|七、十六|S02进入新的独立query|query配置、报告|deferred|truth-last评分|必须使用未参与选择的新query|
|S15P-06|八|S16-A/B混合norm范围|adapt、配置、测试|implemented|精确trainable-name集合|Q2A/Q2B保持4500步长程结构对照|
|S15P-07|十六|Q3=S02的300步快速版|配置|deferred|Q1通过后再启动|条件候选|
|S15P-08|十三|向量化LOO prototype|adapt、测试|implemented|与旧实现logits/loss等价|含梯度有限性测试|
|S15P-09|十三|稀疏validation计划|adapt、配置、测试|implemented|仅预登记step验证|F1-F3启用；Q2保持S00逐步选择对照|
|S15P-10|十三|仅潜在top-k候选计算state-distance|adapt、测试|partial|选择结果等价|已从每步全模型CPU距离改为稀疏验证时许可参数距离；top-k前置判定未单独实现|
|S15P-11|十三|KD=0不复制完整teacher|adapt、测试|implemented|输出等价、无teacher副本|teacher仅在KD权重大于0时创建|
|S15P-12|十三|snapshot只保存head与许可delta|adapt、bundle、测试|implemented|非许可state严格相等|重建以完整anchor为底座，snapshot仅许可参数+head|
|S15P-13|十一|deployment-only全support单次入口|runner、CLI、测试|pending|无CV、固定300步|资源基准|
|S15P-14|十二|S02冻结前缀缓存|模型、测试|deferred|logit差<1e-5|Q1通过后实现|
|S15P-15|十四|delta-only FP16 bundle及严格loader|bundle、runner、测试|pending|重建logit差<1e-5|先FP16 delta-only|
|S15P-16|九|类别可靠性head anchor|adapt、配置|deferred|单独变量实验|不与首轮混合|
|S15P-17|十五|保留M02-HP与S15-FAST档位|配置、报告|evaluated_no_promotion|配置严格回读|F1–F3快速档性能未过门，不提升默认|
|S15P-18|十八|报告研究/部署时间、forward、validation、bytes、RAM/VRAM、延迟|runner、报告|support_closed|artifact字段完整|wall/RSS/启动采样显存已闭合；独立query推理延迟尚未测|
|S15P-19|十九P2|跨seed/scene/receiver/K确认|后续矩阵|deferred|独立确认|不阻塞首轮|

## 首轮实验冻结原则

- F0：原S15，300步、reference4500。
- F1：只把reference改为300、warmup改为0.10。
- F2：F1+OOF温度。
- F3：F1+head60/joint240+OOF温度。
- F4：F3+M02 cross-fitted OOFKD0.1。
- Q2-A/Q2-B：相对S00只改变norm混合集合，保持4500步长程support OOF；不与F轴混合。
- Q1使用新的独立query；若合法新query尚未就绪，保持`deferred`而不复用旧truth。

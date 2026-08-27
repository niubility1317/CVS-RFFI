# SF-TAPFT Fast-Strong V3设计追踪

来源：用户提供的《SF-TAPFT Fast-Strong V3》设计报告，2026-08-27。

|ID|来源章节|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|FSV3-01|§1、§12|固定目标余弦head、`rho=0.5`、scale=8、balanced CE、LOO-proto=0.5、L2-SP=1e-4、label smoothing=0.05|适配核心、矩阵配置、测试|verified|81项聚焦测试通过|H0–H4/H6保持`rho=0.5`|
|FSV3-02|§1、§5|禁用time Adapter、完整t3、frequency/domain分支；首轮all-time-norm，H6选择S02的`head+t3.norm(weight+bias)`结构|适配核心、矩阵配置、测试|verified|矩阵解析与trainability回归通过|H0–H5为all norm，H6为S02 t3-only|
|FSV3-03|§2、§3|H0严格复跑S15轨迹：300步、ref4500、warmup0.05、head LR1e-3、norm LR1e-4|Fast-Strong V3配置与runner|verified|H0配置实例化通过|待N607真实等价验证|
|FSV3-04|§3|H1追加150步局部cosine尾段；H2追加300步局部cosine尾段，使用报告给定head/norm起止LR|适配核心、矩阵配置、测试|verified|局部cosine端点测试通过|总步数450/600|
|FSV3-05|§4|H3在适配后缓存60×160 embedding，冻结backbone/norm，进行50–70步head-only polish|适配核心、runner、测试|verified|缓存精修不增加backbone训练forward|首轮固定70步|
|FSV3-06|§8、§11|H4维护trainable-delta EMA，`beta=0.99`，只覆盖head与许可norm，并与final状态按support-only规则选择|适配核心、bundle、测试|verified|许可delta EMA测试通过|不触碰非许可状态|
|FSV3-07|§6|H5实现类置换不变的自适应`rho_c=0.25+0.5(1-q_c)`，`q_c=R_c·sigmoid(M_c/tau)`|适配核心、审计、测试|verified|公式数值与样本置换测试通过|R为球面集中度，M来自冻结源logits|
|FSV3-08|§7|H5加入可靠性head anchor，首轮`lambda_h=0.01`|适配核心、测试|verified|按`q_c[1-cos]`实现并随配置启用|H5使用0.01|
|FSV3-09|§8|M02 OOF teacher蒸馏不进入每次部署主线|设计追踪、最终报告|deferred|报告边界|只有现成cross-fitted teacher时才有意义|
|FSV3-10|§9|OOF温度只对达到性能门槛的高性能候选拟合，不能修复欠拟合候选|runner、矩阵选择、报告|implemented|温度只读support OOF，输出校准前后NLL|晋级仍同时要求BA和floor，Query不参与|
|FSV3-11|§10|保留向量化LOO、稀疏validation、KD=0不复制teacher、许可delta snapshot、anchor重建|现有核心、回归测试|verified|81项回归通过|无teacher复制|
|FSV3-12|§10|部署模式单次full-support fit，不运行4-fold；输出delta-only bundle：head、norm delta、temperature、class IDs、base checkpoint ID|runner、bundle、严格loader、测试|verified|FP16 delta严格读回且测试bundle=2593B|训练选择仍做4折，部署只加载一次full-support结果|
|FSV3-13|§11|最小矩阵H0–H5；Q2结束后只追加H6，不做全排列|矩阵配置、预登记报告|verified|七行矩阵解析通过|H6=S02结构+H3流程|
|FSV3-14|§13|晋级门槛：BA≥86.17%、floor≥60%、NLL≤0.5394；提升判据BA>86.67%或NLL<0.5094且floor不降|selection与最终报告|pending|待实现|最大Q180用于真实对比|
|FSV3-15|§13|资源门槛：≤约1500变化元素、无Adapter/完整block、分钟级、bundle<10KB、Query推理成本不变|审计、GNU time、GPU采样、报告|pending|待验证|H0–H6逐行报告|
|FSV3-16|用户当前请求|预测前不读Query truth；每候选在真实最大Q180上比较`DA0_REG0/DA1_REG0`|query closure、独立scorer、报告|pending|待执行|Q60+Q120零重叠并集，每类30条|
|FSV3-17|§1、§11|明确排除`rho=1`、R16/R32 Adapter、完整t3、P4深层解冻、KD0.1、F1/ref300、head-first|矩阵配置、报告|verified|七行配置审计通过|不进入H0–H6|
|FSV3-18|全文|完整记录实现、同row实验、各类准确率/NLL、资源、失败与限制并发布GitHub|根报告、Git镜像、设计追踪|pending|待交付|严格设计追踪，不以单一winner替代全矩阵|

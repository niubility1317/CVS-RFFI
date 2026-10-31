# MILKR 当前support选择源残差强度

R42复用R41唯一固定600步/每orbit的纯源NN和源K1规则，不重新训练网络，也不继承目标SUP适应状态。新假设是多样本SUP能评估源残差的可靠性，避免所有case固定完整残差。K1没有类内物理留出，保留原fullNN源先验；K>1在固定强度0/.25/.5/.75/1上，联合24核×5ridge作当前SUP物理OOF选择，共600候选。优先old/new宏H，其次较小平方误差、较大ridge、精确tie时较弱残差。相同物理fold用于全部强度，每fold带宽/gate/KRR仅trainSUP；选择后只保留本格一个模型，查询不作选择或五模型集成。

base和full分别是同一处理输入的原CorrCCA和有界NN特征，计算base+s*(full−base)后沿原RF各branch单位化。端点0/1直接返回原特征避免改变端点数值；head概率不混合。S0的两个表示都来自raw，S1都来自DURA输出，故不是把raw混入DURA support，也不增加K/物理记录/标签。全部强度同样遵守逐记录计算、跨K/N/S独立，query无批统计、角色、配额、标签或状态更新。600候选OOF用于选择，不能当独立泛化证据。

源NN checkpoint只含源CCA/NN参数，固定在R41目标预测前保存，原target SUP KRR/prediction/truth/optimizer从不加载。NN加载前核对其源数据/F/G角色与源训练零target/query/Utruth读入；同一次launcher首步无query smoke检查真实Tri-SIG-TIME SSL、DURA及已加载NN，对2条合法source L的raw/G输出检查base/full和五强度逐记录独立性。PASS直接继续，模型对象在进程内传给源冻结保存，不引入smoke票据或审批。

新源拟合记录为空、新NN更新0、model_fitting_rows0；父训练6次模型/7200 NN更新仅作继承历史，部署权重原600步/每orbit。源K1 120grid及规则可继承，因为K1 full端点不变；父K20验证只标继承，不冒充新选择器已验证。CCA557440FP64+NN684032FP32、合1241472数值参数/7195648未压缩字节不变。额外成本为多样本SUP的600候选拟合与同路径双表示缓存，实际时间报告；查询只用本格已选模型。

Tri-SIG-TIME SSL、DURA、原C4head及buffers冻结，R31headgate/prior和R26 query融合保持，不引入R38/R39 RFgate/prior。源L6300/12600配对祖先，F曾见源六TX，代理old4/new2及源3景对目标6景/已曝光开发基准边界保留。源encoder不由target拟合，SUP只用于本格KRR/表示强度选择。

首RX1-1/P01/MIX01/seed2026092701，全部K1/5/10/20×N2/5/10/20，B24+C96共120预测、40fit/40fusion全部固定后独立truth-last。两S每KN raw较匹配原raw≥5pp、QF>raw且raw旧新均值guard≥原版−1pp才冻结同法扩其余六RX；最终七RX等权all16目标不变。G-only同输入配对与融合跨输入差分开。

sourceGPU4只smoke/复用、targetGPU0，启动前检查余量并保护其他owner。最小流程的一次独立P0/P1、针对性检查、一次传输校验/remotecompile、实际PID/CWD/argv/GPU/log增长及独立评分保持。实际commit、命令、环境CWD、来源、数据句柄、矩阵与技术停止条件见experiment.json。

## 发布前核验

36项相关检查及4项非法协议输入检查通过，compile/diff检查通过。唯一独立P0/P1审查PASS，独立新增6项+worker alias共7项PASS：NN加载前parent_metadata核对源数据/F/G角色、固定600步源审计及纯源checkpoint；smoke返回同一模型上下文给freeze，无新训练或父target状态；25字段为同处理路径full/base各12RF和同一original headp，S1无raw混入；K1 full精确端点；K>1五强度物理TRAIN-only OOF600候选排序正确，查询同blend；旧K20源验证明确仅继承；worker新fit/桥及120行计数闭合。一次合并真实F/G/NN无query smoke为launcher首步PASS继续。

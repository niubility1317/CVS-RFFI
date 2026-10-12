# 普通 cos 头：跨 RX 元反馈利用 U 的源端预登记

依据用户新上传优化报告，研究主线改为用其他源 RX 的真实身份反馈选择 U 更新；原30轮和最常用cos头要求保持。此前分类路由/多专家方案不继续，E30＋70预算建议已被拒绝。刚准备的浅中层矩对齐也尚未发布，记为备选而非已运行结果。本文机制仅待验证，84%尚未达成。

先执行只读源诊断：三个旧SELF_SSL30原完整6660步，共19980步遥测；全3×56700=170100 U的冻结置信/预测类/RXday普查；每seed、5RX×3day、每组64个U，共45组2880个U梯度探针；完整6300 L的logRMS/FFTflatness/lag16变量方差/分位数和TX/RX分布；已有五RX三成员27000源V预测的共同错误、rescue/harm、净纠错和NLL检查。真实旧checkpoint只诊断，严格原配置/真实sourcecontract/final30/完整从零祖先核对，无新身份更新、旧输出覆盖或target访问。梯度探针有限，不冒充全U梯度；原optimizer状态未保存，探针P=I，明确是SGD局部方向诊断。冻结置信掩码不是历史两遍稳定门控；实际历史接受率来自完整训练日志。旧日志不包含逐RXday接受身份，不能事后编造，新实验实际记录。RMS已归一化、lag16仅经验量，无可信eq0/eq1物理配对、不把补零叫新增观测。

三个候选 CE_MATCHED／UNIFORM_SSL／META_SSL，全五整源RX留出，单模型seed2026092701，共15新native模型，每模型从零30×222=6660。原L/U/V=7/63/30、训练池标注率10%，fitL5040/heldV5400；SSL合法fitU45360无TX标签，整个heldRX L/U均排除。元反馈只取训练L，不取V，因此内层反馈RX属于训练域，外层原V只最后冻结后评分。

每步四fitRX各32个L，RX内TX/day均衡完整遍历，同一128L及其CE在三arm完全匹配。一个fitRX按step循环作为反馈M，其余三RX是内层；SSL的128U来自内层三RX43/43/42，分成4个32样本微批。反馈额外使用同批32L的确定性eval forward/gradient，无新增标签接触；UNIFORM和META也匹配这些前向和梯度诊断计算。CE_MATCHED纯CE没有U/teacher/metagrad昂贵分支，实际成本分别报告，不声称等计算。前5轮只CE；1111至6660步启用U，自身学生clean确定性软目标、confidence≥.95、opaquePID在相邻完整源RX遍历中同预测且高置信的两次稳定门控；强视图额外实通道noise标准差.015和私有dropout，ramp1110步、名义KL系数.25。无EMA、预训练teacher、星地/其他增强。

反馈梯度对象为全部66个optimizer骨干张量，排除常规cos类别权重，明确实际参数名；包含真实浅中深可训练身份路径，不建立专用分类器。P为当前AdamW bias-corrected second moment的 detached逆平方根（加optimizer epsilon）；每个U微批的梯度与反馈CE梯度算P度量cosine。正值除以正值均值、cap2，非正全零时不执行U更新；权重停止梯度。UNIFORM四权重恒1，其他流程相同。此为冻结P的一阶局部近似，未包含完整AdamW momentum/decay变化，不称精确虚拟AdamW贡献；是微批重加权而非每个packet的精确元导数。正内积不证明伪标签正确或实际跨域收益。全步记录raw/weightedKL、RXday实际接受率、预测类分布、P梯度cosine、权重、加权U/L骨干梯度范数比、梯度范数及30轮均值。

单身份表征joint160保持，分类严格30×cos(joint160,W6×160)，常规归一化eps1e-4，无margin、RX专家、路由或集成，177025参数。Phase1普通CE/SSL训练该权重；Phase2可丢弃它，以合法support重新拟合cos头。已经验证同一表征可接新9类权重，不使用任何Phase2样本；本次不改变Phase2权限。

训练AdamW WD.0001、lr.0002至.000001按30轮cosine、clip10、label_smoothing0。所有实际新初始化无checkpoint继承；历史完整diagnostic产物不参与新初始化/target选模。source晋级固定为META均值≥CE_MATCHED+1pp且≥UNIFORM+0.5pp、分别至少4/5RX非负、最差RX分别非劣1pp。全15row完整后判定。若通过，预先固定全五sourceRX、每step轮换4＋1，L128在排序RX分26/26/26/25/25（反馈25或26），U在四内层各32；三arm×三primaryseed2701/02/03共9模型，每模型仍从零30、同单cos。逐row冻结后自动clean＋六practical各168000，prediction完成后独立truthlast；未通过不访问target，不更改阈值或按目标分数重排/重跑。现有基准已暴露，报告保留该背景。

来源：用户报告及[Ren等的样本重加权原始研究](https://proceedings.mlr.press/v80/ren18a.html)。本任务的训练L跨RX反馈、AdamW冻结P和微批均为任务扩展，不把原论文结果当本项目性能证据。[SWA原始研究](https://arxiv.org/abs/1803.05407)留作独立备选，不直接跨随机seed平均参数。现有输出集成已完成，历史正式三primaryseedclean81.3254%，低于84%；该事实只报告，不用其target错误挑新机制。80/20全标签参考现在已完成，但训练角色与更新预算不同，绝不继承它或解封本run U真值；匹配全监督参考需另行明确数据权限。

唯一owner、普通N607账号、GPU总实验进程≤3含其他owner/free≥12000MB、不改健康旧任务。一次P0/P1、聚焦负测和真实合法checkpoint noquerysmoke、actual code/config Gitcommit；一次不可覆盖release/run/output、N607preflight、单archive SHA比较和remotecompile，postlaunchPID/CWD/argv/GPU/loggrowth。精确矩阵/command/environment/path/stoprule/expectedartifacts见experiment.json。仅系统技术/数据权限/输出碰撞/不可执行/不能闭合法预测/两row同指纹预测前故障停止所属run，低准确率不停止；保留全部产物。旧额外seal/hash/report-onlyreview不阻断。

本地实际CUDA聚焦检查PASS：13个负测、普通cos等式及替换9类、逐包无状态更新、冻结P虚拟SGD数值方向导数、真实元反馈梯度/冲突权重；反馈L采样各TX5或6。唯一独立P0/P1审查PASS；采样顺序修复定向复核PASS，160个32包批次六类各5/6、每RX首遍1260索引唯一，U不读取TX。无额外重复审查。

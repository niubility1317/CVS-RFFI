# F＋跨 RX 同 TX：实际特征位移风险约束

状态：LOCAL_VERIFIED。实现、聚焦检查和唯一独立P0／P1审查完成，准备发布首seed两条件；没有正式训练或新目标结果。

基底固定为原始 F CROSS_RX50：真实三路独立 SIG_TIME→实际joint160、统一居中物理Sinc、RDrop1、普通增强／LS.01、旧门控＋ownEMA＋cos、默认头正常联合。历史85.1139%仅方法锚点，增量配对复用本机BASE-s2026092701／02／03，不继承或重复训练其权重。无星地增强，测试仅clean。

| 条件 | 同一原生步后的处理 | 首seed | 源条件通过后原样扩展 |
|---|---|---|---|
| STEP_NORM | 保留原完整特征位移方向，匹配投影后实际存储归一化步范数 | 2026092701 | 2026092702／03 |
| UPDATE_CONE | 对完整实际特征位移施加三个当前L分类风险约束 | 2026092701 | 2026092702／03 |

每模型从零50轮＝33L＋17U、11100次原生主优化器更新。前5轮不变；此后每4步，1112至11100共2498计划事件，强度第6轮0.1至第15轮1。每合格事件3次特征VJP，最多7494／模型。新增同一步特征参数写回，主optimizer.step及moments仍每步1次；不增加模型前向、IQ读取、部署参数、推理步骤或头拟合。

## 依据、对象与算法

源域设计依据：固定原3BASE的完整源L类CE原始特征梯度观察到弱1／3冲突；上一轮原始梯度投影的源20-19 marginq10改善，而源14-7几乎不变且默认头召回下降。原始梯度方向与包含moment／自适应缩放／其他损失的实际参数位移不同。本轮以该源证据与优化器事实设计，不用前轮目标分数选类、改权重、阈值、seed或checkpoint。前轮全部负结果已保留。

历史RX1-Uincrement CONE是单路SIG_TIME、RAW／LS0／无普通增强的真正源折，只改临时U增量状态，四个探针均未达到原资格，未形成持久候选训练。本轮研究完整F实际步、三个当前类风险与持续特征纠正，和其数据／对象／轨迹不同；不换名复跑该源折，不加载其权重，也不重复末期微小CE贡献作为训练晋级门槛。

只使用当前实际L128和原两次RDrop的同一增强IQ。每包两次CE（LS.01）平均后，分别对弱TX索引1、3以及其余四类求和／128。三个损失仅用于提取当前特征导数，不增加或重加权原标量loss；原原生梯度、物理clip5和完整AdamW更新保持。两个弱类固定来自既有源V校准（14-7、20-19），不根据新目标调整。

在原主优化器之前保存实际部署id_backbone参数，排除整个原cls_head（W、gain及归一化参数）。完整原生更新（含当时合法U、RDrop、SupCon、moment与物理衰减）后得到d0。每模块尺度s为该模块d0的RMS，零步模块s=0且不动；x0=d0／s，Hi=gi*s并按行L2单位化。解min||x−x0||²，约束Hi·x≤0（三类风险）。用至多3×3小矩阵枚举活跃集；强度eta时采用x0＋eta(x−x0)，写回float32后的实际位移另行核对。

STEP_NORM按同一当前状态计算反事实投影，但保留原d0方向；匹配float32实际存储的归一化特征步范数，必要时固定40轮标量二分，误差≤0.001。它不是梯度范数，也不表示两个训练轨迹的幅度相同。记录分别保留candidate投影风险与实际applied风险；对照不得冒充通过投影约束。

预定义回退：小矩阵无合格解时零投影；完整强度写回后风险残差超0.001×原归一化步范数时零特征步；无法达到存储范数匹配时两种反事实均保留原位移。所有回退显式记录并阻止扩seed，初始行仍保留合法clean结果。约束只对完整强度、非存储匹配回退的UPDATE_CONE实际特征位移，在当前更新前头的L导数下一阶成立；不保证更新后头的非线性CE、其他RX或目标性能。

## 默认头、Sinc与EMA

默认分类头完整保留原生联合更新，不投影、不冻结、不另训。只在原生步之后增加同一步f参数纠正，不改变任何optimizer moment或额外推进step。钩子顺序为unit计数post→新特征post→原Sinc观察post→optimizer返回→原自身EMA；原生训练代码已核对先scaler.step，再_update_ema_model。

Sinc采用原p=p0+25000q、Fs=25MHz、物理Hz全局clip5、lr／eps单位映射及包含origin的物理衰减，原生阶段只执行一次，不叠乘历史倍率或二次衰减。后续约束为额外特征位移力，因此最终位移不声称仍等于无约束AdamW。模块RMS度量与导数内积对共同单位缩放不变；实际q.copy_递增叶版本，保留原滤波缓存失效。student、EMA、推理和导出使用同一坐标表示。

## 数据、冻结、扩种子与测试

复用既有READY／EXACT_MATCH源角色、原native缓存元数据r02与VALIDATED_ONCE clean capsule，不重建划分。源RX1／3／4／6／8、day1／2／3，split_seed392005；L6300可见、U56700真值隐藏、V27000。新模型严格scratch，自身EMA，无初始化权重或祖先。匹配原BASE checkpoint只用于一次丢弃的真实no-query smoke，不继承至正式训练。

源V只作no_grad既有默认头健康及实际joint160描述几何，不进入训练、拟合分类器或原型决策。out_RX只从描述质心中排除该RX，编码器已见所有源RX，不能称为未见RX模型验证。固定final50，不按V或目标挑checkpoint。

首seed两条件训练完成后、任何新目标汇总前，分别与同seedBASE比：每个弱TX的outRX marginq10≥BASE＋0.002，RXspread／gap≤1.05倍，零特征0；默认头整体下降≤0.2个百分点、最差RX≤0.5个百分点、任一TX≤1个百分点。2498计划、合格≥99%、实际非零纠正≥10%、数值／存储回退0、存储归一化范数误差≤0.001。仅通过条件原样扩02／03；失败的初始两行仍自动clean评分。门槛和顺序固定，不按目标调参、选父权重、重排或补seed。

逐row固定student final50和checkpoint内默认头，clean168000全部prediction完成后，由独立CPU truth-last评分。报告准确率、Macro-F1、6TX召回、最差RX／day、逐seed配对差、实际n、均值、样本SD与正差数、UPDATE_CONE−STEP_NORM、U曝光与接受率、执行和成本。n1无样本SD，全部负结果保留；接受率不代表伪标签真值正确率。已暴露Phase1探索，不宣称独立盲测、Phase2或星地性能。

## 发布、停止与验证

环境：native Windows Python3.10／PyTorch2.10+cu128、FULL_FP32_NO_TF32、threads2／workers0。CWD：IMMUTABLE_LOCAL_RELEASE/cvs_F_weakTX_actualStep50_local_20261020_r01。输出：LOCAL_RUN/20261020-phase1-FcrossRX-weakTX-actualStep-e50-clean-local-m2-r01，新建不可覆盖。物理GPU0全owner合计最多3个source／predict worker，派发前空闲RAM≥4.5GiB；独立CPU scorer最多1个。保留所有健康任务。

命令：

LOCAL_BASE_PYTHON -X utf8 -u -m experiments.cvs_f_weaktx_actualstep50.driver --profile LOCAL_PROFILE --metadata LOCAL_SOURCE_NATIVE_GATE_METADATA_R02 --base LOCAL_RUN/20261020-phase1-FcrossRX-weakTX-actualStep-e50-clean-local-m2-r01 --reference-base LOCAL_RUN/20261019-phase1-FcrossRX-TXrisk-geometry-e50-clean-local-m4-r01 --commit RELEASE_COMMIT --max-workers 3

身份为本次实际code/config Git commit，发布后记录于launch_verification及逐行source_commit；一个release archive复制核对、提取后编译和本机preflight。停止规则：Protocol/role/unknown or mismatched ancestry/config/output collision, nonfinite loss/gradient/actual-step tensors, invalid native-step ownership/Sinc/head boundary, incomplete fixed budget or failed worker stop NEW dispatch; preserve all artifacts and healthy workers. No automatic or performance-based retries. Predeclared numerical/storage fallbacks are logged scientific outcomes and block seed expansion, not legal initial clean scoring.

预期产物：初始化、actual_native_args、完整50epoch／11100step日志、Sinc／RDrop／SupCon观察、actual_feature_step_progress／steps／observation、source_frozen／source_expansion／frozen_matrix、clean predictions、独立score及配对报告。原始IQ／权重／日志／逐包预测／本机私有运行状态留本地，Git只提交实现、配置与精简结果。

独立数学检查PASS：三风险QP、相反／冗余／零约束、float32存储范数、共同物理单位缩放、实际sum／128类别贡献及隐藏标签负例。真实F CPU与CUDA路径PASS，原两次前向／标量／RNG保持；CPU主梯度逐位相同，CUDA原F自身反向约1e-7漂移已独立复现，对非head梯度仅设2e-7绝对比较容差。相同实际梯度重放下，原生位移、全部头参数与所有AdamW moments逐位一致；投影与对照实际风险明确分开，Sinc后观察看见最终状态，导出160保持。

这些检查包含每设备4个丢弃的合成原生optimizer步（默认头正常联合），没有真实IQ、正式模型训练、目标读取或单独头拟合。一个匹配真实checkpoint no-query smoke通过且丢弃。源扩seed正／负例通过，数值回退不得晋级。唯一独立P0／P1审查PASS，无下一真实run直接阻断项；这些均不是网络性能结果。

本实验借鉴[Gradient Episodic Memory](https://arxiv.org/abs/1706.08840)的风险不等式思路，当前类组、完整实际参数步及存储范数控制属于具体适配，不声称完整复现、首创或保证有效。REJECTED_EXTRA_GATE：通用设计审批暂停、全项目测试和报告再审不属于最小白名单；已有用户授权，采用聚焦验证与一次独立P0／P1审查。

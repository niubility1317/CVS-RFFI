# 固定 SIG_TIME：真实骨干梯度方差训练预登记

状态 RUNNING（独立核实三新模型已完成首轮）。实现 9c981766a0727cf072755ddd46cd5b639bae89ce，分支 codex/phase1-feature-challenge-20261013。本报告不包含新目标结果，也不宣称方法有效。唯一独立 P0/P1 审查修复后 PASS；无额外设计审批、seal、成员hash或重复数据验证门槛。

## 研究问题与既有证据

现有 SIG_TIME 和普通默认头保持。既有真实 L180 梯度诊断在历史基线上测得 t1/t2 的 RX 归一化方差差异 raw2.455803/1.938690，去TX/day共同均值后1.714985/1.373195。这不是当前SIG的有效性或因果RX证据，需在本轮固定SIG实际训练中检验。目标分数不用于选择系数、父权重、轮次或种子。

Fishr 的一般思路是匹配域间梯度方差，见[原论文](https://proceedings.mlr.press/v162/rame22a.html)与[作者实现](https://github.com/alexrame/fishr/blob/main/domainbed/algorithms.py)。本轮是明确的项目改造：测量真实 t1/t2 pointwise 参数导数，不是 logits 或额外线性分类器的代理；增加归一化与TX/day去均值对照，不使用方差EMA或优化器重置，不称原论文逐项复现或套用其理论保证。

## 固定矩阵和方法

Run 20261014-phase1-backbone-fishr-oldgate-cos-e50-clean-manysig-s01-m4-r01，seed2026092701，split392005。四行完整配置仅维护于experiment.json：历史旧门控+cos复用，不重训；三新臂SIG_TIME ZERO、BONE_RAW_FISHR、BONE_COND_FISHR从零各50轮。网络均SIG112→joint160/id_feat_joint，fθ195041，既有cosine6×160 scale30正常联合训练。没有新增模块、另训分类头、RX/day诊断头或替代读出。PAIR、签名排序、MixStyle、原型和卫星增强均不叠加。

保留原33label+17pseudo、11100更新、batch_neighbor门控、ownEMA.999、cos日程、λU=.16、λentropy=.01、原label_smoothing=.01。源角色L6300/U56700隐藏TX/V27000，训练池标注率10%，源RX1/3/4/6/8、days1/2/3、全部6TX与physicalIDs不变。NoExplicitCFO保持，内部逐包补偿保留。V仅原健康验证，无额外梯度、EMA或拟合。数据复用既有合法source handles，不新建hash链。

真实L按原固定shuffle2026210712的70个L90合成35个L180 cohort，每TX/day/RX恰好2包，每RX36包；物理ID不重复，35cohort覆盖全部L6300。每轮batch16/32/.../208共13次额外eval前向保留导数，所有三新臂均650次/117000额外源包。模式、CPU/GPU RNG与W requires_grad恢复；不更改原L/U顺序或优化器更新数。逐包前向产生真实defaultCE(.01)对t1PW3456/t2PW6912的梯度g_i，create_graph=True。

RAW用g_i；CENTERED用g_i减去同TX/day在全部5RX、10包的平均梯度。对每个RX用n−1=35求逐坐标方差v_r；每层惩罚是 mean_RX,coordinate((v_r−mean_RX(v))²) / mean_coordinate(mean_RX(v)²)，分母可微且floor1e-30；两层取固定均值。系数固定.01，5轮预热、10轮ramp，不在线按梯度范数配比或按目标调权。该归一化防止直接用单位尺度比较，但不保证域因果或性能收益。

默认W仅在额外辅助图中作为常量，不接收正则梯度；正常原L/U默认头学习保持。正则数值仍使用当前默认W参与CE，因此不声称W无关的训练目标。部署特征仍独立于W，训练后实际检查导出0差、W置零/删除独立与fθ数值更新。

## 真实观测、成本和科学资格

每新模型epoch1/17/33/50的全部13个额外批保存实际逐包t1/t2梯度矩阵，并保存mainLCE、.01RAW、.01CENTERED与effective对t1/t2/SIG投影的导数。52cohort、104矩阵、156层cohort记录/624向量，每模型逐包矩阵原始float32约0.39GB。原始PT/日志/权重/预测留远端，不提交Git。训练梯度观察和二阶导数有额外时间/显存成本，不承诺加速。

本轮三新共150model-epochs/33300更新/1950extraL180/351000额外包；历史50/11100另列。每模型≤50，固定最终50轮不冒充已证明收敛。记录完整源曲线、U覆盖、原始/条件化方差、实际梯度方向与耗时，分析不把源几何改善或梯度更强当泛化代理。

首种子科学资格只读source：所有新臂终轮LCE≤.5，预算/物理布局/导出/W独立完整；每候选实际L720z差≥1e-6；t1/t2两个实际对象在17/33/50每期至少一次非零effective梯度。第三SIG投影仅诊断，不阻塞。两个候选均PASS后，原样扩固定02/03，不根据目标排名、删候选或重跑。source资格只验接入与健康，不证明DG。首种子不替代三种子84%目标。

## 执行和自动收尾

环境为N607既有普通账户CVS-RFFI，fullFP32、CPU2threads，CWD PROJECT/releases/cvs_backbone_fishr50_20261014_r01。唯一owner codex/root/backbone-fishr50-20261014。共享锁按全部owner每GPU最多3实验进程、启动前空闲≥12000MiB，实际GPU启动后记录；不触碰健康他人任务。

本地发布入口：项目原生Python运行 experiments.cvs_backbone_fishr50.publish --output local_artifacts/backbone_fishr50_delivery_r01。远端入口 python -u -m experiments.cvs_backbone_fishr50.dispatch。代码commit上列；执行commit在本报告提交后由不可变release保存。远端run PROJECT/runs/20261014-phase1-backbone-fishr-oldgate-cos-e50-clean-manysig-s01-m4-r01，每row独占source、predictions、scores与logs/<row>.log。一个release归档、本地/远端一次SHA比较、远端编译与真实固定SIG checkpoint无query smoke通过后自动继续，不建立smoke token。smoke的固定LOCAL_ORDER首SIG_TIME ZERO seed01 final50已由既有own_model核对完整sourcecontract及final-only来源，新训练不继承该权重。

每行训练冻结后自动clean168000默认头预测，prediction固定后独立truth-last评分，共4模型672000决策/72组指标。报告逐seed配对差、Macro-F1、最差RX/day、修正/引入错误及实际特征边界；不做新Phase2、新TX或卫星声明，不把当前暴露基准称新独立盲测。

停止规则仅所属run系统技术失败：数据/query/继承协议、路径碰撞、无法执行或合法预测、非有限/跳过更新、明确预算/modeRNG/W恢复违反、错误scorer，或至少两row同pre-prediction系统指纹。先验证本run PID/CWD/cmdline/父子和输出；保留产物，无自动重试，不按低分/负收益停机，不改动其他任务。

## 本地已验证

local_checks.json：实际网络一/二阶导数、主头正常更新、auxW常量、模式/RNG/参数开关/buffers恢复、零预热主梯度一致、原U loss对象不变、归一化可微分母有限差分和尺度不变、9协议负例、发布脚本编译均PASS。唯一独立P0/P1修复了源资格误用第三诊断对象，窄回归PASS。环境标准预检已直接核对身份、项目和GPU，最初本机whoami路径错误在进入SSH前退出，后以仅子进程PATH修复并成功；无系统或远端配置修改。


## 启动后实证

执行提交 600afeeefe84e0d9f50f07c38ae0767bf1cabf07；归档一次SHA/远端编译/真实checkpoint无query二阶smoke通过后，三新模型实际在GPU4/5/0启动。各PID/CWD/argv与日志增长均VERIFIED，全部owner每GPU进程数在3以内；历史旧门控行复用并已关闭。首轮每新模型实际13个L180/2340额外包，逐包参数梯度PT存在且生产端保存重算VERIFIED，有效正则梯度为0符合5轮预热。内置logits-Fishr仍关闭，新hook的真实骨干梯度目标已实际运行。完整训练与终轮独立产物复核尚待完成，没有新目标性能结果。一个只读观察脚本最初按关闭gzip读活跃日志得到EOF，已改为读取当前完整压缩片段，训练无故障、未重启或热改。


## 第17轮实际接线证据

三新模型各13L180/2340额外源包、逐包t1/t2矩阵存在、生产端保存复算VERIFIED。RAW/COND t1与t2各13批effective梯度均非零，ZERO均0；RAW effective/mainLCE范数比均值.658707/.616297，COND .512669/.492663。独立L128与auxL180图的导数，不作为累计AdamW或DG收益。具体源统计见epoch17_source_evidence.json；终轮完整独立审计仍待。


源资格 PASS，完整首seed已闭环；固定02/03双候选原样扩展决定在回收新目标数值前提交。源证据与决定见 source_qualification.json；完整独立审计与目标分析随后执行。


## 首seed独立完整结果

ANALYZED/VERIFIED：SIG_TIME83.5333%，RAW83.7542%（+.2208个百分点），TX_DAY_CENTERED83.6548%（+.1214个百分点），n=1。完整200epoch/33300新step/672000默认决定/72指标及468层cohort、312实际逐包梯度矩阵独立重计。源决定先提交后目标读回；全部候选02/03原样扩展，未凭n1优劣调参或重排。完整结论、曲线与边界见final_analysis.md/results.json。

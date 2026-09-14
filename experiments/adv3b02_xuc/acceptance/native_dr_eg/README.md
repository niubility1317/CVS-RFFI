# 本地实施验收：原生DR＋完整EG

交付状态：PREPARED_NOT_LAUNCHED。未运行正式训练、未读取ManySig或目标truth、未启动调度器、未访问N607。测试中的模型forward/backward、一次事务和复制域头的有限步拟合仅使用本地合成输入。

## 实验配置

`configs/native_dr_eg/matrix.json`包含18个已解析的正式候选配置：R0四格×392005/392006/392007，以及DR开启背景下sim/EG两种adv=0对照×三个seed。seed沿用基线报告已发布记录，不改变数据划分。另有9个逐项R1承接模板和一个R1参考候选；后续模板的承接者必须由source V规则选择，不能从目标评分选择。

每个独立JSON同时保存`row.joint`、`resolved`和`resolved_dr`，后两者来自实际训练入口和DROT共用解析器，而非人工复制参数。所有配置`launch=false`。准备工具不会训练；训练入口默认也只解析，不创建run输出。

## 按计划开启情况

| 机制 | R0 | R1参考候选 | 验收依据 |
|---|---|---|---|
| 原生DR | 00/01关闭，10/11开启 | 开启 | 实际DROT双场；关闭时DAOT及H/P加权身份项为0 |
| 普通U底层目标 | 四格均E1起domain/self | 同R0 | 同U256序列，不回退pseudo CE/entropy |
| 每步完整EG | 01/11开启；00/10单场 | 开启 | 两次完整场，正式optimizer仅一步 |
| controller/curriculum | off/fixed | off/fixed | 入口不再由控制器覆盖固定求解器 |
| κ | 1 | .5 | 预测LR缩放及恢复与独立AdamW参考结果一致 |
| 实际归一化除数 | 原行为重估，明确对照 | 第一场实际除数复用 | L/U顺序、第二场梯度及统计一次提交测试 |
| DAOT | 两视图mean、feature=.5/logit=.2 | 两视图mean、feature=.5/logit=.1 | 已解析实例、实际教师调用[2,1]仅一轮 |
| H/P预算与基础系数 | .15、.6/.4 | 同R0 | resolved_dr、真实路由预算日志 |
| H/P身份启动 | 原生E21 | E21=0、E40=1线性升权 | 边界测试；只缩放加权身份项 |
| L身份GRL倍率 | 1 | .5 | 域头梯度不变，身份骨干该项梯度减半 |
| U身份GRL | 0 | 0 | 真实模型U ADV对身份骨干零梯度、域头非零 |
| adv=0对照 | L adv CE及U adv-head CE关闭 | 不混入R1 | 保留U z_dom CE/self，避免与GRL倍率混淆 |
| EMA | .999、接受主步一次 | 同R0 | 两场期间教师不变；提交在事务外 |
| 原生卫星曝光 | 整批Bernoulli、随机scene、x_main输入 | 同R0 | 直接复用BaselineOriginSatViewAugment |
| 卫星CE | E80起独立forward，原系数/schedule | 同R0 | 原生concat_masked语义；不扩大clean几何目标 |
| 原型 | 保留底层原型统计/原目标 | 同R0 | 不等同DAOT prototype蒸馏 |
| 新增prototype/relation/tangent/nuisance/fingerprint | 全部关闭 | 全部关闭 | 参数显式0，原型矩阵None |
| N、冻结anchor、U satellite hard、X/Ux | 全部关闭 | 全部关闭 | 配置和执行路径检查 |
| 梯度预算、动态EG、L/U统计重设计 | 按计划后续研究 | 不启用 | 不属于本轮伪装成开启的机制 |

不能以安全路由空分支判定功能未接入。受控H/P测试证明梯度可达；实际未训练合成模型不保证达到.98安全阈值。日志分别表示配置、阶段、选择和非零梯度。

## 已修正的保真问题

1. 固定完整EG不再被CORRECT动作开关覆盖。
2. 模型seed与split/order/augmentation/evaluation随机流分开。四格构造相同附加头与EMA，公共初始张量完全一致。
3. 原生卫星视图使用整批Bernoulli而非逐样本mask；scene随机选择而非轮转。clean先基础增强，再生成卫星view。
4. 原生concat_masked clean独立forward，E80起才独立sat forward。core/Fishr/几何和原型commit仍用clean。
5. 教师和medium view在原点生成并缓存在本步context，第二场复用；兼容identity_sequential/forward_identity_only。
6. fixed-divisor模式记录每次L/U normalize调用的实际除数；保留原点共享统计递推，不混入独立scale新算法。
7. κ只影响虚拟预测参数组LR，不泄漏到正式步、优化器矩或scheduler。
8. FullDROT保留历史FULL默认，但支持显式覆盖并拒绝未知字段，不再无视显式参数。

## 诊断与评分

训练入口已接入逐步loss/components、路由门控漏斗、ready原因、候选大小、风险分位数、分组通过数、预算前后质量、ESS、同U ID路由稳定性；模型seed不读取U隐藏TX。

统一身份骨干参数范围记录TX/sat/DAOT/RC4-id/ADV梯度及余弦。求解器另记参数角色的双场余弦、相对场变化、裁剪和实际相对参数更新。教师—学生KL、夹角、校准年龄以及可用梯度非零证据独立记录。

每1000步只读诊断一次：A复制在线域头、使用真实L/U域权重和当前域头LR/WD，40步评估经验优化差距；无改进记invalid/null。B在当前L批次轮换TX留出，组按TX/RX/day隔离，线性与32宽MLP各40步，报告拟合/监测。它是source小批次跨TX诊断，不声称独立采集泛化；不把A/B合成单一gap。控制器始终关闭。诊断恢复模型模式和所有随机流。

表征日志同时报告同样本跨信道距离、异TX距离、比值、有效秩与类别margin，避免将整体塌缩解释为不变性。source V每epoch输出clean Macro-F1及RX/TX/day/交叉分组；不使用目标指标选模。

`joint_metrics.py`与独立scorer提供四场景Macro-F1/accuracy、最弱RX/TX的LEO平均、最弱RX×scene、CVaR、混淆和配对Rescue/Harm。CVaR固定为等权RX×LEO场景单元风险最差20%，数量向上取整。统计以配对训练seed为单位。未实施窗口级独立bootstrap，避免伪重复。

RC4校准新增原有risk交叉拟合估计的H/P覆盖、精度、名义Wilson区间、候选大小及RX×预测TX拆分；temperature/APS与阈值仍在同一V拟合，因此明确不是完全独立审计保证，不能外推为target .98安全保证。主方法行为没有为了美化诊断而改写。

效率记录接受步、场评估、身份骨干/域骨干的学生和教师forward累计、L/U实际监督曝光、诊断与校准耗时、CUDA峰值显存和整体耗时。forward计数涵盖identity-only调用；不是只统计顶层model.forward。受控GPU小时、吞吐和等计算预算结果需要正式实验，当前不提供虚构估计。

## 本地验证与边界

完整本地验收见`acceptance.json`及`tests.xml`。命令：

```text
C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8 experiments/adv3b02_xuc/tools/accept_native_dr_eg.py
C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8 experiments/adv3b02_xuc/tools/prepare_native_dr_eg.py
```

首轮扩展测试中3个历史测试因GBK读取UTF-8失败，见`failed_encoding_attempt.json`；显式UTF-8消除命令环境问题。没有修改方法或放宽测试断言掩盖失败。

数据集、source物理角色和目标预测覆盖本次未实际读取验证，正式启动时仍核对既有source契约；不因本地单元测试通过而标记真实数据已验收。没有N607资源分配、SCP或远端代码落地，亦没有训练/预测/评分结果。后续启动需用户另行指示。本次没有创建自动化。

默认安全解析示例（不会训练）：

```text
python -X utf8 experiments/adv3b02_xuc/code/scripts/train_native_dr_eg.py --config experiments/adv3b02_xuc/configs/native_dr_eg/R0-11_s392005.json --output NOT_LAUNCHED/R0-11_s392005
```

正式prediction可复用`predict_phase1_truth_last.py`的opaque输入包及checkpoint loader；新增`score_native_joint_predictions.py`兼容该JSON schema并在独立侧连接truth、恢复分组元数据，严格要求每scene168000。当前没有调用这两条评测路径。

## 审查与科学结论

独立P0/P1审查发现的3项保真问题已修复并定点复审，没有未解决P0/P1。正式运行中的计数、完整数据吞吐、源校准实际选择率、44,400步闭合、最终性能和协同结论均仍未验证。不得将本报告的LOCAL_VERIFIED写成RUNNING/ARTIFACTS_COMPLETE或正交互证明。

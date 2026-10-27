# RXTrust-A：同包接收响应扰动下的可信 raw 身份锚

状态：RUNNING。三个seed自己的真实E20梯度校准和native checkpoint/sourceL2检查均PASS，已开始独立E20+30。普通PL、原L及默认头保持，H硬锚已产生非零记录；尚无A性能结论。A完整规则在读取PM目标分数前已固定和发布，PM全部12job结束后才接续；不依PM分数决定是否执行。

源依据是已完成的六模型三seed raw/H读出：正确类被集合包含，并不保证集合内部消歧；当错类概率较大，−log∑_S p 的直接logit步仍扩大其相对优势。该推导和源代理不是实际P/PM更新、U真值或性能因果证明。本候选保持自身接收响应输入路线，与另一聊天的跨样本特征/邻近参考路线分别验证，不发送消息或操作其任务。

原完整 PAMF、单 SIG_TIME、默认头联合训练、fullT1 L、普通 rawPL 阈值0.95/权重1、E20+10+20至E50/11100、13L37LU全部保持。训练只有原clean IQ及同包星地H、已知RX及其有界RX均值差作用，不用DURA/恢复IQ/EMA/新分类头。物理强度0.1、±0.5dB/±0.025rad、四源RX逐包最大同base KL选择、15raw/15H日程、额外64L仅detach监测均不变。

唯一目标变化是H阶段raw已被原PL接受的样本：原PM用raw/H判决并集，A直接对同包选中RX风格H计算rawargmax硬CE（无LS）。raw拒绝者继续原同H softKL，raw阶段仍全U同base softKL。全部U作分母，不按接受池重新放大；原rawPL损失和原L损失不变。使用detach当前eval raw预测，无UTX真值或新教师。

系数算法与PM一致：各seed在自己的真实原E20上测12个固定训练L/U批（6raw6H），0optimizer、参数/buffer保持，alpha=min(1,0.1/max_base median(norm_aux/norm_primary))，444步ramp。由于损失变了，必须重新测梯度，不能复用P/PM旧梯度；预计校准33792 F包/model，与训练成本分列。它只约束冻结源中位参考比例，非每步、未来epoch或AdamW硬上限。

首真实E20/full863730参数/单SIG/defaulthead与U-only梯度检查PASS后自动校准并训练。每seed从原自身合法scratchE20独立+30，非PM E50续训。每模型U56700各30遍、raw/H各850500，H硬锚+fallback合计850500、每RX170100；legacy P_H_pair恒0，P_H_singleton即所有raw接受的H硬锚。训练F包18724860/model与PM相同，推理仍原始IQ单F，无新增参数/视图。

全部新GPU任务等待整个PM complete.json，纯资源依赖不读取其分数。随后own≤4、每卡own≤1、总CUDA+pending≤2/free10000MiB、总CPUscorer≤1，其他owner健康任务保持。首次本run技术/非有限/输入权限/独占输出错误停止新派发，健康子进程结束、原输出保留；科学负结果不取消seed。

逐行finalE50后自动原sourceV raw+3H诊断、rawclean+六practical、cross_rx126000+cross_both42000。每行全部14组预测固定后独立truth-last，3×168000×7=3528000正式判决。完整逐seed、三seed均值±样本SD、同seed PL/Sup/PM配对、scene/TX/RX/day、macroF1和实际成本全部报告。源诊断、F0矩阵、单景或单seed改善均不能替代原rawclean≥85.98174603174603%、六景>72.0961970899471%与所有声明范围严格优于PL/Sup；原完整目标active。精确命令/路径/CWD/数据契约/父源及停止规则见experiment.json。

本地验证：7项硬锚/拒绝KL/全U分母/正确logit纠错方向/原PL与段间父源/三seed队列隔离/模式校准检查及3项完整结果加权/错参考缺seed/未完成矩阵拒绝检查全部PASS；5模块编译PASS，本候选唯一一次独立P0/P1审查PASS。这段是发布前节点；实际A E20校准和checkpoint smoke现已逐seedPASS，见下述运行证据。结果/完整日志工具已就绪，训练与测试连续执行，健康PM不变。

实际发布与训练已VERIFIED：执行commit `1402e4898872f23f3b82e114e9e253614a354474` 推送及独立上游OID对照完成；单archive传输校验VERIFIED，remote compile PASS。CPU控制器及三条CUDA轨迹的PID/CWD/argv/UID/实际显卡和日志增长均核实。自己真实E20 source-gradient检查（SIG/defaulthead非零）与首native checkpoint roundtrip/sourceL2/full863730/6类有限输出均PASS。原PM/其他owner未改动，source缓存与RXbank复用，原有控制不重训/重测。

| seed | 实际系数 | raw源中位比例（%） | H源中位比例（%） | 快照新增U更新 | 快照H硬锚/双类/fallback |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026092701 | 0.092806029960 | 0.488452 | 10.000000 | 336 | 26739/0/2445 |
| 2026092702 | 0.100504626136 | 0.435207 | 10.000000 | 322 | 23600/0/2000 |
| 2026092703 | 0.089225469121 | 0.598630 | 10.000000 | 334 | 26386/0/2286 |

以上是实际训练初期快照，非完整曝光或性能结论。三份校准各12新源批/33792计账F包，0optimizer/V/target，参数与buffer保持；系数符合预登记max_rawH_median规则，两种模式中位参考≤10%。H双类字段恒0，拒绝者仍softKL，原PL门控与ramp/尾批计数逐步通过；真实allU/各RX/逐scene完整性仍待E50全部日志核验。见launch_verification.json。

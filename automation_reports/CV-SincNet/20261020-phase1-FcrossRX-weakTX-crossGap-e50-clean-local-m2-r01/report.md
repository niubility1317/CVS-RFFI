# F＋跨RX同TX：两组样本共同支持的弱TX类差

状态：ANALYZED_NOT_ADOPTED。两组首seed已完成从零50轮、clean168000预测及独立CPU truth-last评分，完成审计VERIFIED。未通过源扩展规则，不新增02/03；保留原F＋跨RX同TX基底，弱TX改善目标仍未实现。CPU与CUDA真实F检查通过，一次匹配真实checkpoint CPU no-query smoke通过且丢弃；唯一独立P0/P1已PASS；随后发现的全头梯度边界P1修复已通过CPU/CUDA，该单项定向复核PASS。实际代码/配置commit为`39d7bb72a67a495912bd5ce8db68686210bfaad4`，远端已独立核实；不可覆盖发布包单次校验、发布目录编译及真实进程工作目录/GPU/原生step日志增长已核实，见launch_verification.json。

原基底固定：SIG_TIME_X3真实joint160＋统一居中单位Sinc、RDrop1、LS.01、普通增强、旧门控＋ownEMA＋cos、默认分类头正常联合。原本机三BASE源记录和clean预测只读复用，不重复训练/测试、不继承权重。历史85.1139%是方法锚点，增量与本机同seedBASE配对。新模型从零50=33L+17U，222步/轮、11100主更新；新机制无额外参数、模型前向、IQ读取或optimizer步。无星地增强，测试仅clean。

|条件|RX/day内两组类差的统计量|首seed|仅源条件通过后扩展|
|---|---|---|---|
|SELF_GAP|0.5*(||deltaA||²+||deltaB||²)|2026092701|2026092702/03|
|CROSS_GAP|deltaA dot deltaB，保留符号|2026092701|2026092702/03|

## 源证据与选择

直接复用既有三BASE完整源V缓存（每模型27000，实际joint160；原诊断run为20261019-diagnostic-FcrossRX-feature-geometry-sourceV-m3-local-r01，来源模型commit31467086c3ced30237f3ac86e91eae68645425d7）。本次CPU分析没有新模型前向、更新、U真值、target或拟合头。全部15TX对均保留，source_pair_relation.json。弱14-7/20-19的跨RX方向余弦均值.88516/.86176/.91122，低一致性排名3/1/9，不支持把纯方向对齐当作稳定独特问题。相比之下，两类的单位特征均值间距在RX3为.23535/.26386/.26196，RX8约.287至.290，RX6为.48418/.53971/.47234，三个seed均存在较小的RX3/RX8间距。

上一轮当前L一阶风险约束实际生效，但源14-7的outRXmarginq10仍下降，源20-19的增量也未达原阈值。下一假设以源证据改变训练对象：直接加强同一RX/day下两弱类的可靠特征差。固定间距0.4来自上述源几何尺度，在读取本候选任何target前选定；不根据上一轮target调权、选类、选seed或checkpoint。不是重复历史跨RX原型读出CE、逐样本RX_MARGIN、浅中层弱类尾部SupCon或条件梯度交互损失。

## 计算和边界

只用当前实际L128的原两次RDrop joint160。每个前向逐包L2规范化（eps1e-4）后，两视图在同一包内平均。源RX/day原15域由实际rx_i/day_i核对；弱TX索引固定1和3（来自旧源V校准）。每个TX×RX/day至少2个唯一物理包，按当前批次内出现顺序交替分到A/B，分别求两类均值差deltaA/deltaB。两RDrop视图不是统计独立组；A/B始终为不同物理样本。无足够cell时图内零损失。

二者统计差严格等于0.5||deltaA-deltaB||²。自身平方可以因不共享的样本波动变大；交叉项用于研究共同差异，仍不保证泛化。两个条件都取mean_validcell ReLU(0.16-statistic)/0.16，同一5轮预热、10轮ramp到权重.1，1111至11100共9990计划事件。不同统计量也改变loss/梯度幅度，不能把比较差异全部归因于去噪而忽略目标强度。有限池无放回分半不是独立IID，不能声称严格无偏或统计显著性。原CROSS_RX浅/中SupCon、RDrop、CE、U/ownEMA均保留；新增损失使用与实际joint160数值完全相同的辅助张量：捕获原前向的base_norm/pa_norm输入、参考投影及三路池输出，仅在辅助张量的LayerNorm与gain计算中detach整个cls_head参数；参考、三路池和骨干仍有梯度。增加两视图各两次LayerNorm等张量计算，无新模型前向。原student输出和默认头主联合训练不变。CPU/CUDA均核实全6组头参数辅助梯度None，主头梯度与原基底精确相同，部署特征数值精确相等。CUDA骨干VJP重算差最大3.56e-8，按FP32绝对容差1e-7、相对2e-5核对；CPU精确相同。

student/EMA/导出仍为同一F坐标和joint160；本机制不进行post-step改写，不改原物理Sinc更新公式、衰减、缓存或clip5执行次数。总梯度自然包含新特征loss。不会使用原型作为分类器，也不另训身份/RX诊断头。

## 固定决策与测试

首seed两行先完成固定final50，读取任何本候选新target汇总前，按原源规则决定是否原样扩种子：每个弱TX源outRX marginq10相对同seedBASE至少+.002；RXspread/interTXgap不超过1.05倍，零特征0；默认头整体至少BASE-.2pp、最差RX至少BASE-.5pp、任何TX至少BASE-1pp；9990计划事件、有效事件至少85%、有hinge违反事件至少10%、15个RX/day cell各至少20次。只对合格arm扩展02/03；初始失败行仍自动clean评分。源V只作设计、健康和描述性几何，没有V样本进入新损失或梯度；所有五源RX都参与训练，outRX中心诊断不是未见RX分类验证。

逐行冻结后clean168000（6TX各28000、7目标RX、4day），预测完成再独立CPU truth-last评分。报告逐seed配对差、均值、样本SD、正差数、Macro-F1、每弱类和其他类、最差RX/day、U曝光/接受率、全部gap记录、时间和CUDA峰值。n1无SD或稳定性结论，所有负结果保留。不把接受率当成伪标签正确率。不声明独立盲测、Phase2或星地性能。

## 发布与停止

环境native Windows Python3.10/PyTorch2.10+cu128，FULL_FP32_NO_TF32，threads2/workers0。CWD为IMMUTABLE_LOCAL_RELEASE/cvs_F_weakTX_crossGap50_local_20261020_r01；独占输出LOCAL_RUN/20261020-phase1-FcrossRX-weakTX-crossGap-e50-clean-local-m2-r01。物理GPU0全owner最多3个source/predict worker，派发前空闲RAM至少4.5GiB；独立CPU scorer全局最多1个。保留所有健康任务。

命令：

LOCAL_BASE_PYTHON -X utf8 -u -m experiments.cvs_f_weaktx_crossgap50.driver --profile LOCAL_PROFILE --metadata LOCAL_SOURCE_NATIVE_GATE_METADATA_R02 --base LOCAL_RUN/20261020-phase1-FcrossRX-weakTX-crossGap-e50-clean-local-m2-r01 --reference-base LOCAL_RUN/20261019-phase1-FcrossRX-TXrisk-geometry-e50-clean-local-m4-r01 --commit RELEASE_COMMIT --max-workers 3

角色/唯一物理ID/RXday/来源或配置错误、输出冲突、非有限特征/loss/梯度、原生预算错位或worker失败，停止新增派发并保留产物和健康任务；不自动重试、覆盖或依性能重跑。科学上的低有效率、活跃度或覆盖只阻止扩种子，不阻止合法初始行clean评分。预期产物包括原生参数、完整epoch/step/stdout、Sinc/RDrop/SupCon观察、gap_progress/11100记录/observation、source_frozen/source_expansion/frozen_matrix、clean预测/独立评分及同row报告。IQ、权重、逐包输出和私有进程状态只留本地，Git保存实现、配置和精简结果。

数学RED到PASS、已知噪声反例、自身与交叉统计恒等式、旋转/换标签、隐藏标签/物理ID/域负例通过；实际CPU/CUDA各2个丢弃合成主更新，原两次前向、Sinc/浅/中梯度、默认头主联合、物理clip/Sinc步、EMA eval和导出检查通过。随机稀疏合成批中SELF已满足间距，检查曾错误要求非零；已更换为明确违反间距的合成批验证反向，机制和间距不变。这些均不是真实训练或性能结果。

均值嵌入背景参考[Gretton等2012](https://jmlr.org/papers/v13/gretton12a.html)，本实验并非完整MMD检验或首次提出。REJECTED_EXTRA_GATE：通用设计审批暂停、全项目测试和报告再审不属于最小白名单；依已有用户授权，采用聚焦检查和唯一P0/P1审查后本机发布。

发布前修正：原检查仅覆盖W，发现gain及两LayerNorm直接收到新辅助梯度；正式训练未启动，已改为上述辅助常量头路径。保留首次失败和修复事实，不追溯改写历史实验。匹配真实checkpoint只做过一次CPU无查询smoke，丢弃且不继承；本修复复核不重复加载。

首个启用轮次E6真实记录：两行均222计划事件、212有效（95.4955%）、全部15个RX/day覆盖；SELF_GAP仅2个有hinge违反（0.9009%），CROSS_GAP为194个（87.3874%），见first_active_epoch.json。仅为源训练执行证据，两统计目标实际强度差异明显；不据此热改间距/权重、提前停止、扩seed或推断目标收益，固定final50及既定源扩展规则不变。

源域门控补充（只读，当前训练配方不变）：完整解析匹配本机3个BASE的33300条原生step记录，共2891700次U观察。按实际256/末批124加权，seed01/02/03平均教师置信度为98.2228%/98.2710%/98.1832%，strong一致率为99.1023%/99.1071%/99.1071%，temporal邻近一致通过率为1.6475%/1.6685%/1.6577%，最终接受率为1.5694%/1.5894%/1.5692%。最终接受数占temporal通过数为95.2582%/95.2559%/94.6617%。若保持temporal掩码不变，单独放宽其余置信度/一致门控最多再接受753/763/853次观察，分别占全部U的0.0781/0.0792/0.0885个百分点；这不是放宽temporal本身的效果或性能增益。

该日志只含全局计数，没有逐预测TX/真TX接受分布，不能据此认定弱TX伪标签饥饿，也不能推断隐藏U正确率。temporal包含同批物理邻居、预测同类和最低置信度，不是纯结构上限；教师置信度均值也不是阈值通过率。历史L代理诊断仅用于解释邻居可用性，不代替当前U精度。此结果不改变当前SELF/CROSS_GAP、源扩种子规则或检查点。已试验过的更早/更多U也不能由此重新预设有效。重现入口为experiments.cvs_f_weaktx_crossgap50.gate_context，仅读取既有源日志；无模型前向、IQ/target/U真值读取、训练或头拟合。完整逐轮计数见baseline_gate_context.json。

首个完整U轮次E34复核：每行完整解析step1..7548，前33轮U曝光0；第34轮222步、221个256包和末批124包，U曝光均56700。SELF_GAP接受883（1.5573%），CROSS_GAP接受876（1.5450%）；temporal通过929/933，strong通过56161/56143，U真值可用数0、星地活跃步0、跳过非有限更新0。只读解析没有新模型前向或训练更新，未改当前配方；接受率不代表U精度或目标性能。见first_U_epoch.json，当前两行原进程已进入第35轮且保持健康。

最终clean（各n1；同seed匹配BASE=85.1393%，不是历史85.1139%）：SELF_GAP=85.1339%，整体−0.0054个百分点，14-7−1.2000、20-19+0.6107，弱类均差−0.2946，其他四类均差+0.1393，Macro-F1−0.0025，最差RX−0.7542；CROSS_GAP=84.7702%，整体−0.3690，14-7−0.5821、20-19−1.3679，弱类均差−0.9750，其他四类均差−0.0661，Macro-F1−0.3978，最差RX−1.6708。全部正负逐TX/RX/day、配对差、成本、曝光和完整曲线见results.md/results_summary.json；n1样本SD为null，不推断稳定性或显著性。

源决定先于目标预测：两组均不扩种子。CROSS源14-7 outRXmarginq10提升0.052021、RXspread/gap比率0.810863，但20-19 margin下降0.002014、RXspread/gap比率1.251860，未共同通过；源分类健康通过也没有替代逐弱类几何条件。SELF有效率95.5055%而活跃仅45/9990（0.4505%）；CROSS同样有效9541/9990，活跃6076/9990（60.8208%）；15cell全覆盖且两组cell调用/当前L包数量相同。这是具体源证据，不将已见源RX的几何提升解释为未知RX泛化。

完整核验：每行50epoch/11100更新，原RDrop/SupCon/Sinc/ownEMA/默认头联合保留，L1420800/U963900/V2754000次观察（均含重复读取），U接受15152/15074，真值可用0，星地活跃0，非有限跳过0；两行各1976行stdout、50epoch/11100native step及11100gap记录完整解析。6个source/predict/score worker自然退出0，source扩展决定早于预测、逐row预测完成早于独立评分，现无所属worker；completion_verification.json。训练秒4283.456/4280.005，预测秒33.975/33.432，峰值CUDA均1972.051MiB。

控制器期间一次Windows进程观察子进程reader发生UTF-8解码错误；后续资源读取及所有自动派发继续成功，无停止/重启/热改训练。原控制器日志保留本地，不因该观察层瞬时错误重跑任何模型。该轮结果不采用，目标保持ACTIVE_UNMET。

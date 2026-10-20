# F＋跨RX同TX：两组样本共同支持的弱TX类差

状态：LOCAL_VERIFIED，正式训练尚未启动。CPU与CUDA真实F检查通过，一次匹配真实checkpoint CPU no-query smoke通过且丢弃；唯一独立P0/P1已PASS；随后发现的全头梯度边界P1修复已通过CPU/CUDA，该单项定向复核PASS。代码/配置身份提交后记录为实际Git commit，不要求额外SHA或批准。

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

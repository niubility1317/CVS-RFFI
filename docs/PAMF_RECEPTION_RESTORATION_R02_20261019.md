# 接收观测恢复第二轮：设计与报告核对

状态：RUNNING / VERIFIED，N607六行源训练及独立评分进行中。12项聚焦检查通过，独立P0/P1复核无阻断问题。保持原目标：单独、无 TX 标签训练的星地恢复器输出同一份 IQ，直接接现有冻结 F 与原默认头，证明身份细节属于对应输入，而不是仅降低波形误差。

依据：用户先后提供的《接收链逆问题》与《六类恢复器、两阶段训练、三层验收》报告。采用报告提出的比较顺序；其建议参数不是已证明的最佳配置。原七网络 run 与所有负结果保持。

## 已有源证据

源域可观测性诊断 run：`20261019-phase1-restoration-observability-source-s2026092701-r01`。全6,300 L_s与56,700 U_s不读TX标签；同一源V1,800条使用原六practical与原U-Net波形。不使用目标结果。

L/U的lag16/32/64/128复相干均值最高约0.14，不支持重复前导码/周期尾部补全假设。low urban原U-Net全窗NMSE0.18691，名义prefix232为0.18145、tail24为0.23961；可观测前缀也仍有较大误差，不能只归因于尾部裁剪。固定24点前移＋零补全的NMSE为1.10795，周期补全1.21485，远差于原U-Net。故本轮不增加周期复制器，也不以已知场景选择clean旁路。

24点是数值滤波延迟。分数延迟sinc旁瓣、多径与噪声影响真实可辨识性；不能直接把24个位置解释为精确的不可观测秩。最优复增益/相位计算仅作为源诊断，不进入恢复输出。

## 固定研究问题

第一阶段均为纯配对自监督；TX标签、F logits/类别、F特征损失、PL均不进入恢复训练。已有F仍属于监督预训练模型，整体系统不冒称全无监督。模型推理仅接收当前IQ；采样率25MHz、post_sync与noEQ属于固定已知处理配置，不输入真实信道、锁定状态、TX或测试RX编号。

|行|网络|训练观测比例|目的|
|---|---|---|---|
|T1|同一残差U-Net|15%clean＋85%residual/post_sync/noEQ|当前链控制|
|T2|同一残差U-Net|15%clean＋85%full/post_sync/noEQ|完整链控制|
|T3|同一残差U-Net|15%clean＋60%residual＋25%full|同曝光混合增量|
|T4|复数时频U-Net|同T3|镜像与时频处理增量|
|T5|多尺度条件扩散|同T3|持续观测条件的v目标与完整采样|
|T6|多尺度条件流匹配|同T3|从received附近开始的路径|

每行从零E20，每轮63,000物理样本各一次、batch256/247更新，合计4,940更新、1,260,000次曝光，L/U分别126,000/1,134,000。每epoch独立信道随机流；所有行共享同物理样本、同epoch量化抽样与初始物理状态，混合数据替换既有配额，不额外增加更新。full与residual从同一个clean分别生成，不连续叠加；初始latent seed、RX参数与state_start必须相同。其质量估计/时间演变仍允许存在实现差异。

优化设置：AdamW、LR2e-4、前247更新warmup、cosine到2e-5、weight decay1e-4、全局梯度裁剪1、FP32；记录原/加权loss、裁剪率、LR、route/scene/L/U、模拟锁定/状态/新增信噪比分组与物理曝光。生成模型另记内部时间四分位，与物理退化独立。暂不EMA。配对目标为原坐标IQ NMSE＋0.1平均绝对log(power+1e-3)差；扩散/流为自身v/velocity MSE＋0.2配对目标。正式指标仅来自完整采样，训练clean-dependent估计明确标为代理。

两种源选择策略在目标预测前一并固定：A仅输入clean与六residual原坐标NMSE，clean NMSE≤0.02后按六场景平均NMSE最小选择；B明确使用有标签source V身份诊断，在low urban至少+0.5个百分点、clean下降不超过0.2个百分点、clean NMSE≤0.02且平均NMSE不恶化的候选中选AsF low urban最高。B是有标签选模，不称为全流程无监督，恢复器训练仍无TX标签。最多测试A/B去重后的两个候选；B无人合格时只保留A的一次负结果确认，不据目标分数追加。目标复用父run固定1200条输入字节，属于阶段性配对诊断；正式完整目标和独立族证明仍待源资格成立后推进。

先在源域固定候选与采样规则。正式预测不依赖clean构造起点，clean也通过同一个恢复器。主指标保留原AsF frozen-head low urban、clean、真实TX净纠错；波形选模与标签身份诊断分别披露。目标分数不用于新设计、调参或重排。最终low urban>60%仍需固定正式received上的独立验证，当前源诊断或1,200条配对面板不能冒充最终完整证明。

## 报告要求追踪

|ID|要求|实现/产物|状态|验证与边界|
|---|---|---|---|---|
|R02-01|全L/U无TX配对恢复|原IQSource＋新data.py|verified|20epoch缓存与前三行真实曝光完成；0 TX读取|
|R02-02|full/residual共享latent且不串接|data.shared_pair|verified|原生引擎同seed/RX/state；跨epoch变化测试|
|R02-03|同U-Net三种route、同曝光|六行矩阵与route_array/runner|verified|实际训练函数使用同一向量路由；曝光与LR窄测试通过|
|R02-04|深度适度的U-Net、线性IQ、零输出初始化|models.ReceptionUNet|verified|256→128→64→32；初始严格恒等、幅值不剪裁|
|R02-05|复时频、正确-k共轭镜像、有界增益|models.ComplexTFUNet|verified|FFT(x*)=conj(FFT(x)[-k])测试；幅度gain1/4至4|
|R02-06|单恢复IQ供全部原PAMF分支|原views/frozen接口＋checkpoint_loader|verified|实际七视图接线与前三行真实冻结F读出通过|
|R02-07|扩散多尺度条件、每步条件、复用编码|models.MultiscaleDiffusion|verified|完整16步；条件编码每次推理一次；无状态更新|
|R02-08|流匹配目标和推理起点一致|models.MultiscaleFlow|verified|sigma0.02/16Euler；两生成模型条件编码梯度测试通过|
|R02-09|新信道流、route/scene元数据不进模型|新cache与runner|verified|20个epoch原生缓存完成；模拟器truth/route不输入R|
|R02-10|基本两类恢复损失、warmup和实际预算|新runner/design|verified|固定IQ＋log-power；端点LR与有限梯度测试通过|
|R02-11|shuffle条件与完整采样分开诊断|新source预测诊断|pending|固定source V置换，禁止目标选择|
|R02-12|source资格/选模冻结→逐行预测→truth-last|新controller，复用独立scorer|implemented|无TX选择字段与错路线checkpoint负向测试；原truth路径显式传入|
|R02-13|clean＋六practical＋full交叉源诊断|新source数据与评分|pending|同状态两实现是相关观测，不冒充模型seed|
|R02-14|batch1延迟、显存、调用数|新预测诊断|pending|批量吞吐和单样本延迟分开|
|R02-15|物理展开需先验证有效算子|未启动|deferred|原49tap圆周FIR只是近似，不能据此宣称真实接收链反演|
|R02-16|粗恢复＋生成残差与确定性二次修正控制|未启动|deferred|先有合格粗恢复器；残差不命名硬件指纹|
|R02-17|E14分叉有限L身份监督，默认头冻结|未启动|deferred|仅后续有源证据时单独登记；明确其非纯自监督|
|R02-18|PAMF联合微调与同预算直接As控制|未启动|deferred|当前先固定原F，不把联合编码冒称忠实恢复|
|R02-19|full/pre_sync粗同步独立入口|未启动|deferred|不让post_sync无差别强同步，不阻塞当前主任务|
|R02-20|未训练接收设置或独立信道族验证|后续冻结候选验证|pending|不把换参数冒充独立物理来源；不宣称真实在轨|
|R02-21|低质量身份保持、随机性、多seed最终证明|最终验收|pending|固定单次主采样；禁止真值best-of-N；未通过则目标保持active|

本文件是工作设计与追踪；实时训练与效果证据见对应run报告，不以本表代替评分证据。发布前以最小experiment.json固定commit、命令、输出root、GPU、停止规则和预期产物；只执行项目最小流程，不增加设计审批、签名链或重复审查。

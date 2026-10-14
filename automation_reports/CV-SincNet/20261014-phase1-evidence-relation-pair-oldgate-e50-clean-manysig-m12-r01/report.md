# 跨 RX 关系自蒸馏与 CVS 增量证据响应

本轮研究“给无标签样本提供什么表征监督”。全部合法 U 参与连续关系学习，既有 PAIR 门控伪标签 CE 和全局 U entropy 原样保留。新增目标不产生六类标签，不经 W，不新增分类器、投影器或部署参照查询。实际机制、矩阵、环境/CWD/命令/路径/stop rule 详见 experiment.json。

## 固定目标

180 个合法 L 个体参照由 ownEMA 刷新：每 TX/RX/day 2 包，每 222 U 更新刷新一次，共 17 次、3060 额外 L 前向。标签用于合法 L 均衡采样和原主 CE，U 隐藏 TX 始终不读。bank 与教师输出 stopgrad，允许最大 221 更新的版本间隔并记录年龄。

每 U 比较 120 个参照，各 TX 20 包，各 TX 的 day 构成为 7/7/6。B 使用全 5 RX，每 RX24；C/D 排除自身 RX，使用另4 RX，每RX30。每个参照 RX 分别 softmax，再平均损失，不把 RX 的目标机械拉成相同向量，不按 TX 聚合为标签。

教师/学生关系分别为 T=softmax(normalize(z_T)·normalize(a_T)/0.2)、S=softmax(normalize(z_S)·normalize(a_T)/0.2)。L_rel 为逐 U、逐 RX 的 KL(T||S)。学生使用原强噪声0.015与Dropout，教师原clean弱视图。温度与系数均在新源结果前固定，不锐化目标后再以其确定性自证。

真实 CVS 增量 z=a+b_PA+c_ref。对 PA/ref 分别用 alpha=0.5 得到 z-alpha·b；教师和学生均削弱同项证据。响应 D=(R(z)-R(z-alpha·b))/alpha 是有符号关系变化，不要求缺证据时仍输出完整答案。L_resp 使用 SmoothL1(beta=0.02)，sum参照坐标、mean RX/证据路径/U。B/C固定lambdaR=1；D另加lambdaE=1；A新增目标为0。四组参照刷新和计算匹配。

新增辅助重组的 z 与部署joint160数值一致，只对 PA/ref 融合增益 stopgrad；增量内部特征不断梯度。原CE仍正常联合训练 W、增益与骨干。只减可分离的 PA/ref实际增量，不把t_proj当作最终可加项。网络证据削弱不等于移除真实PA硬件效应，跨RX参照也不保证RX信息已消除。

## 矩阵与科学推进

| 组 | 新目标 | 固定问题 |
|---|---|---|
| A：ZERO | 0；相同参照/计算 | PAIR原配方匹配起点 |
| B：ALL_REL | 全源RX L_rel | 通用关系监督是否有用 |
| C：CROSS_REL | 排除自身RX L_rel | 跨RX组织是否提供增量 |
| D：CROSS_RESP | C＋signed L_resp | CVS增量证据响应是否提供增量 |

三个预定义modelseed2026092701/02/03。每row scratch50，33L+17U、11100mainupdates、3774辅助调用、963900U包观察；默认cosine头不变、无另训头。完整12模型600modelepoch/133200更新，无历史checkpoint训练继承。已有50epoch模型只作真实checkpoint smoke/sourceL诊断，不能视为新方法父权重。

先seed01 A/B/C完成固定源训练，D等待其源科学标准：finalLCE<=.5、相对A源V均值下降<=1pp、最差RX下降<=2pp，B/C非零关系比例>=.5、四预定实际aux/L特征梯度比均值>=1e-5。D完成同标准后才扩02/03。所有资格只读源信息并保存，目标评分不进选择。科学失败保留已完成行/缺项并停止pending；不凑完整矩阵，不调参补过。

## 验证、诊断和结果边界

核心负测PASS：同TX/day参照数量匹配、排除自身RX、教师/学生完全一致时损失零；完整joint相同而PA增量相反改变的代数反例，其relation项零但response项非零。真实native CPU检查 W/两增益的辅助梯度None，t1与参考投影梯度非零，实际特征参数变化；这是接线证据，不是泛化收益。一次独立P0/P1审查PASS，无阻断缺陷。remote首步严格加载实际同契约student/ownEMA，零IQ noquery smoke；用合法L180参照＋不重叠L720探针记录语义/信号诊断，不拟合参数或阈值。

训练在四预定更新7327/7548/8880/11100记录实际L/UCE/entropy/aux/R/E fθ梯度、aux-L夹角、t1/t2/t3/PA卷积/分支投影分组，原clipping和真实AdamW步后的参数变化。另记关系熵/样本关系差异、参照方差、PA/ref增量范数、损失median/max/top1%贡献，防止以覆盖率、标量loss比或末端变化代替骨干更新证据。发现塌缩不自动叠加其他正则。

每row final50固定后直接clean168000默认头预测，再fresh CPU独立truth-last评分；完整矩阵报告3seed均值/SD/配对差、各RX/day/TX和最差RX/day，负结果保留。B有效不代表跨RX成立；C有效不代表响应有效；只有D稳定优于C才支持CVS响应解释。源已见RX诊断不称未见RX泛化。本项目基准已暴露，不称新的独立盲测，不预告84%。

唯一owner codex/root/evidence-relation-20261014；普通N607账户，max3总进程/GPU含otherowners，>=12000MB空闲，不停健康旧任务。源码身份以release_commit.txt的Gitcommit为准，archive只核对一次运输SHA，remote compile和smoke通过后继续。错角色/继承/输出冲突/nonfinite/实际重组不一致/默认预测或评分技术失败停止pending，保留active和产物，无自动重试。

## 依据与创新边界

[Relational Knowledge Distillation](https://arxiv.org/abs/1904.05068) 研究样本关系传递；[ReSSL](https://arxiv.org/abs/2107.09282) 使用实例相似关系及弱增强动量教师；[PAWS](https://arxiv.org/abs/2104.13963) 使用有标签参照非参数构造类监督。本轮关系目标保留个体参照分布，不按TX求和形成类目标。关系蒸馏和参照本身不是新发明；待检验贡献是匹配跨RX组织＋真实CVS增量的有符号响应监督。[SimSiam](https://arxiv.org/abs/2011.10566) 对Siamese结构退化的讨论也不能作为本任务EMA必然不塌缩的保证。

用户最新材料是35f4dc861分支的未执行研究建议。这里按本工作树真实ResidualPhysicalHead/ReferenceCVS与默认头实现，不把材料的预期效果写成已验证事实。此前切向路线独立原样确认剩余种子；可信纠错完整源诊断41/41正确但量不足固定门槛，保留结果，不继续搜索接受规则。

当前状态：RUNNING / VERIFIED；真实checkpoint/sourceL smoke已通过，首seed A/B/C训练健康运行，尚无本机制完整泛化结果。


## 独立发布回读

执行Gitcommit `4e5bd3275289a91cb407392461eeefe20d171820`已push且remoteOID一致；归档运输一次SHA比对、remote compile、真实checkpoint noquery smoke PASS，PID/CWD/argv/GPU/log独立回读VERIFIED，GPU最大总实验进程3含其他owner，未停止其他任务。当前row状态：ZERO-s2026092701 epoch4, ALL_REL-s2026092701 epoch3, CROSS_REL-s2026092701 epoch4。此为局部源训练启动证据，headfit0，不代表识别提升或84%。详见launch_evidence.json。

固定真实源L诊断：180参照＋720不重叠探针，新增关系非零比例100.00%；实际t1辅助梯度范数0.11520934104919434，W/PAgain/refgain辅助梯度均None。各源RX同TX关系质量：1=0.7474, 3=0.6566, 4=0.7057, 6=0.7178, 8=0.7226（基准chance1/6；骨干已见源RX，不是未知RX泛化证据）。这些只是固定源学习信号诊断，没有阈值/模型拟合。

## 实际初始化与标签阶段匹配

已直接读取本次firstseed A/B/C各自initial_smoke.pt：全部191个state tensor逐项相同，原生native args除row身份/输出路径外一致，不只依赖本地构造测试。完整可用step prefix比较至4873步，仍在U启用7327前；新增关系/响应尚未执行，ownEMA按原流程维护。

不宣称GPU训练轨迹逐位相同：B/C最早CE差异分别step2/4，初始差为2.3842e-7至4.7684e-7；首100步最大差0.0022104/0.0011998，后续可用prefix最大差0.1211541/0.1021382。硬件同为RTX3090，实际backend tf32均关闭、benchmark=false、deterministic=false、precision=highest。该现象与浮点非确定性及轨迹累积相符，但这些观察不能证明唯一原因或随机流逐位相同。保持冻结运行，不热改determinism或重复训练。最终用固定3modelseed与matchedcontrols报告实际差和SD。

证据只覆盖当前完整可用源日志prefix，不能替代全部50轮/11100步分析；当前无新机制默认头识别结果。详见initial_matching_evidence.json。

## 首种子 A/B/C 完整结果

三行已分别完成50epoch/11100更新、默认头clean168000预测和freshCPU truth-last评分，CM独立重算一致。全部源日志11100step、3774辅助记录和完整stdout已扫描，无nonfinite跳步、fatal错误或U真值可用。D已按预先源规则自动放行；该规则在读取本次target汇总前完成，不因下列结果修改健康D。

| 条件 | clean accuracy | Macro-F1 | 最差RX | 最差day | final源LCE |
|---|---:|---:|---:|---:|---:|
| ZERO | 81.0488% | 0.805569 | 71.2000% | 76.4357% | 0.285831 |
| ALL_REL | 79.3310% | 0.787608 | 69.1625% | 73.5548% | 0.302233 |
| CROSS_REL | 79.5982% | 0.790459 | 69.7792% | 73.7524% | 0.303324 |

ALL_REL−ZERO＝−1.7179pp；CROSS_REL−ZERO＝−1.4506pp；CROSS_REL−ALL_REL＝＋0.2673pp。这个固定首种子中，关系损失确实执行，但未转化为默认头收益。跨RX组织的小正差不能掩盖其对匹配ZERO的负差。当前只有一个modelseed且缺D，不称完整3seed结论，也不据此改温度/系数、停止D、调参或选择性重跑。

固定源判据通过：三行finalLCE≤.5，B/C相对ZERO源V均值/最差RX下降分别约0.1407/0.4259pp、0.2296/0.8333pp，均在原1/2pp健康界内。B/C四预定aux/L梯度比分别 93.92%, 37.31%, 22.53%, 61.61% 和 74.71%, 34.89%, 20.40%, 49.50%，平均约53.84%和44.88%，远大于固定1e−5学习信号下限。

新关系已进入实际t1/t2/t3与PA/参考投影，W和两融合增益辅助梯度均None；四次真实AdamW更新与原clip记录齐全。ZERO辅助loss/梯度始终0。B/C所有3774次可见关系非零比例1，963900 U观察均有关系目标，17次180包L参照刷新共3060前向。bank_age最大221，与固定刷新安排一致。

关系teacher熵均值从ZERO0.6836变为B0.6182、C0.6461，bank归一化方差从0.6688变为0.7796/0.7792；目标更锐和特征变化不是收益证明。top ceil(1%)包的关系loss贡献均值约6.34%/8.52%/8.67%，最大约10.91%/20.07%/20.42%，未由极少数包垄断。部分aux-L夹角为负，不能仅凭梯度存在宣称优化有效。

C中response只观察、不加入更新；其原始梯度可到PA卷积，但只有D实际加入后才能检验响应贡献。当前不追加白化/Fishr/MixStyle或根据target反调系数。完整源曲线与所有四点梯度保存在local_artifacts/full_source_prefix_analysis.json，公共紧凑证据见first_ABC_evidence.json；后续D与剩余seed按原冻结计划继续。

## 首种子四条件闭合与扩种子

D已完成50epoch/11100更新及默认头clean168000预测、freshCPU truth-last评分，四行共672000固定决定、72CM独立重算一致。首seed结果：ZERO81.0488%、ALL_REL79.3310%、CROSS_REL79.5982%、CROSS_RESP79.7000%。D−C为＋0.1018pp，D−ZERO为−1.3488pp。D最差RX69.7542%比C69.7792%稍低，最差day74.1190%比C73.7524%高；不得只写小正增量而省略基线负差。

这是一个固定modelseed，当前不能支持有效机制、稳定响应收益或84%。四行源健康/非零信号/真实gradient标准在读取本target汇总前通过并保存，dispatcher按原规则已启动seed02/03全部8行。后续参数、候选及预算不依据这份target结果改变，不停止健康实验、不选择性重跑。实际PID/CWD/argv/GPU/log独立回读见first_quad_evidence.json；仍缺完整3seed结果。

## 完整三种子结果与执行闭环

**ANALYZED / VERIFIED**：12 个新 scratch50 模型全部训练、冻结、默认头 clean 预测和独立 truth-last 评分完成，自然退出，无缺 row。216 个混淆矩阵、2016000 个固定决定以及均值/SD/配对差由独立读取重新核算。

| 条件 | clean 均值 | SD（百分点） | Macro-F1 | 逐seed最差RX均值 | 逐seed最差day均值 |
|---|---:|---:|---:|---:|---:|
| ZERO | 80.6899% | 2.4977 | 0.801647 | 68.1625% | 76.5341% |
| ALL_REL | 80.1006% | 1.7798 | 0.795593 | 68.9264% | 75.1897% |
| CROSS_REL | 80.0782% | 1.9138 | 0.795548 | 68.2847% | 75.2667% |
| CROSS_RESP | 80.1050% | 1.5494 | 0.795832 | 69.2347% | 75.3849% |

| 配对 | 均值差（百分点） | seed01/02/03（百分点） | 正差种子数 |
|---|---:|---|---:|
| ALL_REL−ZERO | -0.5893 | -1.7179 / +0.8030 / -0.8530 | 1/3 |
| CROSS_REL−ALL_REL | -0.0224 | +0.2673 / -0.3851 / +0.0506 | 2/3 |
| CROSS_RESP−CROSS_REL | +0.0268 | +0.1018 / +0.3482 / -0.3696 | 2/3 |
| CROSS_RESP−ZERO | -0.5849 | -1.3488 / +0.7661 / -1.1720 | 1/3 |

**原配方＋所有合法 U 的关系目标未确认收益。** ALL_REL 对 ZERO −0.5893个百分点，CROSS_REL 对 ALL_REL −0.0224个百分点；CROSS_RESP 对 CROSS_REL 仅＋0.0268个百分点，第三seed为负，不满足“响应稳定优于关系”的核心解释。CROSS_RESP 对 ZERO −0.5849个百分点，整体默认头识别没有改善。其逐seed最差RX均值比 ZERO 高1.0722个百分点、种子SD较低；这是均值识别下降下的取舍，不能写作整体更优或三个seed均稳。

固定 primary CROSS_RESP 三seed80.1050%，距84%3.8950个百分点。机制实际执行、学习信号非零与身份泛化成功分开；保留全部负结果，不据此修改本run τ/α/权重、重跑特定seed、重排parent或终止其他健康任务。

完整源记录核验600模型轮次/133200更新、45288辅助调用/11566800隐藏TX的U观察、36720额外合法L anchor前向；所有12模型每轮222完整日志，非有限loss/grad跳步0、U truth availability0、headfit0、stdout fatal0。48真实梯度点与48实际AdamW更新点齐全；ZERO auxiliary梯度为0，9个关系训练模型的t1/t2/t3与PA/ref路径实际收到非零辅助梯度，W及两个增益auxiliary梯度始终None。source曲线完整保留为source_epoch_means.csv。

首种子source-only决策在target汇总前通过，后两seed原样扩展，无目标成绩反馈。初始张量匹配与CUDA非确定性限制仍见initial_matching_evidence.json：native first33label-only轨迹未逐位相同，不能将全部paired差异断言为纯auxiliary因果效应。原种子、预算和后验缺陷均保留，不补做挑分重跑。

同包source梯度诊断已独立完成，参见源信号report：strong下同包CE/aux方向相关弱，teacher signed response不自动代表有益身份作用。该seen-L证据不是U真实正确率或目标下降唯一原因，不能强制所有分支对每包只提供正证据。

用户要求的既有结构收益继续做独立匹配组合：COMPACT_LINEAR/LONG_SILU×ZERO/CROSS_RESP已固定新矩阵，损失与配方不变；这是原source-pass后预定后续验证，不是按本次target结果改损失。Sinc居中坐标独立保留，不同时叠全部结构。当前84%研究目标仍active，当前关系矩阵已完整闭合。

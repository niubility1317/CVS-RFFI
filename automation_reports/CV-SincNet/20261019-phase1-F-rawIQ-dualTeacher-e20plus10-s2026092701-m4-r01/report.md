# raw IQ 补充特征通路：E20＋10 预登记

唯一新假设：旧局部adapter读取t1/t2已有隐层，仍未充分拟合训练见过的受损IQ；改为从当前这次前向收到的IQ直接提取补充特征，再沿用原浅中层、SIG_TIME和joint160默认头。该机制由用户adf48c方案条件授权，源诊断已完成：S3已见118/180、Ts154/180；同low已见28/59和42/59。原固定配方SOURCE_FAIL保留，不扫温度/权重或重选seed。

原F固定为SIG_TIME_X3＋统一Sinc/NoExplicitCFO。每次当前2×256IQ生成16×128和16×64表示，bias=False的k7/s2/p3与k5/s2/p2两层卷积各接GN(8,16)和GELU；两个输出1×1投影分别16→72和16→96，仅其权重初始化为0，加到原t1/t2的pw输出、原GN之前。4256个新增参数、卷积MAC356352；旧局部adapter4768参数、MAC433152。MAC不含GN/激活/残差加法，不能当作完整推理成本。

raw前向读取raw，As前向读取当前同一个As IQ，U读取其当前学生/EMA视图；不缓存跨前向clean，不读取场景标签/RX真值作部署路由，不增加分类头。模块始终参与，只有训练可用视图角色分配教师。原物理路径和原默认cosine6×160/scale30分类头保留且冻结，梯度穿过固定后续网络回到新特征模块。

四个新row为NOAS_ADAPTER（raw＋cleanKD）、S1（raw＋As CE）、S2（再加cleanKD）、S3（再加satKD）。均从同一合法B0 FINAL20独立开始，新10轮/2220更新，累计E30/6660；同seed2026092701，完整L6300/U56700/V27000、split392005与物理角色不变。AdamW重置，lr2e−4至1e−6新cosine10，wd1e−4、clip5；EMA继承对应B0原206键和相同新raw8键，局部更新计数重置。不是连续cosine30。

原用户目标第5.8节明确固定原半监督机制，本轮不修改U资格、视图或loss；保留原hiddenTX/ownEMA/batch_neighbor/strong/confidence，U标签始终隐藏。这是本chat具体匹配机制对比，其他并行对话写入的F＋跨RX E50/clean-only、PL或外部方法默认不替代本用户明确纯F星地目标；不修改其健康任务或文件。

As行CE为raw .5＋As .5；NoAs为raw1；LS.01/T1覆盖所有L。cleanKD权重.25用于NoAs/S2/S3，satKD.25仅S3；T4/T²，teacher_correct_only遮罩、分母全部128，教师错误时CE仍完整。Tc是固定cleanF FINAL20，Ts是已核实同契约CSIL fixed last200；Ts原生5cos＋5按×6转common30，教师类别顺序一致、无目标温度拟合。每row都实际前向双教师与生成As，loss为0时shadow，同计算槽位。

逐row完整计划以experiment.json/configs为准。每row新L身份前向284160（NoAs）或568320（As），U学生和自身EMA各567000，Tc/Ts各284160，As生成284160。新增训练4×10学生轮，父E20与Ts历史E200成本单列；新教师训练0。新增训练真实raw/As correct/n直接读取已有logits记录，属于训练随机子网络读出，不替代冻结拟合。

复用已完成旧双教师研究的NOAS_FT/NOAS_ADAPTER/S0/S1/S2/S3同预算source与最终7视图产物。旧FT NoAs保留ordinary输入，新的结构NoAs使用raw；同时报告两类，不用较弱对照制造clean收益。不重训、不重复NN/预测/评分。新4row与旧4结构对应row比较只有输入修正路径不同，参数与MAC差异按实披露。

源推进primary固定S3，必须同时满足：源midlow7200观测相对新S2≥.5个百分点；相对旧localS3≥.5个百分点；fullVclean27000相对新NoAs/旧NoAsFT/旧NoAsAdapter最大值≥−.2个百分点。源决定只读source资料，先于任何新目标评分；不按已见目标成绩重排候选。未通过则关闭固定raw配方，不自动E50或扩seed。

所有新row完成固定FINAL30后自动做clean＋六practical单包预测，各row168000/视图；全部7视图固定后独立CPU scorer连接truth。各row预测可立即排队，独立评分在全矩阵源决定固定之后。目标IQ/真值/旧目标成绩不参与本轮设计、训练、源选模或晋级。已暴露基准不称新的独立盲测。

N607普通账户，GPU4/6/7候选，每GPU最多一个本runworker，所有owner总数≤2、free≥12000MB，共享锁分配，CPU2、FP32/TF32off。一个新不可覆盖run/output/release，代码/config Git固定；单发布包只做一次SHA传输比较、一次远端编译和preflight。真实父CP无query smoke作为launcher首步，PASS立即继续。一次独立P0/P1审查仅直接运行缺陷。具体command/CWD/path/expectedartifacts见experiment.json；private runtime留本地不进Git。

技术停止仅限所属run的协议/角色/来源/输入/非有限值/冻结头或F状态/预算/输出冲突、不能产生合法预测等系统故障。保留partial、默认不重启、不停止健康旧任务。科学负结果不触发技术杀进程。

本地聚焦验证及唯一P0/P1独立审查已PASS，源场景轴metadata更正为实际low_suburban/mid_urban/low_urban，未改变任何IQ、数值或筛选门槛。

实际代码/config commit：`28d14761575b080077fde038cc96baedd4d982cc`。归档身份由release_commit.txt保留，可能包含后续纯登记元数据提交，实际训练代码保持。

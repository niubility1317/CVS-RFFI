# 分支通道adapter与LocalRidge联合实验实施说明

当前状态：实现与76项不同合成检查通过，已预登记PLANNED且preflight独立读回VERIFIED，尚未发布、没有实际性能结果。沿用户已授权的BranchLocalRidge优先联合微调路线推进。核心设计见[D92_JOINT_NEXT_MECHANISM_20260930.md](D92_JOINT_NEXT_MECHANISM_20260930.md)，新结构审阅见[D92_JOINT_CHANNEL_DESIGN_REVIEW_20260930.md](D92_JOINT_CHANNEL_DESIGN_REVIEW_20260930.md)。

## 联合训练与对照

固定Phase1合法来源及原五块support缓存。原b/a只构造一次，五个子块分别学习保范对角通道adapter，共736个参数、731个有效自由度。最终仍用LocalRidge统一面对全部注册类。监督信号来自合法support内折的正确类与最强错误类间隔；通过闭式LocalRidge解的伴随梯度更新adapter，不另叠加自由分类logits。

主线为R_channel_seq：B仅用旧6类support，从零训练adapter；C把B参数同时作为初始化与近端锚点，使用新旧合法support重新拟合全注册头。R0为原BranchLocalRidge；R_channel_reset使用相同C训练support、零初始化与零锚点。三条路径全部报告，不能据外层结果挑出最优路径。新增0类时直接复用B。8步固定Adam与最终状态，不早停、不选最好步、不扫参。

## 固定范围与测量

新run预登记名为`20260930-phase2-d92-joint-channel-support-m2-r01`，release为`d92_joint_channel_support_20260930_r01`。root为唯一launch owner；prepare已生成三份新配置。已有配置或输出必须保留并核对，不覆盖或重复启动。当前PLANNED不是已启动证据，实际运行以独立PID/argv/CWD读回及原记录为准。

复用已验证的160个support任务：模型seed为2026092701/2026092702、rx3/rx1两个cohort，每行两个接收机/场景组合，support seed为2026092711，旧6类、新增0/2/5/10/20类、K1/5/10/20。实际场景只含practical_high和practical_low_urban；practical_mid未测。星地信道固定practical residual、post_sync、无均衡、25MHz。

K为原任务每类support量，OOF外折实际每类训练量K5为3/4、K10为6/7、K20为13/14；内部held属于adapter训练。真正外层held只在状态固定后评分。true K1仅数值检查、不拟合、不报留出成绩；每类1样本proxy精确复用原方法，无伪造样本。逐任务与K×新增类数及模型/cohort、接收机/场景完整分层都保留。

A适应前旧类准确率本support pilot不可测，记N/A；B0为原support分类器，不能当A。报告B旧类、C旧类/C新类、H、B−B0、B−C旧类下降、逐任务新旧绝对差，并标明B−A不可测。理想指标不是每轮全满足的新增门槛。inner训练和outer留出间隔、正确率变化及预测变化分开记录，不回流更新或选参。

## 权限与计算

不读取源域样本、源域逐样本特征或query。B→C只在同一任务/外折内继承，核对旧物理ID、标签与五块原特征完全一致；不同折、模型、cohort不得继承。运行不加载新checkpoint。现有库存是用户授权的重复开发比较，不能称独立新数据确认。独立验证按用户要求延后。

全pilot最多936个实际训练阶段、7488步更新、8424次内层目标评估、25272个内层头、936个新增最终头；连同3168个R0头，最多29376个头拟合。实际反向求解另记`derivative_triangular_solve_count`，它复用Cholesky但有真实计算开销。最终目标不反向，因此最多44928次反向三角求解；实际行为由完整日志证明。退化/恒等复用按真实发生次数计数，失败尝试仍计入。

736个float64参数为5888bytes，但不是部署包大小。记录完整最终头与原b/a常驻状态、训练临时空间、Adam矩、峰值RSS/GPU、运行墙钟、训练/推理耗时和平台。新增ground数据/统计payload为0bytes；运维代码release单列字节数。没有测量的部署包/星载传输记N/A，不预先声称节省算力。

## 实施责任与验证

core负责新模块、冻结算法、合成数值测试；entry负责三路径执行与独立summary；独立审阅负责新结构直接P0/P1。root负责五个编排工具、串行ssr-gpu相关检查、正式记录、镜像、Git和唯一发布。保护并行修改及无关暂停草稿。

已通过76项不同检查：core与编排60项、入口/summary15项、新增100位Decimal近重复梯度/距离回归1项。证据依次为本机`.codex_tmp/pytest_utf8_1790755858887852000.stdout`、`pytest_utf8_1790756007768157800.stdout`和`pytest_utf8_1790756026674225700.stdout`，解释器为ssr-gpu；没有并发Conda或重跑无变化检查。范围含u0精确原方法及活跃梯度、隐式梯度独立数值对照、近重复/零带宽/规范化floor、合法投影、Adam固定预算、B→C错配拒绝、物理折隔离、完整计数及独立summary。基线标签绑定P1修复后定点读回，日志区分头拟合目标与held间隔监督目标。发布后独立读回，不凭exit status声称启动。旧谱实验已完成，不能为此重跑。

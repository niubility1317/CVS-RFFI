# 固定监督对比方案的三成员等权集成

两个既定方案 ALL_TX/CROSS_RX_TX 全部进入相同矩阵；不按单模型 target 排名选择某个方案。每个主种子组复用该方案固定 FINAL30 成员0，另外成员1/2从零各30轮，不加载成员0或其他 checkpoint训练。各组初始化为 primary、primary+10000、primary+20000；L/U采样和强噪声种子保持对应primary，dropout使用actual member seed。该矩阵在当前单模型target结果前固定，不调temp、辅助权重、epoch或融合权重。

|ID|来源|要求|目标|状态|验证|备注|
|---|---|---|---|---|---|---|
|V1|原固定三成员路线/报告训练扩展|全部两方案的三个主seed独立初始化成员，成员0来源合法|design/model/members.py|implemented|实际优化器67身份张量，29独立初始化；父源契约及成员证明入口，CPU/CUDA PASS|初始化不同，不平均不同seed参数|
|V2|用户覆盖|每新成员30轮；同普通cos，无新融合分类器；U隐藏/V无性能门槛|source/contrastive.py|verified|CPU/CUDA30轮预算/实际梯度/成员及U强前向dropout seed检查PASS|每组3倍参数和累计训练成本|
|V3|自动测试约定|等权T1平均概率、clean先固定后独立truth-last|predict/dispatch/score.py|implemented|T1等权概率及cos/refit检查PASS，正式clean重算待完成|不声称单模型或Phase2集成部署已验证|

六个集成组，各三成员；共12新模型79920更新、复用6已固定模型39960历史更新，总18成员119880更新/540模型轮。每组531075可训练参数，单成员177025，每成员160维接口与普通cos(scale30,W6x160)。集成只有固定softmax平均，零训练参数、零校准、零target拟合；不创建RX专家或路由。Phase2可直接丢弃各成员普通cos W重新拟合；本轮只证明Phase1冻结clean集成表现。

全部原source L6300/U56700隐藏/V27000，RX1/3/4/6/8、day1/2/3、7/63/30角色不变。两方案各保持既定浅/中层descriptor、temperature.2/weight.1/warm5/ramp10与相同UNIFORM_SSL sourceU配置。不按源V筛选或早停，每成员固定FINAL30。历史已完成原三成员CE集成只作同物理ID clean配对参照，其source采样与本轮不完全相同，不能把差异全归于对比损失。

每组两个新成员结束后复核三checkpoint来源，逐包独立面对全部6注册类，softmax温度固定1并等权均值；每组168000 clean预测固定后exec独立scorer，共1008000决策/108条指标。三组primary2026092701/02/03分别计算均值/SD/配对变化，完整保留两方案及单成员0对照；不得把只有一个ensemble冒充三重复或把三seed平均参数当集成。84%仍以完整三组均值为准，已暴露基准不声称新盲测。

普通账号、唯一owner、不可覆盖root。每GPU最多3个总实验进程，free>=12000MB；现行健康任务不热改或停止。仅明确runowned数据/权限/query/输出/非有限/无法闭合prediction等系统技术失败停止，不按低精度停止。一次P0/P1、一个代码/config提交、单archiveSHA/compile、真实checkpoint noquery首步及postlaunch PID/CWD/argv/GPU/log增长；精确矩阵、commit、路径、命令、环境和输出见experiment.json。成员0必须先正常冻结，未冻结时不加载、不启动该组训练。

唯一独立 P0/P1 审查发现 U 强前向仍用 primary dropout seed；已仅修正为 actual member_init_seed+step1979，强噪声保持primary+step9901。真实前向hook的CUDA检查及该问题唯一定点复核PASS，其余无P0/P1。未新增完整复审。初始化检查先冻结为真实optimizer身份参数，67张量中29随机张量不同；构造阶段183可训练标记不当作训练参数数目。member FINAL与group ENSEMBLE的冻结读出明确区分。

运行后 VERIFIED：12新成员各30轮完成，6组1008000 clean预测及独立评分均完成。最终总汇总因score.finish未导入preflight而FAILED；原dispatcher已退出，未终止其他任务。既有行级结果完整，不重新训练、预测或评分。以新run/output 20261013-phase1-crossrx-contrastive-ensemble-aggregate-manysig-m6-r02 只恢复总表，保留原错误/产物。

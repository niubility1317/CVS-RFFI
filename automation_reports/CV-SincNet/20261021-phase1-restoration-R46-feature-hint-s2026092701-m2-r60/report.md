# R60 原型：训练时保留压缩图的中间响应

状态：**VERIFIED / WAITING_R58_ACTUAL_EXIT**，已发布、未训练；唯一一次独立 P0/P1 审查通过；完整 launcher 已实现，47 项聚焦检查和 CLI help 通过，无源验证或正式性能结论。方案及核心原型在读取 R58 源验证和正式分数前固定。直接使用指定 R46 prune_refine 350521 父，不继承 R53/R55/R57/R58 训练后的学生。

只优化现有压缩网络的训练：保持原weight-only剪枝的273765参数/context185615、普通GN4、U-Net与3prior6CG，与R53 prune_decay相同数据自由通道图和初始化。没有新增推理参数、buffer、分支或更换恢复网络。新增训练时中间响应约束，教师仍为同一冻结350521恢复器，0TX标签/F权重/F输出/V/target训练。

固定两行context_hint/all_hint。两行均保留pairedclean IQ NMSE＋0.05谱误差＋线性1→0.1三阶段IQ教师，在此基础上加0.02×context特征NMSE；all_hint再加0.02×prior特征NMSE。相同context系数，比较prior额外约束，不能称两个位置平均后的系数相同。168个GN的post-affine/pre-SiLU对应保留通道全部时间位置参与；context和prior各84个模块，分别module等权。每IQ每module误差除以所选教师通道能量（下限1e-4），不跨样本计算。教师特征detach、教师无梯度；hooks仅训练期间存在，完成后移除，不写入推理checkpoint。

中间特征来自原本就执行的教师和学生前向，额外G前向和源校准输入为0；会增加训练激活、归约及内存开销，必须实测，不能称训练免费。原R07固定6300L_s＋56700U_s模拟received/clean配对和顺序不变；每行E20/4940、global256/micro256、重置AdamW，全部coarse/action/prior联合训练。原参数继承80＋新增20＝100轮，无新可学习参数；源曝光L126000/U1134000、teacher/student各1260000，原串行140pass、组件轮次和300与F50成本分别报告。

主要既有控制为预登记R53 prune_decay：相同273765图、weight-only初始化、E20/4940、IQ教师日程、global256/micro256及源曝光，差别为中间特征损失与训练开销。该旧控制只读复用，不按正式分数重排或改变父、宽度、系数。新两行训练后无论源分数如何，自动source1800×7、原冻结F50/defaulthead正式168000×7及独立truth-last，另实测同卡G/G+F的B1/B32时延、吞吐和显存。

仅GPU3/2各一条本任务worker，须等待R58同卡所属训练/预测worker完成并实际退出，总计最多两条；CPU2/共享R41 score.lock，R59身份诊断按原矩阵完成。错误祖先/数据/物理ID、输出碰撞、非有限损失或不完整前向保留产物并停止，无自动重试；不触碰健康任务。完整pipeline已接好，真实checkpoint/2条L_s noquery smoke作为launcher第一步一次执行，PASS后自动继续；完整方法commit、唯一P0/P1审查和发布登记按原最小流程完成，不增加设计审批或重复检查。

性能优先。原70%/5pp与输入TX细节保真仍未完成；中间响应贴近教师本身不是正式识别收益或逐细节TX真实性证明。有clean模拟配对的TX无标签源自监督不冒称无clean盲学习。47项检查只验证接线、梯度、训练权限、严格重载、完整源预算和实际前序进程退出保护，不将合成IQ结果当作有效性。

REJECTED_EXTRA_GATE：头脑风暴 skill 的另行设计审批、书面计划审批不属于本项目八项最小发布流程；按用户明确约定，将方法、矩阵、停止规则保留在本报告，继续执行既有授权，不新增审批门槛。

方法commit `04357684a7c069308e6c89a64d3bb51d54376fdf`。精确命令、环境/CWD、路径与矩阵见[登记](experiment.json)。


VERIFIED/WAITING_R58_ACTUAL_EXIT：归档单SHA/远端compile、真实350521父/F50/2Ls noquerysmoke通过，两scope全组件有限非零梯度、父及F不更新、训练hooks移除后独立权重严格重载一致。controller与两条CPU等待worker的实际PID/CWD/argv/CPU2/日志增长核实；均未建立GPU context、0epoch，等待R58同卡row完成并实际PID退出后才分配GPU3/2，跨run最多两条GPU worker。保持273765普通GN图和指定350521直接父，0新增推理参数/校准IQ/额外G前向，训练期开销待实测；尚无正式性能结论。每行固定E20/4940后自动source七景/成本/正式168000×7/独立truth-last，全部正负结果保留，原完整目标ACTIVE。

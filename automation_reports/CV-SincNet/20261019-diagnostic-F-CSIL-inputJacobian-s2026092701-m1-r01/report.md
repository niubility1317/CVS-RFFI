# 纯 F／CSIL：同受损 IQ 的局部身份敏感性源诊断

状态：PLANNED。只诊断合法源V，不训练、不访问目标。主研究目标仍为固定F的urban>60%、clean相对匹配NoAs可重复正收益及suburban接近clean；本诊断不能证明这些目标。

唯一问题：教师的输入局部判别方向，是否包含不同于后验类别置信、且对现有F骨干有用的源端监督内容。此前完整后验KD和排除真类的条件分布迁移均关闭，不修改它们的权重、温度或seed。当前比较的是 `d logit_y / d IQ` 的有符号单位方向，不是输出分布、全embedding或特征重构；默认头结构和部署前向不变。

[Jacobian matching原始论文](https://proceedings.mlr.press/v80/srinivas18a.html)研究输出对输入导数的信息迁移。本次只是受其启发的单个真标签行、单位方向余弦变体，不是复现原论文全Jacobian损失，也不以图像结果证明RFFI有效。梯度可算、正余弦或低辅助损失都不是识别收益。

固定读取本轮纯F普通As B1的scratch final20/4440默认头，以及已核实同一L/U/V契约的历史CSIL scratch last200/9800默认头。学生不是已关闭M，也不是按正式成绩挑出的checkpoint。两来源均固定终轮、无未知祖先或目标训练/选模。实际加载继续核对checkpoint和source角色；不继承为新训练初始化。旧成本分别保留。

数据复用已验证的V1800两draw完整residual观察及source cache。按每个TX×RX×day单元在原面板顺序取前4包，90单元共360包；每90批各含每单元1包，顺序在任何预测前固定。源RX1/3/4/6/8、day1/2/3、六TX、split392005，V只作本次预登记的机制/健康诊断，不拟合阈值、增强幅度、头或模型。U真值、目标IQ、目标truth与目标分组误差均不使用。研究依据是原合法源域教师优势与类别迁移负例，不是正式误差调参。

视图为clean360、两draw各mid urban360和low urban360，共1800次源观察。As为原缓存condition0的完整IQ，保持接收状态、补偿、AGC、噪声和边界，不生成或弱化物理信道。不改原formal数组，不追加新目标评分，因为没有新训练或新冻结状态。

每批同时计算冻结教师的真标签Jacobian行和错类 `(y+1)%6` 行。错类控制保持同一批、同一学生导数及同一teacherCorrect掩码。辅助值为 `mean_all90[teacherCorrect*(1-cos(gStudent_y,gTeacher_y))]`，对照改用错类教师行。教师导数detach；无学生置信门，源标签只用于合法V诊断。新内容是输入局部判别方向；正尺度归一化后不强迫真类概率、温度或整套非真类关系。

学生源诊断用eval函数（真实部署映射）；首步合成smoke另检验实际train随机子网络中的二阶反向。分别取该辅助值、错类辅助值、原label-smoothed CE对原参数的导数，计算默认头与其余骨干的范数/内积。Sinc q梯度除25000使用原物理Hz坐标。所有autograd.grad只返回导数，不赋值.grad、创建optimizer或更新任何参数/缓冲。合成检查已核实真实900722参数F的Sinc/signature/time/frequency/PA/defaultW二阶路径，2个合成NN/0真实IQ/0更新。

额外中心有限差分只核对局部导数：沿单位教师真类输入梯度，做相对L2=.001的正负数学扰动，并读两模型默认头。它不是身份保持物理增强，不用于训练、正式评价或真实性能声明。报告7200个此类counterfactual源包，不能写成“完全未生成新IQ”。新物理信道draw=0。

固定源rationale条件：urban教师准确率比学生至少高1pp；urban骨干 `cos(gJacTrue,gCE)` 均值≥0，且比错类控制均值至少高.01；clean对应均值≥−.05。均值只按预登记的整体urban/clean，不逐层挑阳性结果。FAIL关闭当前有符号真类行方向，不换归一化、标签、阈值或subset救援。PASS只支持考虑一次固定E20 pilot，不是性能、晋级E50或跨seed证据；无自动训练。未来若训练仍需从零累计E20及匹配B0/B1/计算控制，默认头骨干联合更新，所有合法As保留CE。

最小执行身份：一个owner `codex/root/sameIQ-sensitivity`；入口 `CVS-RFFI/python -u experiments/cvs_sameiq_sensitivity/runner.py`，CWD为本run专属release，private runtime不入Git。固定参考release为`cvs_F_nontrue_CSIL_E20_20261019_r01`，commit`1e7cc414f73f093dfbdccee0e4ba7cf90c7a5324`。代码/config本次Git commit发布时写release_commit.txt。普通账户、CPU2、FP32/TF32off；全owner每GPU最多2进程且至少12GB空闲，锁内核对并初始化自己的CUDA上下文后继续。不得停止或热改健康旧任务。

launcher首步一次两真实checkpoint的合成无query smoke，共4NN，包含实际二阶路径，PASS立即继续源诊断，不需token/receipt。一个不可覆盖run/release、一归档本远端一次比较、一远端compile、一次独立限定P0/P1审查；source/protocol/非有限/接口/输出错误保留失败证据，无自动重试。独立PID/CWD/argv/GPU/日志增长后才记VERIFIED。

预期NN=10804：student5402/teacher5402，其中source base3600、中心差分7200、真实CP synthetic smoke4。新增optimizer/head-fit/EMA更新均0；既有student20轮和teacher200轮不写作免费训练。输出resolved_config、allocation、checkpoint_smoke、data_roles、source_derivatives.npz、20批batch_geometry、source_decision、complete/failure及stdout。原始IQ/derivatives/CP/日志/私有路径不入Git，只交付源端指标及证据边界。


状态：ANALYZED / VERIFIED。完整源观测与20批几何已独立重算，固定真类导数对齐标准FAIL，关闭本配方且不启动E20。0权重/EMA/优化器更新，无目标访问。详见 [完整源诊断](final_analysis.md)。总体研发目标仍在进行。

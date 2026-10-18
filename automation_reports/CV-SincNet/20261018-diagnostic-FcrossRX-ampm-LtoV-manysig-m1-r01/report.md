# 幅度相关相位函数：完整源域单项预登记

本行只检验一个新前提，不训练模型。当前固定频域响应、IQ 椭圆性和 Tukey 输入干预已有负结果，不调整这些已关闭配方。与另一对话正在检验的径向幅度映射不同，本行仅考察幅度保持、相位随功率变化的函数关系。

对每包 RMS 后 IQ x，令 p=|x|²/mean|x|²，v[n]=x[n+1]conj(x[n])、dp[n]=p[n+1]-p[n]。定义 f_x(u)=|sum_n v[n]exp(iu dp[n])|/sum_n|v[n]|。整体相位及线性相位斜坡不改变 f；数学变换 T_b(x)=x exp(ib(p-1)) 满足 f_Tb(x)(u)=f_x(u+b)。此恒等式只为源统计模型提供可证伪预测，不证明 T_b 是真实 RX、物理硬件反演或身份保持增强。本行不对真实 IQ 生成 T_b 训练视图。

原 L6300/V27000、五源 RX[1,3,4,6,8]、day1/2/3、6TX、eq1、IQ256/25MHz、split392005。原 U56700 TX 隐藏且不使用。复用已有 source_cache 和 source_contract，核对原角色 ID；既有缓存附带的模型派生字段可被反序列化但不消费。原 RMS、center=false 保持，不加载模型/checkpoint，不构建 target。所有五源 RX 均保留。

唯一行 AMPM_TRANSLATION_L_TO_V：u 固定[-.5,0,.5]；B=min(.02,.05/max_RX Q99_L(|p-1|))，仅由 L 决定；21 点等距 beta∈[-B,B]，中心强制0。参考 RX 固定1。对18个 TX/day 统计折，每个非参考 RX 在其他5TX×其他2day的10个 L 单元上，以整组三个 u 的 SSE 最小选一个 beta。精确并列时先较小绝对值，再负值。V 使用其参考 RX 的同一函数作为预测输入，只计算误差；V 不拟合参数，不更新网络、EMA、原型、归一化或其他持久状态。统计折不是训练域留出。

比较五 RX 中心化差异的总 SSE、18 折和逐 RX SSE。控制为零 RX 差异、L-only 共享加性偏移、同 TX 其他2day的条件加性偏移；同 TX 控制用10个 L 单元/700包，共享加性用50单元/3500包，披露样本量差异。所有频率、滞后、参考 RX 和系数边界已固定。

科学判据全部满足才有资格做下一步冻结默认头身份检验：identity SSE>1e-12；shared/identity≤.8；shared/additive≤.9；至少15/18折优于 additive；每个 RX shared≤additive（1e-20数值容差）；shared/TXconditional≤1.05；72个非参考拟合均不落边界。任一失败关闭此固定幅度相关相位平移前提；不改滞后、u、边界、分位点或阈值救跑。通过也不是 Ag 识别收益，不自动启动 E30 或 target；后续模型训练仍受当前 E30/固定F/defaultcos/原U/clean份额规则约束。

预算：33300个合法源包，180个单元均值，2097900个包级函数数值，534964500个相邻项，72次21点标量统计拟合，0NN/teacher/headfit/训练更新/新真实IQ视图/target预测。不借此运行新的模型seed。保留私有派生观测供独立 CPU 聚合复算；图只展示源 SSE/RX分层/拟合系数，不编造身份准确率或训练曲线。

命令：CVS-RFFI/bin/python -u -m experiments.cvs_ampm_source48.run。CWD=/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_ampm_source_20261018_r01；输出=/home/szu2070436088/2510044040/CV-SincNet/runs/20261018-diagnostic-FcrossRX-ampm-LtoV-manysig-m1-r01；日志=/home/szu2070436088/2510044040/CV-SincNet/runs/20261018-diagnostic-FcrossRX-ampm-LtoV-manysig-m1-r01-launch；owner=codex/root/ampm-source48。CPU2、CUDA_VISIBLE_DEVICES为空。代码/config commit 由本次 Git 提交固定并写入 release_commit.txt/resolved_config；实际值后补现有登记。启动首步7项合成 source-statistic smoke，PASS立即继续，无模型消费所以不加载额外 checkpoint。一次普通账户preflight、一个归档一次SHA比较、一次remote compile、一次PID/CWD/argv/环境/日志增长检查。

仅所属路径冲突、源角色/输入不符、非有限或异常触发技术停止，保留所有产物，不自动重试、不干预健康旧任务。真实 source FAIL 是正常科学结束。预期 resolved_config/checks/active/progress、私有 observations.npz、result/completion或failure、独立recount和源RX/18折CSV、PNG/PDF、同row分析及登记/Git交付。额外审批/签名/成员哈希/重复数据验证/报告only review均不增加。

# 三步物理展开：共享与不共享先验恢复 R17

状态：LOCAL_VERIFIED，未发布。代码配置 `cd97744884b5e460926e48c8b7efb7d78c6da0f4` 已提交推送并独立比较远端OID；43项聚焦检查、compile/help、一次独立P0/P1无阻断及普通N607预检通过。入口先执行真实2Ls、R07父/新展开checkpoint/原R16单次checkpoint/原F50以及每个学习步梯度smoke，PASS立即继续。

本轮执行用户报告R3的3步/共享与不共享/估计作用/软观测约束思想，采用warm-start加权MAP的CG实现，是接收链适配，不是报告梯度步公式或[MoDL原论文](https://arxiv.org/abs/1712.02862)MRI实验的逐式复现。R06/R14已有源已知作用诊断，波形改善未带来相称身份收益；新展开oracle反事实尚未做，不称部署上限。R16 IQ-only平均改善仍损伤TX14-7，本轮检验交替表示与求解过程，不用目标反馈或标签去训练身份损失。

每步Core9输入received2/coarse2/current2/单位RMS的Aᵀ(r−Ax)2/lognativevariance1；prior=coarse+SCALE×Core，随后2次warmCG，三步总6次。R07估计A/nativevariance和coarse全冻结，不输入真实信道/噪声/TX/场景；全部步骤端到端，最终IQ NMSE唯一训练目标，0TX/F权重/输出/target训练。共享835842、不共享2507526可训练参数；后者三Core初始全克隆同一个Core，两种初始完整输出相同。2CG重启数值路径与R16单次6CG不同，不能称旧初始化相同。

全6300L/56700U，各20轮/4940、L曝光126000/U曝光1134000，batch256、AdamW2e-4→2e-5、warm247、WD1e-4、clip1、FP32。加载R07后重置model2701，data2001与R16相同；R07原20轮缓存15%clean/85%residual，未增加独立物理记录或噪声。只继承R07冻结上下文，不继承R16/R15训练后prior；新prior E20，组件E60，F50历史E50另列。每样本三次prior前向，计算量与单次网络不同，参数/训练时长/显存/延迟如实报告。

固定最后20轮，每候选1800sourceV七预测先固定，独立CPU再评分。每候选必须同时通过R16单次/native及R15FFT/native两套原护栏：mean6NMSE≤1.05倍、cleanNMSE≤.02、low/midurban各≥+.5pp、clean相对raw下降≤.1pp、最差低urbanTX≥−1pp、其他五场景最差变化≥−.5pp。只在合格者中按低/中urban平均、低urban最差TX、最差RX依次最高、最后名称字典序选一个。两种都可晋级，但不能在新目标访问后重排；V标签选择单独披露，0V持久更新。

无合格者0target；若选定，自动“选定展开+固定R16单次IQ”两行正式确认，各原168000物理记录×七视图，每行七预测全固定才独立truth-last。旧单次仅target-only加载原完整权重，新训练0轮；原F50/defaulthead冻结，单份IQ供所有原FFT/PA/时域分支。三郊区、三urban、clean、TX/RX/day、repair/damage、NMSE及总体FFT/配对特征和成本均保留。最终lowurban≥70%，其余五场景≥原rawclean84.6761904762%−5pp=79.6761904762%，恢复clean损伤单列；不称新的独立盲测。

一个owner、独立不可覆盖root/release；池0/1/5/7择空卡、最多两条本任务worker，每卡一条，全owner cap2/free12GB，共享capacity4锁到CUDA PID可见，子进程不重复锁。技术失败停失败行/pending暂停、保留输出、健康任务继续，无自动重试。完整矩阵、命令、环境/CWD、相对路径、GPU与停止规则见[登记](experiment.json)，[追踪记录](../../../docs/PAMF_UNROLLED_RESTORE_R17_20261020.md)。


发布后状态：VERIFIED。单次归档传输 SHA 比对、远端编译、真实两条源样本检查通过。两结构初始完整输出相同，梯度仅进入新 prior。控制器及两条 worker 的 PID/CWD/参数/GPU 和日志增长已核实；物理 GPU 0、1 各一条，未干扰其他 owner。无校准、无 TX/F/target 训练。逐行源评分及有条件正式确认继续自动执行。

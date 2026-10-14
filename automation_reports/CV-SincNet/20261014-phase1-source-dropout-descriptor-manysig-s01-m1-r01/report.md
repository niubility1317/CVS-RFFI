# 固定源时序描述子 Dropout 取点诊断

依据用户附件报告三·路线三，先量化 t2.pool 与 t2 完整输出的关系，以及末端 .1 Dropout 对路径面积的独立扰动。当前 moment/area 的六个健康训练保持原方案；本诊断不根据目标分数改变取点、Dropout 概率、参数或父权重。固定使用预定义最低 seed2026092701 的 SIG_TIME final50/11100，来自 20261014-phase1-moment-area-oldgate-cos-e50-clean-manysig-s01-m4-r01，执行512c10a267b5e654cfd2c983ab1debdeebc7f626。该源 checkpoint 为从零旧门控＋EMA＋cos，默认头联合训练，原物理数据契约已核实，实际零额外 fitted probes；source_proof 及固定执行ID在加载前核对。

单行 SOURCE_DROPOUT-s2026092701。L6300 中按 physical ID 排序每6TX×5RX×3day单元前8，固定720个源包。无 U IQ／隐藏真值、无目标或 V forward；复用已准备 source cache，V可被materialize但立即discard。NoExplicitCFO及内部补偿保持。骨干、头及buffers全部eval/frozen，既有 cosine结构不变，无训练/优化器/辅助分类器拟合。

真实 t2.pool 为96×32（norm/activation/pool后、末端Dropout前）。先在所有eval的真实fθ前向验证 pre/post 完全相同，捕获 h；只在该冻结中间值上回放8个固定seed的 inverted Bernoulli p=.1，再使用实际冻结6D Conv1d+tanh／4×8窗／SIG_TIME pool。统计increment6、feature-feature area15、feature-time area6、SIG_TIME28、COV_SHAPE27、AREA_SHAPE27、实际added-residual160；overall／5RX／3day共63记录，输出 clean RMS、noise RMS、noise/clean分母、有偏mask均值扰动与跨mask变化。它是单处隔离扰动，不是完整真实训练随机流、BN或EMA轨迹，也不由此证明性能因果效果或跨RX不变。

预算：0新训练轮／0 optimizer／0 head更新；720真实source feature forwards、8中间回放，8次回放不是8模型seed。所有原模型参数／buffers／eval状态结束前逐值相同，无grad积累。保存 generated source descriptor arrays 到 observed_descriptors.pt，独立重读复算不得新增IQ forward；原始中间数组不上传Git。只上传聚合结果与报告。

已有本地 native 零IQ路径、确定性回放／零噪声复算、固定63统计、8个协议负例和remote scripts compile PASS；未读取源或目标IQ。一次独立P0/P1只审新增实际诊断执行，不重审旧descriptor训练核心。Git实现与配置提交后执行既有最小发布流程，一个archive单次SHA比较／remotecompile／固定真实CP两零IQ无查询smoke，PASS直接继续。无额外token、hash、数据重验或审批。

入口 python -u -m experiments.cvs_source_dropout.probe，普通N607既有CVS-RFFI CPU，CUDA_VISIBLE_DEVICES为空，Torch/OMP/MKL/OpenBLAS2threads；CWD /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_source_dropout_20261014_r01；不可变output /home/szu2070436088/2510044040/CV-SincNet/runs/20261014-phase1-source-dropout-descriptor-manysig-s01-m1-r01。source contract复用原392005及既有L/U/V角色。不占GPU、不停止或热修改其他owner。

Stop：角色／来源／非有限／形状／模型状态变化失败保留输出并停止，无自动重试；不按目标或本诊断结果重采样、换checkpoint、扫Dropout参数。源诊断本身无目标测试或84%结论。预期results／resolved／observed_descriptors／completion、release preflight／submit及日志。独立PID/CWD/cmdline与产物复算核实完成后登记ANALYZED，Git提交推送并核对remoteOID。

实现提交 d34a7b91b91fb8672e622c3e39fb5142a1774dce；唯一新增执行 P0/P1 审查 PASS，无发现。固定来源／全eval／L720／8确定性CPU回放／保存重算及零训练拟合路径已核对，尚未发布实际源诊断。

实际已完成 ANALYZED／VERIFIED，source720／8mask／63stats，0训练拟合、模型状态不变、保存数组重算通过。feature area噪声/clean 31.9940%，time area21.7255%，COV形状12.7249%；见 final_analysis.md/results.json，不是性能结论。

# Phase3 多模态密度统一方差检验（2026-10-31）

Run `20261031-phase3-tied-density-s2026092701-r01`。当前用户选择的研发基线继续为 `predicted_ratio`，原 `pooled_ratio` 保留；本轮只检验一个独立候选，不按结果改写旧方法。尚无新性能或达标结论，完整目标 ACTIVE。

## 唯一候选和固定矩阵

原 source-only r03 signal、原 F0/default6、DURA v1.0 R69、4af class-density 的白化、known90/OE32 中心和权重冻结。新评分令全部已知与背景分量共享原 `pooled_L_variance` 标量 s，s 必须有限且大于0；不从 V/OE/U/Q 再估计或搜索方差。旧密度11数组逐值保留，新增评分分布与旧的逐模式异方差分布明确区分。

令 w 为原单位 z160 经原白化后的坐标。新分量项为 `log(weight) + w·mu/s - ||mu||²/(2s)`，分别稳定计算已知与 OE 的 logsumexp，再相减。它与共同方差 Gaussian 似然比严格等价：共有的 query 二次项、Gaussian 体积和白化 Jacobian 抵消。研究假设是减少这部分域偏移影响；它不保证任意方向的接收机偏移不变，更不预先保证未知拒识改善。此前 MD/RMD 已用相对距离，本轮不泛称首次相对评分。

四列顺序固定：`pooled_ratio`、`predicted_ratio`、`tied_pooled_ratio`、`tied_predicted_ratio`。唯一主为最后一列，倒数第二列只诊断。旧两列原分数/阈值/身份精确复用；新 pooled 用全部90已知模式各1/90，新 conditional 只按原 F0 预测 c 选择其15个 RX/day 模式各1/15，背景沿用旧32模式及权重。未知真TX、query RX/day、真实K和配额不路由、不拟合；身份始终由原默认头输出。

## 数据、校准与评估顺序

实际原 F0/r03/G checkpoint 的各2条无 query 合成 smoke 为首步，通过后立即继续。原源状态与评分政策先固定；仅源已知 V27000 按既有 known12 风险规则校准两套新六阈值（12个标量），旧两组六阈值保持。V只校准阈值，不拟合特征、密度或normalizer。全部24阈值冻结后才计算源 NV3150及 clean/star Q各864；四列×三个DURA条件全部预测固定后，唯一独立CUDAempty CPU scorer才连接源NV、cleanQ、starQ真值各一次。源科学指标不通过仍完成全部已登记Q，不提前评分或取消。

源 L6300、隐藏 U_s56700、OE6134 均只继承旧训练/统计成本，本轮不读取其IQ。L占原训练池10%，V27000是额外已知校准监督。源RX1/3/4/6/8对应1-19/18-2/19-2/2-19/3-19，日期2021-03-08/15/23；旧六TX14-10/14-7/20-15/20-19/6-15/8-20，源NV未知19-15/19-16/7-1各1050。

跨RX Q为24-1/index26、2021-03-15同864物理记录：六已知各96＝576，三未知各96＝288。clean/star均复用4af已实际通过同一DURA的原signal160/default logits及对应opaque IDs，不新增真实神经前向。clean仅冻结测试例外，不进入Phase3学习。G10 MIX01固定600km/25MHz/2.45GHz、residual/post_sync/noEQ/refSNR30dB/reflect、原六场景权重和郊区城区各0.5、receiver2027/scene2026103001/channel2026104001，单记录单固定received；不重抽信道。同物理跨条件处理不增加独立样本数。

目标U384及正式RX24-18不访问。sourceNV/Q均历史曝光；单seed2026092701、单目标RX、日期与源重合，不是新盲测。practical_high未知三类数0/4/2，缺19-15，宏RU/H保持N/A，不能认证六景全通过。

## 指标、门槛和成本

主报逐真实TX宏A_K/R_U/H，另报C_K、同DURA原F0闭集−A_K、AUROC/OSCR、逐TX/RX/day/scene及完整coverage、配对正确接收和未知拒对gain/loss/net。硬门槛沿用源known12、跨RX clean RU≥95%、星地RU≥90%、CK≥90%、同路径原F0闭集损失≤2个百分点、预登记主场景完整类别覆盖；主星地AK和RU均不低于原predicted控制。H没有替代单项约束的独立门槛，30个源RX×scene组为诊断。全部原运行点及负结果保留，不按Q调阈值/换主/选择性重跑。

新增NN更新、密度/normalizer拟合、新拟合方差参数、真实F0/G/signal前向、源IQ和目标U读取均为0；新V阈值12个标量，旧12个复用，24个FP64 τ共192B。原density11数组364340B，部署神经F0 232993＋signal334721＋G266951＝834665参数；祖先F0 11100更新、signal既有2220、G4940和源统计/缓存成本继承。三CP合成各2条另计，不部署逐行source/U bank，不把进程墙钟称为单包时延。

## 执行与停止规则

一个代码/配置Git commit固定执行身份；实际commit、完整命令、解释器/CWD、不可覆盖run/release root和六类seed在实验登记 `experiment.json` 唯一维护。入口 `experiments.pamf_phase3_open_world.star_tied_density_launch start --runtime <release>/runtime.json`，矩阵仅单seed单候选，自动predict→独立score→同row分析。N607既有CVS-RFFI Python；GPU0至7中启动时实际剩余≥4096MiB，每卡本任务新增训练/预测worker最多2，本轮仅单producer；评分CPU且CUDA为空。不得干预其他owner工作。

按项目最小流程完成聚焦负测和一次真实CP smoke、一次直接P0/P1独立审查、一次preflight、单release包SHA比较与远端compile；启动后核所属PID/CWD/argv/GPU/log增长。非法角色/祖先、非有限s/分数、输出碰撞或无法形成合法prediction时保留全部产物；不覆盖/自动重启，不让partial predictions接truth。UNKNOWN先核原run/PID后处理。源科学低分不是技术故障。原F0/G/r03不更新，不加载延后的native-energy草稿；无目标DA/发现/匿名REG，AN/HOW为N/A，完整目标保持ACTIVE。

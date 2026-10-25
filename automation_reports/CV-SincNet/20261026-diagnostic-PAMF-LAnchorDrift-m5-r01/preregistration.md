# 同源E30/F50的固定与当前L参照诊断

PLANNED。五个checkpoint不是五个种子：全部query seed2026092701，分别为原E30、既有PLControlF50、任务缩放1x/2x/3x F50。源剂量三条留出净增36/34/28、参与RX收益却扩大；本run只检验固定E30参照是否随训练与当前空间失配，不把该假设当作原因定论。

只原可见L6300、同原G-LU。E30通过既有load_sources核对ownE20+10；四F50先validate_final拒绝target/Utruth/换头/错seed/错sourcecontract，再用既有load_reader/load_F核对真实defaulthead及完整20/10/20来源。F50不是新增训练父模型；PLControl为预定义匹配对照，未据目标选它。VALIDATED_ONCE和原source角色不变，U/V/目标不打开。

每个冻结模型用同一L真实标签/RX重算单位类中心CurrentL，与自身原E30固定中心FixedE30比较。所有类/RX同公式，排除query自身RX，其他RX等权、温度.1；没有新投影、headfit或参数优化。E30自身的两个中心应数值一致（maxabs≤1e-5），并检查对应预测与梯度一致，作为零漂移控制。

query覆盖raw/Graw、三种low suburban/mid urban/low urban的H/GH，共8视图。H生成沿既有source_channel、namespace PAMF/set_reference/L/fixed、channel/hardware seeds20261016421/20261016423、residual/post_sync/noEQ/fs25e6/fc2462e6/SNR30/reflect，G为reuse=True/graph=True/native=False、FP32/TF32off。参照身份分数只作非参数特征诊断；默认头不变。可见真标签CE只在独立detached feature求导，模型参数.grad全None。逐TX类别索引/RX/day与overall、H/G三景汇总，共1500cell，另150个中心位移。L是既有训练来源，不能称为新目标或真正未见RX验证。

全部F/G参数及persistent/nonpersistent registeredbuffers必须不变。新增训练0；query模型历史30+四20尾＝110唯一F轮，若计另两E30 bootstrap前缀，共170已存在F轮；不称为新训练或免费成本。新F前向283500/G25200包，另计真实L2smoke；3000次特征autograd，无encoder或optimizer更新。

N607既有普通账户Python/Torch2.1、CPU2threads，GPU0准入时计入全局CUDA+pending<2/free≥10000MiB，保护其他owner，不安装依赖。命令PYTHON -u -m experiments.pamf_channel_pl.anchor_drift_source dispatch --config PROJECT/releases/20261026-diagnostic-PAMF-LAnchorDrift-release-r01/runtime.json；CWD该release，输出PROJECT/runs/20261026-diagnostic-PAMF-LAnchorDrift-m5-r01。代码身份runtime.code_commit，不可覆盖run/release，一次archive SHA传输比较、remote compile、启动后PID/CWD/argv/GPU/loggrowth检查。短run若观察前完成，注明NOT_CAPTURED，用最终artifact证据核实，不为观察窗口重跑。

两个来源/角色负例和几何控制检查、相关compile PASS；本候选唯一一次独立P0/P1审查PASS，无P0/P1。实际五checkpoint/defaulthead/G L2 smoke为launcher首步，通过立即继续。任一来源/协议/smoke/非有限/状态更新或技术失败终止本run，保留产物，不自动重提、不热改或干预其他任务。私有中心/逐包数据/权重/runtime不提交Git。

源几何不是目标性能。全部结果保留，本run不自动训练、扩种子或更改默认；后续方法仍从原ownE30到E50，原+.3/- .1源条件和三种子rawclean+1pp/raw六景>1.5pp/同G六景超过对照的最终要求保持。

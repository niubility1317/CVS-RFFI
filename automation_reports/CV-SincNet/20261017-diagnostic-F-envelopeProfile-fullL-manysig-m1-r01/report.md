# 全源慢变包络：固定依据检验

状态：ANALYZED_SOURCE_STATISTICS_ONLY／VERIFIED。固定包络依据失败并关闭。模型训练预算仍E30，本行0训练。

唯一假设：现有EQ1/RMS IQ256中存在跨TX/day可重复的RX关联慢变功率形状，能为后续轻度时变增益观察提供依据。它不同于已关闭的整包常数gain（后续RMS消除）和平稳频谱FIR（旧有限PSD保真失败），不重开或修改旧候选。

[AD9361参考手册UG-570](https://www.analog.com/media/en/technical-documentation/user-guides/ad9361.pdf)与[官方AGC说明](https://wiki.analog.com/resources/eval/user-guides/ad9361_agc_tuning)说明接收gain改变涉及settling过程，前导期间可能调整gain。这只支持物理可能性；未证明WiSig采集启用AGC、当前EQ1保留AGC残差或RX关联等于纯接收机因果。不得据厂商说明宣称数据中已有该效应。

固定观察：全256点，不裁切/移位；16个连续不重叠块、每块16点的I²+Q²均值，除以包总平均功率，relativefloor1e-6取log并减16维均值，再投影正交DCT modes1/2。不选择其他window、block或mode。整包常数gain严格消除；慢时变形状可以保留。当前仅测度，不生成IQ变体或选择gain幅度。

合法数据：复用既有source_cache.pt及source_contract句柄，L6300全体、6TX×day[1,2,3]×RX[1,3,4,6,8]的90格各70；U56700 TX始终隐藏、不使用，V27000与target不使用。缓存可反序列化其他源角色旧特征但不读取其标签/进行统计。模型训练无RX留出，当前没有模型训练或CP。本行18折仅为统计拟合：每折同时排除一个TX与一个day，只用其余5TX×2day×全部5RX，预测被排除TX/day的RX中心化二阶系数；全部5RX在每折中保留。不是L5580/dev720或训练域留出契约。

固定对照：对预测RX效应循环错位1个RX；不搜索control。全部18折/5RX/90格报告，不丢包或删除失败RX。PASS需同时满足：总predictionSSE/zeroSSE≤.9；至少15/18折优于零RX对比；总predictionSSE小于固定错RX对照；每个RX的汇总相对MSE<1。零分母视为无可检验依据并FAIL，不能筛除。任何失败关闭固定包络测度，不改block/mode/floor/window/RX/seed/阈值救跑。

即使PASS，也只证实部分RX关联形状，不能证明AGC、身份保持、Ag、As或联合收益。之后必须另行固定有界IQ算子与原默认头身份保护，再进行匹配E30训练；本行不自动训练或打开目标。本行F＋跨RX同TX基线不变；未来同模型默认cosine头/NoExplicitCFO/必要逐包补偿/原U/全源契约保持。

最小矩阵：一行ENVELOPE_FULL_L，1次cache、6300唯一IQ、90格、18折；0CP/NN/新IQ视图/teacher/headfit/updates/target。无模型阶段不增加未消费的checkpoint smoke或额外闸门。合成检查PASS：8项输入反例、18项排TX/day检查、常数gain不变、慢包络可观测、可加RX效应恢复、纯TX及TX×RX交互拒绝；0真实IQ/CP/NN/更新。唯一限定独立P0/P1审查PASS，7文件与2段发布脚本语法已核对。

commit为本行实际Git提交（发布自动填入）；命令 CVS-RFFI/bin/python -u experiments/cvs_envelope_profile17/runner.py；CWD=PROJECT/releases/cvs_F_envelope_profile_source_20261017_r01；环境既有普通N607 CVS-RFFI/CPU2/CUDA_VISIBLE_DEVICES空，不占GPU。owner=codex/root/F-envelope-profile-source。immutable run=PROJECT/runs/20261017-diagnostic-F-envelopeProfile-fullL-manysig-m1-r01。只在输入权限/非有限/输出碰撞/系统性技术失败时停止本行，保留产物，无自动重试或健康任务修改。

预期checks、resolved config、私有source_observations.npz（只统计量与sourceID，未存新IQ）、result/completion、stdout及独立90格/18折复算。仅代码、登记和汇总进入Git；原IQ、身份ID、缓存和私有日志保留在既有受控位置。一次preflight/归档传输比较/远端compile；启动后核实PID/CWD/argv/CPU环境/log增长或自然终态，不从启动返回值推断成功。

## 实际源结果

执行commit `9aea3ef017db7491a903e460d59caf0642cd5e72`；单归档传输比较、远端compile与合成checks PASS。普通CPU进程640052的PID/CWD/argv/CUDA空配置已观察，随后自然结束，completion齐全且failure不存在；已独立核实终态，不将publish返回值当成功。

全L6300、90格各70全部参与；1次cache，6300唯一IQ，0CP/NN/增强/头拟合/教师/训练/target。主体耗时 0.746861秒。原U/V仅旧缓存特征可反序列化，不读取U真值或使用其统计。

预测SSE=6.856663376364，零RX对比SSE=5.894453939727，固定错RX对照SSE=6.857530288377；总相对MSE=1.163239792（要求≤.9），仅3/18折改善（要求至少15）。错RX对照虽然单项略差于正确映射，但不能替代其他失败条件。

| 源RX | 相对零RX对比MSE |
|---|---:|
| 1 | 1.100670617 |
| 3 | 1.140431345 |
| 4 | 0.765527377 |
| 6 | 1.349024850 |
| 8 | 1.186332689 |

已证实：固定测度会消除整包常数gain，保留非恒定形状；当前观察在全源上缺少跨TX/day稳定RX依据。被否定的是这一固定DCT1/2慢包络模型作为共同RX扰动依据，不能扩展为所有包络或真实AGC效应不存在。只有RX4汇总改善，不筛RX4，不修改block/mode/floor/window/阈值或重抽样救跑；不生成包络IQ，不启动该候选E30。

独立复算6300观察的12600个DCT系数、90格均值/方差、18折排除/预测/错映射与5RX汇总，最大系数/格/折误差均≤4.45e−16；所有科学判断一致。复算0原IQ/CP/NN/更新/target，只读保存的源统计。

本行没有新的身份识别成绩，不能宣称Ag、As或完整联合目标改善。上一轮固定N0训练配方仍关闭；当前无已通过来源依据的新Ag训练候选，既有模型和测试产物保持。

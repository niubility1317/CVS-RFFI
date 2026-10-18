# P74 冻结源域四贡献噪声归因

目标优先级保持星地优先；clean 相对匹配 NoAs 下降不超过 0.1 个百分点只是阶段目标，最终仍需可重复双提升及 low urban >60%。本轮是源诊断，不新增训练模型。

P73 概率均值 CE 已完成且相对 mean CE 的七视图均为负，固定候选关闭。P70 分层风险、P71 有界风险、P72 mask 候选也已关闭，不换名重跑。另一对话正在研究 U practical 路由，本轮不重复。

唯一假设：完整 As 后剩余噪声损失是否集中于一个真实加法贡献，或该贡献是否在噪声下误导默认头。冻结原 P43 B00 NoAs30 与 B01 As30（seed 2026092701，同原物理契约；来源严格核查），使用固定 V1800，6TX×5RX×3day 每格20包，两条命名信道流，mid/low urban。保留原 clean、full、no_noise/fixed。缓存为既有 P24 2.45GHz 诊断条件；不冒称原训练2.462GHz。IQ2×256、fs25MHz、原逐包RMS center=false；residual/post_sync、EQoff即G1、ref30、reflect、接收质量适配/相位跟踪/IQ补偿开启，正式测试不改。

四贡献为 base_norm(TF)、tanh(gain)×pa_norm(PA)、tanh(response_gain)×response_projection(REF)、已含gain与LN的整个SIG_TIME_X3。每次只将一个 full 贡献置零、换成同包 no_noise/fixed、换成同包clean，再按原顺序相加并通过原cos30头重做L2。clean另做四个zero控制。不拟合头、不改特征、不把logits线性相减；不强求子SIG/T/F解耦。

预先固定读法：As中同一分支的去噪替换在mid/low×两个draw每格都净增≥2pp且恢复全去噪增益≥50%（全去噪净增须为正），才支持局部噪声证据路线。同一分支zero在四格都增≥1pp且cleanzero代价≤.1pp，才支持轻量屏蔽值得后续研究。未满足则不按单格、最佳draw或近阈值宣称支持。二者均不是自动训练或E50晋级。任何oracle替换不称可部署收益/可达上界。

统计原头accuracy/macroF1/worstRX、margin÷30、修复/新增错误、固定NoAs clean正确集合上的保持率、state/lock/state×lock/addednoise/RX/day分层及learned coordinate gains；不额外训练诊断头。两draw不作为两个modelseed。

代码commit：PENDING_ACTUAL_CODE_COMMIT。环境为N607既有普通账号CVS-RFFI Python，FP32 CPU2，CUDA_VISIBLE_DEVICES为空；CWD PROJECT/releases/cvs_FcrossRX_branch_noise_sourceV_20261019_r01；命令：CVS-RFFI/python -u experiments/cvs_branch_noise74/runner.py。唯一输出根 PROJECT/runs/20261019-diagnostic-FcrossRX-branchNoise-sourceV-s2026092701-m2-r01。

新增训练0epoch/0update/0优化器，两个既有总E30冻结权重；32404整网样本前向（含两个checkpoint各2包真实smoke），219652额外head-only样本调用（含52smoke），新物理模拟0。首次真实smoke通过立即继续；失败保留输出、不自动重试、不触碰其他健康任务。一次最小P0/P1评审、归档一次SHA传输对比、远端compile及PID/CWD/argv/GPU/log增长核实。

预期产物：resolved_config、实际模块__file__、真实CPsmoke、panel_metadata、逐模型predictions/complete、整体complete、analysis_r01 JSON/CSV/readout与三张PNG/PDF。无新训练曲线和正式目标评分；源诊断本身不能完成最终目标。

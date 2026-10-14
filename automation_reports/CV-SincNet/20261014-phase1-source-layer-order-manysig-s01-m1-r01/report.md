# 浅中层有界身份排序：全 L 覆盖源诊断

报告建议围绕尚未满足的真实身份关系提供有限监督。当前路线独立于另一分支 joint160 排序原型，直接考察 t1/t2 既有时序输出的4格有序池化视图；不改变分类头、SIG_TIME结构或旧门控。

固定首定义 seed2026092701 的 SIG_TIME uniform scratch50 final checkpoint，真实来源见 experiment.json。先调用既有source_proof核对6300/56700/27000角色、split392005、33+17/11100、own scratch/无祖先及fixedfinal，再严格加载。目标成绩不参与模型选择。

70个平衡90包cohort覆盖所有6300个合法L物理样本，每样本一次；每cohort1800个同day跨RX身份三元关系（正负参照来自同RX），每视图共126000个采样关系，非所有组合。margin=.1、temperature=.2、辅助系数=.1在发布前固定，不根据活跃率增大margin。t1/t2局部视图与joint160分别统计关系覆盖、端点样本覆盖、30TX对/20RX对/3day组。对实际t1/t2/t3.pw求导，比较同图默认头CE、各层ORDER、两局部层平均ORDER及参照RX SupCon；辅助不依赖classW。

本次0epoch/0optimizer/headfit，无U IQ/真值、target或Vforward。CPU2线程，GPU不可见；6300源forward加90zeroIQ真实CPsmoke。原健康训练不受修改。state/buffer/.grad/RNG必须不变；raw PT留远端，保存后重算全部统计。角色/来源/形状/非有限/状态失败立即保留输出并停止，不自动重试。

精确环境、不可变路径、命令、owner、期望产物及逐行matrix均在experiment.json；代码身份为实际Gitcommit，归档单次SHA比较/远端compile与真实CPnoquerysmoke由发布入口完成。仅一次直接P0/P1审查。早中期真实梯度尚不在本次冻结final诊断中，后续若训练则预登记真实1/17/33/50观察点。

当前LOCAL_VERIFIED；9项协议负例、关系角色、全部6300物理样本一次覆盖、满足间隔的精确零loss/gradient检查PASS。源诊断不是target性能提升证据，84%目标尚未达到。

最终状态 ANALYZED，完整证据见 final_analysis.md 与 results.json；源诊断不代表目标性能提升。

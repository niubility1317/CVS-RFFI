# PAMF第三阶段：共同可行阈值的单调扩展

## 预登记（2026-10-30）

Run `20261030-phase3-star-common-seeded-risk-s2026092701-r01`，实际code `119210ba86fb8b569f6cdeb887d314c1458a2007`。两冻结arm detach_control/joint，三策略common_tail/zero_initialized_budget/common_initialized_budget；唯一主joint::common_initialized_budget。计划、科学规则和接口见`docs/PAMF_PHASE3_STAR_COMMON_SEEDED_RISK_PLAN_20261030.md`，机读配置`configs/pamf_phase3_star_common_seeded_risk_20261030.json`。原两轮负结果不追改，control或zero不替主。

NN固定r02/code5aa041122396d48c49b8cb3d5912649bdc3217e1的两模型及966参数aux，原六类默认头不替换、累计F13320；祖先原F0 E50/11100、L6300/U56700/V27000、split392005。OE公共分数和zeroτ来自前OE risk run/code585e10b07be1f6821edea08b4853582b92512aff，它不提供NN权重。严格原身份、原source契约与同一G10 P1保持。

新fit仅source V27000 true-six风险、原F0同P1闭集baseline和OE6134公共raw max/default-predclass/15域。新阈值从原common开始，逐元素只增；queue必须按真实score<initialτ计入known风险与已有OE层，不能用前一OE层近似间隙known。维持known42组宏CK90/原F0closed−AK2pp，最终exact-Fraction风险核对。沿用原域宏收益/新增风险除剩余预算的贪心规则，非全局最优。两个旧策略的τ与逐IDmargin精确复现；新接收是common子集，可保证同rawscore拒识不会下降，但不保证target保护或90%拒识。

源RX1/3/4/6/8、三日2021-03-08/15/23与历史novel19-15/19-16/7-1各1050仍为开发证据；42584原物理记录不增加。全部缓存由同G10 MIX01 received/DURA得到P1，本轮不新读真实IQ或前向OE/V/novel/G。每arm只有实际checkpoint两条合成G10 P0 noquery smoke，总F4；新NN/aux/G/optimizer更新0。不是新P0/P1实物对照或P1/G smoke，无L/Us/targetUQ/formal读取。

### 执行与权限

- Host N607；Python `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-common-seeded-risk-s2026092701-r01-release`；唯一owner `codex-phase3-star-common-seeded-risk-20261030`。
- 命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_common_seeded_risk_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-common-seeded-risk-s2026092701-r01-release/runtime.json`。
- output `/home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-common-seeded-risk-s2026092701-r01`；configs/logs/rows分目录，row `<arm>/predict`及`<arm>/score`不可覆盖；本地 `E:/type10-7/local_artifacts/pamf_phase3_star_common_seeded_risk_20261030_r01`。
- checkpoint noquery star smoke先于任何新source公共记录消费，PASS立即继续；source config不含truth/target。只读两个父marker/cal/阈值与OE_scores，然后known，全部三τ/cal/known写定后才打开父novel公共数组重算margin。
- row全部prediction完成即独立CPUscorer，private source truth仅该进程 `/home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase3-star-representation-s2026092701-r01-release/score_truth/source_novel.npz`。真实未知TX/K/配额不进入fit。
- 最多2rows分占不同实际free>=4096MB卡，每卡<=2新增，不停止/修改/重启foreign。健康另一row不因已返回分数而改变，科学失败不停止、热改或重跑。
- 预期launch/pipeline/rowstate/完整logs/configs，predict smoke/OE_scores/known/novel/18τ/caltrace/complete，score results/complete，独立observations。快任务做四次短只读观察以尽量捕获实际GPU样本；没有样本如实缺报，不为遥测重跑。

主报trueTX宏AK/RU/H、CK、原F0损伤、known42/unknown12、逐RX/scene/TX、OE域宏代理、pointwise monotone与same-ID配对收益；源crossRX1NN结构guard复用同单位特征，只是代理，不是AN。唯一主known42/RU>=90总体+5RX+6scene/原结构最多降2pp同时过，才准备另记小目标证伪；否则NO_PROMOTION。AN/HOW=N/A，完整跨RX星地拒识、零标签适应及未来匿名注册保持ACTIVE。

技术stop只限角色/来源/权限错误、非有限、输出碰撞或无法完成合法prediction，失败row保留partial，另一健康row继续。未知启动结果先读回不盲目重发，不能用低科学分数作为重启理由。

验证：25tests/42subtests PASS7.09s，4AST与实际runtime schema PASS；包括None旧行为、真实初始known间隙风险、已有OE prefix、单调θ/逐ID子集、两个旧策略精确复现、freeze前不读novel及truth-before-validation负测。唯一独立P0/P1 PASS，其定向6tests10sub与主suite重叠不加计；不重复父suite/报告审批。本run一次N607 preflight核实两父pred及模型checkpoint/原sourcecontract/真实显存通过，target0。按最小流程做一个archive SHA比较、一次remotecompile、first-step actualcheckpoint smoke、PID/CWD/argv/GPU/log增长及完整预测后的独立评分，不增加数据重验或封印链。

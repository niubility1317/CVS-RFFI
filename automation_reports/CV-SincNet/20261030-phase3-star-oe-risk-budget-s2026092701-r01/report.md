# PAMF第三阶段：源OE收益与独立类风险预算

## 预登记（2026-10-30）

Run `20261030-phase3-star-oe-risk-budget-s2026092701-r01`，实际code `585e10b07be1f6821edea08b4853582b92512aff`。两冻结arm为detach_control/joint，各比较common_tail与oe_risk_budget；唯一可晋级主为joint::oe_risk_budget。计划/算法/接口在`docs/PAMF_PHASE3_STAR_OE_RISK_BUDGET_PLAN_20261030.md`，机读配置`configs/pamf_phase3_star_oe_risk_budget_20261030.json`。不追改父r02的NO_PROMOTION，也不换较高分控制。

固定父为原jointOVA r02/code5aa041122396d48c49b8cb3d5912649bdc3217e1的两个完整native+aux checkpoint，累计F13320/source-only、原六类默认head不替换；原F0祖先E50/11100、L6300/U56700/V27000、split392005，来源已核实、无target适应。新NN/aux/G/optimizer更新全0，真实新F每arm6134条OE P1＋2条synthetic G10 P0 smoke，总12272；复用父V27000/novel3150公共预测、原F0同P1 baseline与单位特征，不新读其IQ或新前向G。源L/原Us/目标UQ/formal读取0。

同五source RX1/3/4/6/8、三日2021-03-08/15/23、同G10 MIX01物理记录及DURA缓存。sourceOE6134只有outside-six成员关系和15域，不含真实TX/K；novel三设备各1050已有历史开发曝光。本轮不是未见targetRX/TX盲测，source通过不代替主跨RX星地验收。

common_tail精确复用父ova_max六τ，rawmax及默认身份完全相同。独立算法在六预测类各下一OE完整分数层中分配风险预算，OE域宏拒识为proxy收益，收益/(最大新增风险除以当前剩余预算+1e-12)贪心择优，classindex打ties。V全部42组逐真实旧TX宏CK>=90%、原F0closed−AK<=2pp保持。sourceOE与knownV可参与阈值拟合，novel不参与；先写全部τ/cal/known，再读取父novel公共数组重算固定margin。仅6float64部署阈值，无bank/域标签部署，无最优性或分布外覆盖保证。

### 执行

- Host N607；Python `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-oe-risk-budget-s2026092701-r01-release`；唯一owner `codex-phase3-star-oe-risk-budget-20261030`。
- 命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_oe_risk_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-oe-risk-budget-s2026092701-r01-release/runtime.json`。
- run `/home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-oe-risk-budget-s2026092701-r01`；configs/logs/rows分目录，row `<arm>/predict`及`<arm>/score`不可覆盖；本地 `E:/type10-7/local_artifacts/pamf_phase3_star_oe_risk_budget_20261030_r01`。
- 两arm各自实际checkpoint noquery synthetic smoke先于新OE/V读取，PASS直接继续。最多2rows分占不同实际free>=4096MB卡，每卡<=2新增，不停止或修改foreign工作。source完成立即独立CPU scorer，不等待另一arm；科学分数不影响另一健康row。
- private source truth仅独立CPUscore `/home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase3-star-representation-s2026092701-r01-release/score_truth/source_novel.npz`；source config没有truth/target字段，真实TX/角色/K不进入拟合。
- 预期launch/pipeline/rowstate/完整logs/configs，predict smoke/OE_scores/known/novel/12τ/caltrace/complete，score results/complete，observations。actualF/aux参数buffers不变，所有原输出保留。

主报宏AK/RU/H，CK及原F0损伤、42known／12unknown与逐TX/RX/scene、OE域宏proxy、原结构1NN guard和同ID配对差。唯一主known42／RU90总体+5RX+6scene／原结构最多降2pp同时过，才可准备另行登记小目标证伪；否则NO_PROMOTION，AN/HOW=N/A，完整目标ACTIVE。阈值不能按novel/target重选，controls不能替主，科学失败不重启或扩seed。

技术stop仅协议/角色/来源错误、输出碰撞、非有限或无法完成合法prediction，保留失败row和所有partial，独立健康row继续；不同于低性能科学失败。不在原root重启；任何未知启动结果先读回，不盲目重发。

验证：四模块21tests/34subtests PASS5.56s，含完整tie／局部预算／原baseline／source监督权限／freeze顺序／真实终态字段；4AST/runtime schema PASS。唯一独立P0/P1 PASS，定向6tests/6sub与主suite重叠不加计、不重复审父候选。一次本run N607 preflight读取现有role句柄／原sourcecontract／两个完整父pred／checkpoint及实际显存通过，target0。待一个归档SHA比较、remotecompile、first-step actualcheckpoint smoke及PID/CWD/argv/GPU/log增长，prediction全固定后独立truth-last评分；不增加数据重验、封印或报告审批。

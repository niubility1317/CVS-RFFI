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

## 实际结果（2026-10-30）

VERIFIED/ANALYZED_SOURCE_ONLY，唯一主joint::oe_risk_budget为NO_PROMOTION。两组source预测、冻结校准和独立CPU truth-last评分均完整，终态live/ownedGPU为空；target0，A_N/H_OW=N/A，完整Phase3继续ACTIVE。完整分析见 `docs/PAMF_PHASE3_STAR_OE_RISK_BUDGET_RESULTS_20261030.md`。

| 固定运行点 | 宏 A_K | 宏 C_K | 宏 R_U | H-score | 原F0闭集−A_K | OE域宏代理 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| detach_control::common_tail | 91.292593% | 99.144444% | 11.682540% | 20.714309% | -0.311111pp | 36.647997% |
| detach_control::oe_risk_budget | 90.381481% | 97.192593% | 20.253968% | 33.092172% | 0.600000pp | 40.116098% |
| joint::common_tail | 91.003704% | 98.462963% | 24.920635% | 39.126729% | -0.022222pp | 56.668574% |
| joint::oe_risk_budget | 90.900000% | 98.274074% | 20.285714% | 33.169215% | 0.081481pp | 53.122567% |

四运行点known42通过、unknown12未90%，原F0同P1闭集90.981481%。主预算较自身common净少正确接收28、净少拒146未知，OE代理亦下降；从acceptall收益0增加不保证胜过已有common可行点。不能把主退步归因为代理改善后的novel迁移失败，也不能由该贪心负结果排除全部独立阈值方法。control局部收益不能替主，保持原方法和全部负结果，不重选或重跑。

没有NN/aux/G/optimizer更新；仅OE6134+2合成G10探针/arm，新F共12272，V/novel公共预测复用、原始物理记录不增加，G0。结构guard沿原不变特征，两组仍通过，不是未来匿名注册识别。

driver3090943及source3090989/CUDA4、3090990/CUDA2的PID/CWD/argv/env有独立live读回；首次尚无CUDA显存样本，第二次已完整终态，故不报告实测显存/峰值，不为遥测重跑。source日志532→627/523→600B与全部complete/CPUscore闭合。整run墙钟10.486376s，不是单包部署延迟；otherowner未改。证据本run本地pipeline/observations/configs/logs/两组predict与score目录。

实际code585e10b07be1f6821edea08b4853582b92512aff，prereg643ee789e420d671ae0f8b1987ba21f39da7e3de，archive594571df9184c5b2da01d942e759353a9aa3aa876cdcc10340218cb56a720440／318文件／数据传输0，一次remoteSHA比较/compile/runtime PASS。21tests/34subtests、唯一独立P0/P1通过；没有目标访问、旧输出覆盖或其他owner进程操作。

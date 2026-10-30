# PAMF 身份与开放特征联合密度

状态PLANNED，无新性能。run `20261031-phase3-joint-feature-s2026092701-r01`，实际代码commit `b7f81eca9a2a0bb55afc361acd50f3fbd739b68a`，完整目标ACTIVE。

唯一主joint_predicted_ratio；原pooled_ratio/原predicted_ratio与joint_pooled_ratio为固定控制。原F0单位u160与r03单位z160同记录拼unit320；仅源L6300/OE6134估known90/OE32/共同within-mode白化和球形mixture，固定.9/.1收缩、known70/(70+320)、背景32/25Lloydseed3019。旧160state与两组六tau/cal/report/预测逐值复用，新joint先冻结再Vonly两组六tau，全部tau先NV/两个Q，全部三DURA完整预测后独立CPU三truth各一次。身份原F0/default6与F0/G/r03均冻结，不用局部cov、U学生或P0权重；完整方法见 `docs/PAMF_PHASE3_JOINT_FEATURE_PLAN_20261031.md`。

源RX1/3/4/6/8、2021-03-08/15/23；L旧六每类1050，每TX×RX×day70；OE6134仅outside-six/publicdomain，真实TX/K隐藏。V27000旧六每类4500仅known12校准（30cross另诊断），NV3150为19-15/19-16/7-1各1050。原L比例10%，V额外known标签；源IQ为既有G10MIX01/DURA P1，Us56700新0。sourceu原cache对齐ID/metadata/F0/G祖先，新sourcez前向12434。两个目标Q为RX24-1(index26)03-15同864物理记录，旧六/未知三各96；新Q同G/F0 u共1728，cachedz复用、Qsignal新0，核logits/closed匹配父；clean也G，不能称新物理样本或严格未见day。U384/03-08、正式RX24-18不读，high未知0/4/2缺19-15保留NA；历史sourceNV/Q曝光，非正式盲测。

继承原Tri-SIG-TIME SSL F0 E50/11100、source-onlyr03原2220增量和DURA v1.0 R69 E20/4940，sourcecontract392005/VALIDATED_ONCE句柄不改；classdensity父code4af0668。新NN/optimizer0，新joint统计fit1、tau12标量；sourceL/OE signal12434、QF0/G1728、synthetic各2；原sourceF0cache/父Qsignal及训练成本继承。部署NN834665不变，另报joint/旧state/tau聚合字节，不部署source行bank、不推定B1延迟。

主宏AK/RU/H，并列CK、同DURA原F0closed−AK、AUROC/OSCR、全TX/RX/day/scene/42组与主对三个控制同ID配对。gate为sourceknown12、cleanRU95/starRU90/CK90/2pp/主景覆盖，以及starAK/RU均不低于原predicted；sourceRU仅诊断，源不通过不取消Q。无目标真TX/K/配额/U/queryfit或匿名REG，AN/HOW=N/A，完整目标不缩小。

命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_joint_feature_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-joint-feature-s2026092701-r01-release/runtime.json`。N607环境 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-joint-feature-s2026092701-r01-release`；唯一owner `codex-phase3-joint-feature-20261031`。不可覆盖输出 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261031-phase3-joint-feature-s2026092701-r01`，日志同root/logs。单producer GPU自动选择实际free≥4096MiB、每卡新增本任务≤2；独立CUDAempty CPU scorer，不停止/修改/重启其他owner。实际GPU/PID启动后读回。

29相关唯一tests＋44sub通过（含共享320扩展所需160/640兼容回归），AST/镜像和一次直接P0/P1审查PASS。实际CP noquery smoke为launcher首步，PASS立即继续；单preflight/单archive SHA比较/remotecompile和PID-CWD-argv-GPU-log独立核实后收尾。非法来源/角色、非有限、输出碰撞或无法合法prediction时保留，不partial评分/自动重启；UNKNOWN核原PID/root，科学负结果不停止健康任务。

预期产物：launch/pipeline/configs/logs；predict models/density及joint/source拟合诊断，calibration/4×6tau，known/novel和两个Q predictions，smoke/predcomplete；独立score results/complete；observations与完整分析。IQ、权重、原日志和预测不进Git。

## 最终结果（JOINT-FEATURE，NO_PROMOTION）

- 三条件四方法全部完成，唯一主joint_predicted_ratio不晋级；原predicted重点候选/pooled对照保留，目标ACTIVE。
- 主source A_K/C_K/R_U/H=89.722222/96.785185/66.666667/76.494967%；clean=67.708333/84.201389/100/80.745342%；star=63.368056/84.722222/78.472222/70.115939%。
- 对predicted：source正确/未知拒对净−6/+149，clean−18/0，star−4/+4；星地ΔA_K−0.694444pp、ΔR_U+1.388889pp、ΔH+0.143614pp，H小增不救保护失败。
- 原F0闭集source/clean/star90.981481/80.034722/72.916667%不变。主clean/star身份损失12.326389/9.548611pp，C_K均<90%。sourceknown12过、42未过；high未知0/4/2缺19-15，R_U/H=N/A。
- 联合320统计新fit1（known90/OE32/25Lloyd）及新12个V-only阈值，NN0；joint数组1136236 B、原density364340 B、全24τ192 B，无逐行bank。
- 新signal12434/F0与G各1728真实前向，Qsignal0/sourceF0G0，合成三模型各2；NN834665参数及祖先训练/cache成本保留。
- predict/CPUscore/pipeline24.475319/12.593357/37.150677 s，非单包时延。独立predictorGPU4样本444MiB/日志549→695B，非峰值；CPUlive未捕获，终态空/foreign0。
- 29unique+44sub PASS、一次P0/P1 review PASS，616files/data0、单包SHA比较/compile/runtime VERIFIED；fixture NameError定向修正和旧trapz warnings已保留，非实际run失败。
- 源seenRX/历史NV与Q曝光，单seedRX，U/formal0、无DA/REG，A_N/H_OW=N/A。详见[公报](../../../docs/PAMF_PHASE3_JOINT_FEATURE_RESULTS_20261031.md)；未重fit/rescore/扫描阈值。

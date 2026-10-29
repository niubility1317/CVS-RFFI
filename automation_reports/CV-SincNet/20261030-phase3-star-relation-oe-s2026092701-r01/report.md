# PAMF Phase3 默认头开放暴露与相对结构保持

## 预登记（2026-10-30）

Run `20261030-phase3-star-relation-oe-s2026092701-r01`，code `19691a2cad35dcef6dd1ae42973bfba0e14ea705`。唯一主relation_oe::entropy_global，控制relation_control。完整固定机制/数据/主指标/停止条件见`docs/PAMF_PHASE3_STAR_RELATION_OE_PLAN_20261030.md`、`configs/pamf_phase3_star_relation_oe_20261030.json`，矩阵唯一维护在experiment.json。

两臂分别从原Tri-SIG-TIME SSL F0 E50/11100/default6head/232993独立开始，nativeSinc骨干与默认头联合2220次更新、累计13320（非自然10epoch），fresh CENTER_UNITS AdamW lr1e-4/继承parent wd/clip5，eval BN/dropout/固定buffer。L128 IIDseed2026092701、OE128域均匀seed2026102907，concat256单前向。两臂CE＋T2logitsKD×4＋OE offdiag Gram MSE权重1；唯一主加默认头KL(uniform6||p)权重1，控制该项0。无逐向量保真/aux/VIC/增广/U/PL/EMA/G更新，不扫权重或另换分数。

Teacher仅读旧jointOVA r02/5aa041122396d48c49b8cb3d5912649bdc3217e1的原F0 OE单位特征，非训练后joint模型；L teacherlogits来自同P1缓存。teacher原导出和G缓存成本保留。L6300/OE6134（membership额外源监督）/V27000/sourceNV3150，五已见RX1/3/4/6/8、三日、G10MIX01 P1。旧六标签合法，OE TX/K、Us隐藏真值未解封；NV三个历史设备各1050，仅冻结后独立评分，非新TX盲测。

两臂源训练固定后，knownV仅标量阈值：默认头1−H(p)/log6，单一全局最大合法观测值/完整ties，overall+六主景7组宏CK90及原同P1F0closed−AK≤2pp，其他35/42组诊断，不再加共同tail。闭集损伤已超限则finiteacceptall+known失败，仍预测sourceNV和开发RX24-1 Q864。sourceRU90不前置，两臂不按返回score变化。全部本row源/目标prediction固定后独立CPUscorer重建公有logits/argmax/entropy/margin/known7校准，再连接两truth。

目标RX24-1（源RX互斥）、G10现有futureQ864/2021-03-15，Q-only前向、无U/Q拟合、无未知TX/K/配额输入、正式RX24-18未接触。Q真实old576/unknown288和三未知角色仅scorer。既有U384保持未读；本轮只有DA0_REG0，AN/HOW=N/A/完整goalACTIVE。主报跨RX宏AK/RU/H、CK、原F0损、candidateclosed、AUROC/OSCR、六scene/逐类/CI及两臂同ID配对；七目标组同时CK90/RU90/F0loss2pp才为下一独立UDA资格，控制不替主，不宣称完整目标或新盲测。

### 执行、存储与资源

- N607 Python `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-relation-oe-s2026092701-r01-release`；唯一owner `codex-phase3-star-relation-oe-20261030`。
- 命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_relation_oe_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-relation-oe-s2026092701-r01-release/runtime.json`。
- 独占 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-relation-oe-s2026092701-r01`；每row train→完整source+target eval→独立CPUscore，GPU最多2row分别不同卡，实际free≥4096MB、每卡≤2新增own，foreign不停止/修改/重启。evalconfig13字段无truth/U/K/roles；两truth只runtime/controller和CPUscorerargv，producer不接收。
- 原sourcecache/L/OE/teacher只用于训练；冻结推理loader不回读逐样本源features。单模型参数F232993+G266951=499944、无aux；完整native训练wrapper900722另有不用的域分支，不将其误计当前识别器部署结构。singleτ8B，训练checkpoint含optimizer不当精简结构收益。
- 预期train final/resolved/summary/2220steps/smoke/complete；predict known/NV/targetNPZ、threshold.npy/calibration/smoke/targetsmoke/complete；score results/complete；pipeline/rowstate/logs/observations。新IQ/weights/pred/rawlog只private、不进Git。
- 每臂L/OE各284160曝光、训练F568320真实行；evalcandidate31014真实+4合成、originalF0额外4014真实，train烟测2合成。两臂完整F预算1206708行（真实1206696+合成12）；G新增1732（真实1728+合成4），缓存与教师祖先另计。新消费同42584源+864target=43448既有物理记录，不计额外独立仿真样本，不预报B1时延。
- 系统技术失败只保留failedrow、healthyrow继续；科学失败不重试/早停/重采样/重选阈值，UNKNOWN先核对同PID/root再决定，不重复launch或触碰foreign。

验证：26 passed, 232 warnings, 41 subtests passed in 7.57s，10AST/实际runtime PASS；唯一独立直接P0/P1 PASS，未重复主suite或接触真实IQ/权重/真值。一次N607 preflight核对原F0/teacher/source句柄与目标VALIDATED_ONCE/capsule/split，未读IQ或truth值；8卡实际free14797–20268MB，无foreign干预。单codecommit、单archive比较/compile、actualcheckpoint合成G10首步梯度smoke与launch后PID/CWD/argv/GPU/log增长、prediction全完后truthlast；不另加审批、封印或重复数据验证。

## 实际完成与跨接收机证伪（2026-10-30）

VERIFIED / ANALYZED_DEVELOPMENT，两臂train2220/sourceV校准/sourceNV预测/RX24-1Q864预测/独立CPU双truth评分完整。唯一主relation_oe::entropy_global为NO_PROMOTION；sourceknown7保护通过，但source与target未知拒识未达90%，target旧类保护和CK90也未达。source RU未通过确实未阻止预登记target验证，不改旧run阈值或将source成功当跨RX成功。

| 臂 / 数据 | 原F0闭集 | candidate闭集 | 宏 A_K | 宏 R_U | H-score | 宏 C_K | 原F0 closed−A_K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| relation_control / source | 90.981481% | 91.577778% | 90.103704% | 28.666667% | 43.495239% | 96.251852% | 0.877778 pp |
| relation_control / target | 72.916667% | 72.048611% | 67.013889% | 35.763889% | 46.638044% | 88.368056% | 5.902778 pp |
| relation_oe / source | 90.981481% | 91.544444% | 89.848148% | 57.841270% | 70.376484% | 96.137037% | 1.133333 pp |
| relation_oe / target | 72.916667% | 68.576389% | 55.555556% | 60.416667% | 57.884232% | 76.909722% | 17.361111 pp |

主target known正确320/576、接受443/576、unknown拒对174/288；控制386/576、509/576、103/288。同Q IDs完全对齐，主净正确−66、known接受−66、unknown拒对+71；分别AK−11.458333pp、CK−11.458333pp、RU+24.652778pp，H+11.246187pp，综合分提高不能掩盖known损伤。主candidate闭集395/576比原F0 420/576已少25条（4.340278pp），超过2pp，即便全部接受也不能满足旧身份保护；另外拒掉75条原candidate正确（13.020833pp），总损100条/17.361111pp。不把两种损失归成单纯阈值问题。

源主较控制正确接收346得/415失净−69，unknown拒对1027得/108失净+919；全部IDs对齐。新校准允许总体及六scene7组，42源切片完整诊断、RX/RXscene余35不阻断，无common-tail约束。两个新全局阈值独立由knownV冻结，不读novel/target分布。该作用与此前42组/不同模型实验没有匹配对照，不把跨run变化单独归因于放松校准。

完整4440条步日志/两臂stdout已读取，所有loss/梯度finite、每臂2220个批次逐步完全一致、L/OE各284160重复曝光且unique6300/6134。每臂单Foptimizer2220、Sinc/默认头梯度与参数变化全程记录、持久buffer冻结；无aux/U/EMA/PL/G更新。末222均值control CE0.158770/KD0.046404/Gram0.000553/OE诊断2.386768；主CE0.179459/KD0.108160/Gram0.003672/OE0.077297。OE目标得到优化，但source到target仍有明显拒识与身份差距，不用训练loss宣布泛化。

原F0 E50/default6头独立起点、teacher r02/5aa0411原F0 OE unit、同G10P1与冻结DURA祖先保持；source L6300/OE6134/V27000/NV3150、已见5RX/三日/historicalnovel，target为源外RX24-1/2021-03-15/Q864、历史开发暴露不盲测。targetU384新读0、Q拟合0/未知TXK配额输入0、正式RX24-18读0，AN/HOW=N/A，无新DA/匿名REG，完整v3目标ACTIVE。两臂神经active F+G499944、无aux，900722训练wrapper含667729未用域参数另报，单globalτ8B、不部署逐样本sourcebank。

实际F输入总1206708（训练1136640真实、两臂evalcandidate62028真实+8合成、原F0参考8028真实、train4合成）；G1732=target1728真实+4合成。42584source+864target=43448既有物理记录，不将IID重复或同received重用当更多独立样本；源Gcache/teacher导出/Phase1训练成本保留。进程批量时间与GPU采样不代表B1端到端时延。

code19691a2cad35dcef6dd1ae42973bfba0e14ea705/prereg198e3f7795b26a1331ad7cd6dfdc73c6f44eb1b6，26tests41subPASS7.57s/10AST/runtime/一次独立直接P0P1，单preflight/archive03a4503de06209a7188447c9d75ef54949ed571bd1a2360f5644e53844f2b056（375files/data0）比较/compile、actualcheckpoint syntheticG10梯度烟测及targetF/G smoke均PASS。首次release helper在本地package.json尚未生成时退出，发生于任何SCP之前；已等待同一打包进程完成，只传输同一个归档，无重打包/远端重复release或实验重启。

driver3240533、row3240623/24及train3240632/CUDA4、3240631/CUDA2独立live正确CWD/argv已核实；GPU各706MiB为离散样本非峰值，train日志control759→1551→2326B/main670→1333→2007→2184B增长，eval/score短进程未独立捕获live，仅jobs/stdout，未为补遥测重跑。终态pipeline ANALYZED_DEVELOPMENT/live/ownedGPU空、所有8个complete齐、foreign干预0；pipeline248.125857s含并行/加载/预测/评分，不宣称因果加速。逐TX/RX/scene/CI、完整训练、source1NN诊断和成本见docs/PAMF_PHASE3_STAR_RELATION_OE_RESULTS_20261030.md。

本轮结束，保留失败科学结果，不换控制、不调targetτ/训练、不扩正式或seed。下一科学问题是源端开放暴露造成的跨RX身份损伤及目标域拒识错配，需要目标U的受约束适应和独立未来Q证伪；当前没有据此宣称已获得有效UDA或无标签注册。

覆盖补充：本小规模targetQ的practical_high中，scorer未知类6计数0、类7为4、类8为2，必需未知覆盖不完整，该景R_U/H为N/A，不能认证通过；总体三未知各96条覆盖完整。保留原固定观测和全部预测，不据缺项重抽信道或补挑样本。

# PAMF 第三阶段：独立拒识度量及未知结构保持

## 预登记（2026-10-30）

Run `20261030-phase3-star-metric-embedding-s2026092701-r01`，实际执行code `efb82a4177e9b59533b7e71408ded6b63fc8733a`，固定三臂identity/compact_oe/compact_oe_structure，唯一主`compact_oe_structure::predicted_identity_mahalanobis`。原F0 E50/11100/default6head/232993和DURA v1.0/266951冻结；新MetricNet20704参数，160→64ReLU→160 residual+unit，CPU同seed2026092701、zero fc2，F0默认head仍唯一身份来源。

训练臂各固定1000步Adam1e-3/wd1e-4，identity0步；无EMA/scheduler/早停，identity明确不是同增量预算训练控制。source sampler同seed2026103002，每步L为6TX×5RX各4（120），OE128均匀无放回；两learned臂相同初始化/batch路径。known跨RX对比T0.1（16正/100负，同TX同RX不入分母）+10relu(maxOE原型cos)^2，唯一差别是主再加10*OE offdiagGramMSE。完整1000步loss/gradient/exposure/coverage保存，不按loss或V选择轮次；总2000新g optimizer步，F/head/G更新0。

网络冻结后只L拟合FP64六类均值/overallmean/within与background两whitener（0.2 isotropicshrink，N−6/N−1），四数组418560B，无source行。两score为minimum_relative=d_bg−min d_c（控制）、predicted_identity_mahalanobis=−d_closed（主）；closed来自原F0 head，不由g或近邻替换。主使用直接类距离保留OE排斥学习到的分离方向。所有g/aggregate在V前固定，V仅既有known42宏CK90/原F0同P1closed−AK2pp complete-tie规则校准每法6τ，共12；部署包/cal/known写完后才打开novel。完整prediction后独立CPU scorer全部公有校验/部署参数重建后才private truth，不重train/refit或重验父3score/IQ。

方法与精确合同见`docs/PAMF_PHASE3_STAR_METRIC_EMBEDDING_PLAN_20261030.md`、`configs/pamf_phase3_star_metric_embedding_20261030.json`。用户v3完整目标仍跨未见RX星地CK/RU≥90%、AK距同输入原F0闭集≤2pp，主报trueTX宏AK/RU/H；本轮不以source结果替代跨RX/无标签DA/未来匿名注册验证。

### 数据与权限

复用local-support r01/code84cdc012f7d9641b283c407f6048bfe4e6e21cf6的originalF0同G10 MIX01 P1单位fused160/logits：L6300/OE6134、V27000、novel3150，合42584同physical记录。原bank为jointOVA r02 teacher的originalF0，不是其更新candidate；cache为star-representation r01，DURA ancestry保留。仅L已有合法真旧TX/RX作contrast/sampling，OE仅outside-six成员，不读OE真TX/K；V真实标签只阈值，无模型更新。

sourceRX1/3/4/6/8（1-19/18-2/19-2/2-19/3-19）/2021-03-08/15/23均已见。旧6TX14-10/14-7/20-15/20-19/6-15/8-20，L各1050/V各4500；novel19-15/19-16/7-1各1050有历史source开发曝光，非新RX/TX盲测。原Phase1契约392005/L6300/Us56700/V27000/训练池10%标签、F0 modelseed2026092701不变；source监督不可称全系统零源标签。targetU/Q/正式RX24-18/原Us新读取0，真实未知K/角色/配额不进算法。

新真实IQ/F/G0；每臂首步actual原F0checkpoint syntheticG10P0 noquery smoke2（合F6），不是新G/P1端到端测试。生产g forward：identity L聚合6300+V27000+novel3150=36450，learned各再加120000L+128000OE训练=284450，三臂合605350；独立scorer重算另计。父F30152和历史teacher/P1G恢复成本保留祖先。

### 执行、导出与判定

- N607，Python `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-metric-embedding-s2026092701-r01-release`，唯一owner `codex-phase3-star-metric-embedding-20261030`。
- 命令`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_metric_embedding_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-metric-embedding-s2026092701-r01-release/runtime.json`；三row各source_config/输出独占，root `/home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-metric-embedding-s2026092701-r01`，本地 `E:/type10-7/local_artifacts/pamf_phase3_star_metric_embedding_20261030_r01`。
- parent features `/home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-local-support-s2026092701-r01/rows/original_F0/predict`，cacheL metadata `/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase3-star-representation-s2026092701-r01/cache/L`，原F0 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261027-phase1-pamf-repl-shuffled-views-m16-r02/F/main_split-repl-shuffled_time-3view-s2026092701/source/final_ssdg.pth`/sourcecontract `/home/szu2070436088/2510044040/CV-SincNet/runs/20261027-phase1-pamf-repl-shuffled-views-m16-r02/F/main_split-repl-shuffled_time-3view-s2026092701/source/source_contract.json`。复用VALIDATED_ONCE角色句柄，不重读真实IQ或增加hash/receipt。一次preflight已核原F0/父complete/sourceL合法句柄/实际GPU余量。
- 最多2个本任务source train/predict并行，actualfree≥4096MB、每卡最多2新增；不停止/修改foreign。actualcheckpoint首步smoke PASS即继续，单row source完立即独立CPU score，不等其他healthy行。技术stop限来源/权限/角色错、非有限、输出冲突、无法合法prediction，保留partial；不因指标低停/换主/重跑。未知启动结果先只读对账同run，不盲目重发。
- 每臂部署exact4files model.pt四tensorstate_dict/metric.npz四聚合/thresholds.npz12τ/complete.json，不含source样本/ID/标签/feature bank。仅g权重+聚合+query原F0unit/closed足以重建score。training.jsonl/traincomplete、原F0与g两套单位features/defaultlogits/raw/margin/12τcal/smoke/predcomplete、独立score及configs/logs/launch/pipeline/rowstate/observations为预期产物；模型权重、feature/预测和原始日志只留私有artifact，Git只code/config/report/registry元数据。
- 独立score先验证每臂预算/完整trace/F0defaulthead及来源/部署文件/state/τ/冻结标志；CPU重建g单位rtol1e-5/atol2e-6，再从核验saved单位用FP64state重算raw/margin atol1e-10。private source novel truth仅在该row全部prediction完成与公有校验后 `/home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase3-star-representation-s2026092701-r01-release/score_truth/source_novel.npz`。heldRX1NN结构只post-truth诊断，不是AN或新增gate。

主报三臂×两法宏AK/RU/H/CK/原baseline损/known42/unknown12/逐TXRXscene/AUROC/OSCR/CI、同ID配对、完整loss/曝光/结构/成本。每臂conditionalMD科学gate需known42和unknown总体/五RX/六景12组90%；仅固定主臂通过才READY_FOR_TARGET_FALSIFICATION，control只CONTROL_ONLY_COMPLETE，不能按控制较好换主。source通过只支持另记小target证伪，不扩seed或宣称完整目标；sourceonly/target0/AN/HOWnull/fullgoalACTIVE。

NN参数F/G499944+g20704=520648；每臂g FP32权重82816B/四FP64聚合418560B/两法12τ96B，不含optimizer和研究中间文件。训练/评测g与FP64score/独立复核耗时分别报告，GPU显存是采样，均不冒充B1端到端时延。

验证：21 passed, 816 warnings, 47 subtests passed in 13.07s，四AST/实际runtime PASS、唯一独立P0/P1 PASS；相关定向9tests20sub与主suite重叠不加计。覆盖loss/mask/梯度/采样/公平轨迹/source权限与真实预算、无行部署、冻结边界/truth-last/默认head与固定主晋级。单code/config commit/一个archiveSHA比较/remotecompile、真实checkpoint首步smoke与PID/CWD/cmd/GPU/log增长，完成后完整训练轨迹分析及逐rowtruth-last，不增报告审批/重复父suite/未来stageGate。

## 实际完成及独立评分恢复（2026-10-30）

VERIFIED：三臂模型/全部predictions及三臂×两法评分现完整，唯一主compact_oe_structure::predicted_identity_mahalanobis为NO_PROMOTION。原pipeline实际FAILED_PRESERVED保留，不改写其终态；compactOE独立评分由新score-recovery run补齐，identity/main原score复用。完整结果/数据/训练/配对见`docs/PAMF_PHASE3_STAR_METRIC_EMBEDDING_RESULTS_20261030.md`，完整Phase3 ACTIVE，AN/HOW=N/A。

| 固定臂 / 分数 | 宏 A_K | 宏 C_K | 宏 R_U | H-score | 原F0 closed−A_K |
| --- | ---: | ---: | ---: | ---: | ---: |
| identity / minimum_relative | 90.400000% | 98.681481% | 12.317460% | 21.680801% | 0.581481 pp |
| identity / predicted_identity_mahalanobis | 90.451852% | 98.651852% | 10.539683% | 18.879480% | 0.529630 pp |
| compact_oe / minimum_relative | 90.444444% | 98.900000% | 18.317460% | 30.464941% | 0.537037 pp |
| compact_oe / predicted_identity_mahalanobis | 90.492593% | 98.892593% | 14.634921% | 25.195153% | 0.488889 pp |
| compact_oe_structure / minimum_relative | 90.433333% | 99.048148% | 16.920635% | 28.507366% | 0.548148 pp |
| compact_oe_structure / predicted_identity_mahalanobis | 90.433333% | 98.870370% | 12.476190% | 21.927290% | 0.548148 pp |

六法known42通过、unknown整体/五RX/六景12组90%均未过。主MD−同预算compactOE：known正确41g/57l净−16，unknown拒对151g/219l净−68；主MD−identity：正确净−5、unknown拒对净+61。source heldRX1NN为identity78.793651%、compactOE78.539683%、main79.142857%，只源结构诊断而非未来注册AN。结构项末100步MSE0.058455 vs无结构诊断0.276159，未带来本轮拒识收益，不能据控制较高换主。

完整2000步和10份stdout已解析，两个learned的1000/1000批次逐元素相同；各120000L/128000OE曝光、unique6300/6134，全部loss/grad finite。NN只有g20704参数更新，F/defaulthead/G固定；sourcefree部署exact4files/四参数tensor/四聚合/12τ，无逐样本source行。NN合520648，g tensorpayload82816B、aggregate418560B、12τ96B。生产g605350forward，F6仅actualF0syntheticG10P0 smoke，真实IQ/F/G0；独立复核额外forward另计，父F30152和历史G恢复成本保留。

数据始终同G10P1的L6300/OE6134/V27000/novel3150，source5RX/三日/historicalnovel仍非跨未见RX/新TX盲测；L旧TX/RX及OE成员用于g、V只τ，OE TX/K和target U/Q/正式RX24-18/原Us新读取0。原F0closed90.981481%为父复用，所有身份仍原headargmax，不把拒识g当新身份head。无新DA/发现/未来注册结果。

原codeefb82a4177e9b59533b7e71408ded6b63fc8733a/preregd2638af643fac0f788a41b6901c8d85ee3065e4c/archive963e23c6fa33af39a2f3767c8a46bdedeb699c317590560bc40f0ae4f75214cf（356files/data0）发布/compile/runtime/首步actualF0 smoke VERIFIED；21tests47subPASS13.07s/一次独立P0P1。原driver3178144、source3178462/CUDA2/350MiB与3178464/CUDA3/358MiB、main3179142/CUDA2/360MiB独立实时核实，均采样非峰值；日志完整增长、81.408632s原pipeline失败保留、终态live/ownedGPU0、foreign0。批量g/CPUscore/复核时间非B1部署时延。

compactOE失败发生于CPU FP32公有模型重建，private truth前7/27000略超原unit容差；原FP32weights/units均finite，非近零塌缩。只新20261030-phase3-star-metric-embedding-score-recovery-s2026092701-r01/code4211e8981e85fa26e0ce354267043ecb2116530b将独立参考计算升FP64，原tol1e-5/2e-6与rawmargin1e-10不变；原模型/预测/τ/两个旧score不动。恢复complete明确prediction旧code和scorer新code，实际knownmaxdiff1.1743978e-6/novel9.5367432e-7通过，全部公有校验后truthlast；g2actualcheckpointsynthetic smokePASS，新CPU reference30152/F/G0，PID3193705/log0→1647/终态live0/无GPU。恢复8tests23subPASS10.50s及唯一问题针对复核与原suite分开计；原失败artifact不覆盖、不重训或选择性重预测。

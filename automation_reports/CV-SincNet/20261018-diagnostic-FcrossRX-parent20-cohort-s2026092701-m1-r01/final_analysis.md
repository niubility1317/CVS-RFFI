# 实际 As20 学生父模型正确集合：源域读出完成

已完成唯一source-only冻结读出，无训练、U真值或目标访问。代码0913a798757ccb9f274c12d74abc843c322ea8bd；严格加载P43 B01 stage20/final_ssdg.pth的model键，epoch20/4440更新、scratch/final_only、默认cosine30头，900722参数/206状态键。未使用ema_model冒充父学生。

父模型在固定V1800 clean正确1765个物理样本。其余各行复用既有source_qualification预测，IDs/TX/RX/day逐项同序；读取同一full条件0的两个固定信道draw。每个sat场景分母3530=1765×2，不是3530个独立样本。

| 模型 | clean保持率 | mid urban保持率 | low urban保持率 |
|---|---:|---:|---:|
| PARENT20 | 1765/1765 = 100.000000% | 2573/3530 = 72.889518% | 2059/3530 = 58.328612% |
| NOAS30 | 1762/1765 = 99.830028% | 1546/3530 = 43.796034% | 1269/3530 = 35.949008% |
| AS30 | 1765/1765 = 100.000000% | 2608/3530 = 73.881020% | 2112/3530 = 59.830028% |
| MASS_UNIFORM | 1764/1765 = 99.943343% | 2622/3530 = 74.277620% | 2129/3530 = 60.311615% |
| OBSERVATION_WEIGHTED | 1764/1765 = 99.943343% | 2624/3530 = 74.334278% | 2125/3530 = 60.198300% |

weighted相对uniform的mid/low保持率差为+0.056657/−0.113314个百分点；相对As30为+0.453258/+0.368272。weighted与uniform均在父clean正确集合新错1个；As30未新错。low保持率超过60%不是整体或正式low urban准确率达到60%。正式性能和E50资格仍按主P66已固定sourceFAIL处理，该读出没有新增选模门槛。

先执行2个零输入真实CP no-query smoke，再执行1800clean＋10800cached satellite＝12600 source NN，合计12602。模型state/parameter/gradient冻结核验通过，train counter0/eval counter12602；无教师bank、头拟合或新信道生成。function wall2.529965秒（smoke0.421874/source1.463905），独立启动至complete产物3.840106秒。CPU按保存预测计算cohort新增NN/IQ/CP/更新0；cohort_r02补齐了缓存NoAs30参照，原cohort_r01保留。

7个纯toy及4文件编译通过，一次限定P0/P1评审通过。release archive本地/远端比较一次、25个Python文件远端编译通过。分配GPU7；该短任务在首次postcheck前已完成，没有捕获到活跃GPU PID快照。完整predictions、smoke、complete、输出日志及计数证明实际完成；PID已不存在，未重跑来补截图，也未触碰健康P66/其他owner。

[readout元数据](E:/type10-7/automation_reports/CV-SincNet/20261018-diagnostic-FcrossRX-parent20-cohort-s2026092701-m1-r01/readout_complete.json)；[cohort摘要](E:/type10-7/automation_reports/CV-SincNet/20261018-diagnostic-FcrossRX-parent20-cohort-s2026092701-m1-r01/cohort_summary.json)；[源预测](E:/type10-7/local_artifacts/cvs_FcrossRX_parent20_cohort_20261018_r01/retrieved/predictions.npz)；[完整分层CSV](E:/type10-7/local_artifacts/cvs_FcrossRX_parent20_cohort_20261018_r01/retrieved/cohort_r02/cohort_metrics.csv)；[完整分层JSON](E:/type10-7/local_artifacts/cvs_FcrossRX_parent20_cohort_20261018_r01/retrieved/cohort_r02/full_cohort_metrics.json)。RX/day/TX/state/lock、空层、小层和margin/30均保留。只有一个模型seed，两个draw不是跨seed复现，未建立物理损伤因果。

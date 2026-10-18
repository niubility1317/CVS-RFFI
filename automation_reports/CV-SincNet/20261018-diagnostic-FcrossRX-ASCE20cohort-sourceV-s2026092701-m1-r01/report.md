# 实际AS_CE20父模型：源域正确集合读出

状态：ANALYZED／VERIFIED_SOURCE_READOUT_COMPLETE。仅源域冻结模型诊断，无新训练，无正式测试。

P68真正父为SAT18 AS_CE20；现有P43 As20缓存不是同一模型，不能替代。未找到AS_CE20 V1800缓存，因此补一次固定父默认头读出，取得真实clean正确集合及两个draw三practical保持率。它不参与P68晋级、不修改正在训练的模型／代码／数据或两个源判据，不成为新的gate。

唯一行固定ASCE20 student/model：SAT18 run20261018-phase1-FcrossRX-satelliteDropout-e30-s2026092701-m2-r01、AS_CE-s2026092701/stage20，scratch20／4440更新／final_only、完整L6300／hiddenU56700／V27000；P68实际父加载已经核实EXACT_MATCH与ownEMA，本读出不使用EMA前向。原F900722／206键／默认cos30 eval，不拟合头、不反向／optimizer／EMA／状态更新。

只读同PANEL V1800及已有full practical observations[draw,scene,0]。新增12602 student-evalNN：CPzerosmoke2，clean1800，2×3×1800satellite；0新物理信道调用、0U真值／0target。两draw不是两modelseed。输出真实父predictions.npz与aggregate保持率／逐draw表／complete，原始IQ／权重／预测不上Git。

CWD PROJECT/releases/cvs_FcrossRX_ASCE20_cohort_sourceV_20261018_r01，命令 python -u experiments/cvs_asce_parent68/runner.py --device cuda:0；现有普通N607CVS-RFFI环境／FP32／CPU2，计划GPU1，含其他owner每GPU≤2进程。不可覆盖root PROJECT/runs/20261018-diagnostic-FcrossRX-ASCE20cohort-sourceV-s2026092701-m1-r01，唯一owner codex/root/ASCE20-source-cohort68。继承已发布P68代码f520ae98791a80775b6b5923183fcffba7e91c73与其固定SAT18来源；无热改。

actual code commit待冻结实现后填。焦点负例与一次仅限新诊断直接缺陷的P0/P1、Git固定／发布／单归档比对／compile、CPsmoke2后连续读出，按现有最小流程执行；复用当前任务已核实的普通连接与源句柄，不重复数据校验。错误来源／非有限／状态变化／count不符／碰撞只停本诊断并保留，不fallback、不自动重试，不干预P68。

本增量12602NN与主P68 expected4352336NN分别列账，合计4364938NN。该机制读出不计作性能提升或新训练。实际结果完成后补充。


实际代码 54fe00dce90ae24fa80ce0ed427348e18a825051 已提交／推送并独立OID一致；4文件单归档SHA ec51f14408a51541ff6e6cc108e17aada2b52867b71e1cea1e7fbed82bfbec62 比对与远端compile VERIFIED。已有7puretoy／4compile与新诊断唯一P0/P1均PASS；不重审P68。

真实父clean正确1767／1800。下表是这个固定父正确子集的source保持率，不是正式测试准确率，也不是新模型收益；satellite pooled分母3534＝1767×2draw。

| 源场景 | 固定父正确集合观察数 | 保留正确数 | 保持率 |
|---|---:|---:|---:|
| clean | 1767 | 1767 | 100.000000% |
| practical_low_suburban | 3534 | 3171 | 89.728353% |
| practical_mid_urban | 3534 | 2584 | 73.118280% |
| practical_low_urban | 3534 | 2073 | 58.658744% |

实际12602 student evalNN，0teacher／train／optimizer／headfit／新信道／target，默认头scale30、900722参数／206键、state逐tensor不变。smoke2 PASS后连续完成12600 source；墙钟2.686243秒，其中源读出1.570706秒。NPZ独立CPU重数一致，所属PID已自然退出。短任务在首次/proc读回前结束，因此没有实时PID/CWD/GPU快照；不伪造该运行期证据。

完整六文件在本地 local_artifacts/cvs_FcrossRX_ASCE20_cohort_sourceV_20261018_r01/retrieved_r01，原预测NPZ／日志不入Git，公开聚合见[source_readout_summary.json](source_readout_summary.json)。模型角色与固定source观察证据完整；实时进程快照限制如上。后续P68与SAME在此真实父集合上的比较是0NN源数组计算，不参与晋级；主P68运行及判据未改变。

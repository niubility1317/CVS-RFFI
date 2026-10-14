# SIG_TIME 浅中层有界身份排序首种子矩阵

主假设：对实际t1/t2时序特征提供真实L标签跨RX身份关系，满足margin=.1停止辅助压力，比普遍对比目标更利于最终融合特征。直接使用4有序池化格与每包L2，无新网络/分类头/拟合投影，区别另一分支joint160原型与历史channelmean/stdSupCon。来源为报告31957f78和source-layer-order全L6300，不以源几何或目标高分选择checkpoint。

首seed2026092701四row：旧门控cos历史fixedfinal50；SIG_TIME ZERO/LOCAL_SUPCON/LOCAL_ORDER三新scratch50，同初态/defaultjointhead/batch_neighbor gate/ownEMA/cos/33L+17U/11100steps。分类头原CE/U联合训练，无单独训练/换头。每row固定final50后自动clean168000预测并独立truthlast评分，total672000decisions/72组指标。新增150epoch33300updates，历史50/11100成本另列。

三新每epoch55次额外合法L90eval-with-gradient前向（batch4倍数），2750次/247500包每模型，总742500额外源包。固定sourcecohortshuffle2026210712，70个平衡90包rankcohort循环，三臂曝光相同。原L采样和U gate不变；aux只L、无U关系。每triplet同day，正负参照同RX而anchor跨RX；1800ORDER项包括满足零项，360参照SupCon组含1同TX正+5异TX负。t1/t2均值，固定coeff.1/tau.2/margin.1/5epochwarmup+10epochramp，不扫描权重或margin。

额外forward恢复所有module mode和CPU/GPU RNG，不改变mainDropout计数；classW辅助梯度为None。实际观察epochs1/17/33/50全部55额外cohort，t1/t2/t3真实参数grad；mainL128CE与auxL90不同包，不冒充同包或累计AdamW贡献。rawPT留远端保存后复算，公开统计。SupCon和ORDER不是强度匹配，结果不能纯归因满足即停止。

先完成SOURCE资格再读targetsummary：三finalLCE<=.5，actualfθupdate/export/W独立/initial函数同，ORDER-SIGL720zL2>=1e-6，2750extra/220gradientcohort预算精确，17/33/50actualORDERt1/t2非零。仅SOURCE PASS原样扩预定义02/03；首seed不算84完成，无target调参/选模/重排/选择性重跑。

matrix、合法源角色/physicalIDs/seed/GPUowner/CWD/命令/不可变root/stoprule/产物在experiment.json。每GPU最多3总进程且free>=12000MiB，计入其他owner；不得热改或停止健康任务。仅一次直接P0/P1，单次archiveSHA、remotecompile、realCPnoquerysmoke、postlaunchPID/CWD/argv/GPU/loggrowth。

LOCAL_VERIFIED：13负例、三初态参数和暖期梯度一致、vectorizedSupCon等价、真实nativeaux进t1/t2且t3/W为None、defaultheadjointupdate、mode/RNG/buffer恢复及REMOTEcompilePASS。尚无新训练性能结果，84目标未达。

实际预测readout为ORIGINAL_HEAD，直接使用checkpoint默认头；核心实现commit 26bfdf044caa1be3c966370248c3bc3372009943，执行commit由release记录。

实际发布执行 f34078d1a0775ef6fa66365689481ad5a1932e4c，三个scratch模型均核实PID/CWD/argv/GPU及日志增长，GPU1/2/0，当前三者第3轮，额外拟合头关闭。历史oldgatecos默认头参考row已复用原产物完成truthlast，未重训。singlearchiveSHA比较/remotecompile/realCPnoquerysmokePASS，新目标成绩pending；健康训练保持原冻结配方。

首seed完整ANALYZED：ORDER83.6238%、SIG83.5452%、SupCon83.4583%，ORDER−SIG+.0786pp、修正538/引入406。200epoch记录/33300newsteps/8250auxsteps/672000defaultdecisions/72metrics/1980层cohort梯度复算VERIFIED；0新headfit，W独立精确导出。完整报告final_analysis.md/results.json/results.png；三seed仍pending，84未达。

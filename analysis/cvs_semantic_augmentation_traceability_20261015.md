# 真实标签跨视图身份排序：完成证据与未完成需求

源设计为用户31cc8a22附件。固定SIG_TIME、默认头、原U机制；r02三条新训练全部完成，原A0/A-J控制复用。预登记源条件FAIL，未扩seed。HQ70.97%只属不同源随机流的冻结诊断，不作为训练或正式测试结果。

|ID|需求及主要实现位置|状态|实际证据／边界|
|---|---|---|---|
|S01|固定SIG_TIME／默认头，原联合训练|verified|863730总参数／195041身份梯度；完整参数写回含原头和gain，未另训头|
|S02|实际5580/720物理折及控制来源|verified|实际A0/A-J same-fold scratch final50原头与真实CP smoke；五source契约相同|
|S03|D0保留原generic增强/U/EMA，仅关闭satellite|verified|r02完整resolved native args use_aug=True；原r01 no-generic D0排除主比较、保留自然E50诊断|
|S04|三场景真实residual，正式IQ不变|verified|adv3b02_xuc实际模块__file__、EQfalse G1/post_sync/refSNR30/reflect；共同源与原正式六practical|
|S05|仅真实L关系，同TX非负样本|verified|objective mask／隐藏U负例；实际全步U truth0|
|S06|正负参照同RX/day，anchor跨RX|verified|显式triplet单测及冻结源精确分组均值对照|
|S07|有界margin满足即停及归一化|verified|.1 hinge、最大2.1、逐anchor/fullbatch；满足/无参照负测|
|S08|两端anchor梯度、D4半和|verified|真实配对图两端梯度非0；冻结eval未detach D4 clean|
|S09|额外W/gain0，CE仍联合更新|verified|全50轮边界通过，参数更新组均非0；原头无独立拟合|
|S10|同source epoch5全局.1强度规则|verified|旧2160前向/0更新；D3 .6863719838，D4 1.1577435581；r02原样复用、新校准0|
|S11|E50/11100及旧U/EMA计数|verified|新33300全步，原控制22200已验证步复用；D0主4769400，其余6190200；50 native epochs和实际clip写回|
|S12|3新模型＋2精确旧控制，不重训练/测试旧控制|verified|注册矩阵、原scores/CP/fold；新/复用成本分开，原标识保留|
|S13|逐row原头七视图，独立truth-last|verified|全5条件35视图、5880000预测／630指标（新3528000／378），7视图固定后独立recount VERIFIED|
|S14|clean对D0、low对D1双增益及跨seed|pending|正式D3/D4 low53.4768/53.3542，clean80.6131/80.6107；两者clean低于D0 81.9798。源条件均FAIL；按规则未扩seed，跨seedNOT_RUN|
|S15|真实梯度/clip/更新/分支信息与使用量归因|verified|完整native日志＋冻结36722前向/83.073758s/0更新，五student/ownEMA不变；9真实图逐张查看|
|S16|Bridge-CE/S同成本局部教学|deferred|HQ物理源诊断完成；新训练未实现/发布，不默认堆叠到失败语义候选|
|S17|独立跨RX教师／学生KD|deferred|E50两教师core草稿未发布；另对话E20研究为不同数据折和预算，不直接复用权重|
|S18|共享容量锁、全owner和独立读回|verified|训练发布63f6bd461，原GPU1/2/3进程自然结束；源发布f172f98d6，PID3116403自然结束；无健康任务修改|

计数：verified15、pending1、deferred2、rejected0、blocked0。这是需求证据覆盖，不是性能目标达成。

训练代码/配置commit为63f6bd461834b4def6e3d7cfe4d8e8f00389b13e；源机制发布f172f98d67b19c65d7246b025a7ba2af770d197a。实现位置为experiments/cvs_urban_semantic50/{objective,source,observer,model,dispatch,evaluate,report,_audit_worker,full_log_audit}.py和experiments/cvs_urban_semantic_mechanism/{probe,checks,figures}.py。

匹配恢复8检查PASS6.717s；冻结源5检查PASS.831s；全量日志3负测PASS.066s；compile、一次archive传输比较、真实CP无query smoke通过。一次独立P0/P1及对应直接问题定向复核已通过，无额外报告/未变代码复审。实际检查命令见本轮report；其中`-m experiments.cvs_urban_semantic50.audit_checks`验证完整步数、非有限/隐藏truth负例和未完成矩阵不读取行分数。实际`full_log_audit`和`figures`命令已在完整真实artifact上执行。

严格对应附件：真实L、同条件正负参照、同次前向stopgrad keys、双向半和、原头和gain额外梯度边界、不改U或formal IQ。预定近似：一次源全局梯度比例校准、物理q和eval梯度观测；eval不是Dropout/clip/AdamW的实际写回。全体双向参照均值近似相同源于平衡集合对称性，不冒充两个方向的独立训练效果。

最高风险已由实际结果暴露：几何差改善未转为足够的默认头间隔/正式低urban收益，clean仍有约1.37pp代价。源low BLOCKED仅多7/386正确、<3dB仅多3/206；不据此宣称恢复全部损伤、唯一因果或可训练上界。当前只有一个模型seed，复现性未证实，low≥60%未达成，目标保持active。

原r01 D3/D4首步CPU元数据/CUDA标签错误：0完整epoch、保留失败产物；原r01 D0 generic关闭与控制不匹配，在读目标分数前排除，健康自然完成E50/11100、2327.915211s，额外成本单列。r02重新从零训练正确三行，没有停止/重启健康行。出图物理RX字符串排序错误已用实际数据修复，不影响训练/评分。

主报告：automation_reports/CV-SincNet/20261015-phase1-sigtime-semantic-e50-manysig-s2026092701-m5-r02/report.md；源报告为同根20261015-phase1-sigtime-semantic-source-manysig-s2026092701-m5-r01/report.md。实际artifact分别在local_artifacts/cvs_semantic_augmentation_recovery_20261015及cvs_semantic_mechanism_20261015，原始IQ/CP/日志/图不入Git。

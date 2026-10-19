# CE_HALF 补齐三种子：固定 E30，仅 clean

状态：LOCAL_VERIFIED，尚未启动。

用户于 2026-10-19 明确要求“CE_HALF跑完整三种子”。复用 seed2026092701 的完整旧结果；仅补02/03各一条 CE_HALF，分别从原 P49 对应 B01 scratch20 的 student／ownEMA 追加10轮，总E30／6660更新。新增20模型轮／4440更新，不重训父模型、seed01或原As／NoAs控制。旧 source gate 不抑制用户明确指定的固定复现。

CE_HALF 保持原 .5 satellite CE，两次 logits 分别计算后平均，原全部 symmetric RDrop KL、ordinary/raw 监督、ORD128 CROSS_RX、旧U门控和自身EMA不变。F三路、统一Sinc、joint160、默认cosine30头与骨干正常联合训练。不得替换为新默认sat0.68或plainPL。每段重置AdamW／cos10，lr2e-4、wd1e-4、min1e-6，ordinary使用各自原20参数epoch13。

两个父模型已只读核实 epoch20／4440、scratch／空祖先、final_only、无target、source角色完全相同。实际checkpoint的student／EMA、seed和默认头由launcher首步一次无query smoke核实，PASS后直接训练。

数据为原ManySig split392005，L6300／U56700隐藏TX／V27000，源RX1/3/4/6/8、day1/2/3，6TX；无Ldev或heldRX。训练完整practical lowSub／midUrban／lowUrban等先验、原residual/post_sync/noeq／ref30dB／reflect，实际传播与接收流程保持。

按本对话最新范围，新增两行只测原clean168000，目标RX0/2/5/7/9/10/11、day0/1/2/3。固定FINAL30/default头后逐行自动预测clean，独立CPU exec在prediction固定后连接truth。seed01旧七视图保留，只复用clean参与三种子汇总，不把新结果称为三种子星地收益。

唯一owner：codex/root/ce-half-three-seeds。普通N607既有CVS-RFFI环境、CPU2；CWD PROJECT/releases/cvs_FcrossRX_CEhalf_E30_s0203_20261019_r01；命令 CVS-RFFI/python -u experiments/cvs_ce_half_replicate/runner.py。不可覆盖输出 PROJECT/runs/20261019-phase1-FcrossRX-CEhalf-e30-s0203-m2-r01，日志为logs/ROW.log。只用物理GPU2/3，每卡包括其他owner最多1个GPU worker且free>=12000MB，忙时排队；不停止、迁移或修改健康任务。

停止仅限错误数据／父来源／非有限／输出覆盖／不能启动或产生合法prediction的系统技术失败。保留已完成和部分产物，健康已启动行自然收尾，不派发后续待启动行，不自动重试；不按性能停跑或挑选种子。

35项新接线协议负例检查PASS，复制目标函数的值和梯度与原CE_HALF逐值相同。唯一独立P0/P1只读审查PASS；无多余审批或重复审查。direct ordinary N607预检已通过；一次release归档传输比较、远端编译、真实父smoke、PID/CWD/argv/GPU/日志增长将在发布时记录。

报告三个seed的clean accuracy／macroF1／最差RX、均值±样本SD、对对应As30和NoAs30的配对差，完整解析两条E10日志并记录实际成本。全部固定seed均保留。用户是在看到seed01 clean85.808333%后指定该复现；不能宣称新的独立盲测，不用新分数修改配方或选择性重跑。精确矩阵及artifact路径由experiment.json唯一维护。

发布已 VERIFIED：实际代码 ccd7a6afc5a0fd27ebf3498ff4f087c81290619d 已推送并独立比较远端OID一致；单次归档传输比较与远端编译通过，两个真实父默认头smoke共4NN／0IQ／0更新通过。调度器PID/CWD/argv独立核实，当前等待GPU2/3空位，尚未开始新增训练。

资源说明：GPU2/3仍被其他owner占用。已向用户请求仅本次允许使用空闲GPU4/6；未收到明确回复前保持原2/3队列，不改GPU或训练配方。

终态：FAILED_BEFORE_TRAINING。seed02在首个训练更新前因共享GPU争用退出；两个source目录均不存在，全部所属PID自然退出。训练更新0、target访问0，仅4NN旧烟测。原产物保留，由 20261019-phase1-FcrossRX-CEhalf-e30-s0203-m2-r02 替代；无科学分数或失败补零。

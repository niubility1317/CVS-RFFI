# 多信道实现：四种固定半监督损失的匹配源对照

PLANNED。一个学生 seed2026092701，四条 own 原 ordinaryPL E30 独立+20至E50/11100、13L37LU。只有数据支持改变，旧 fixed H/G 与三 fresh H/G 各使用五个完整 U epoch，不选最佳信道实现。原联合目标保持未达成；固定计划见 [目标文件](../../../docs/goals/PAMF_U_REALIZATION_SSL_20261026.md)。

| 方法 | 原 fixed 对照 | 额外两次输入及原损失 | 固定系数 |
| --- | --- | --- | ---: |
| GTaskMulti | FeatureTask/GFeatureHard | 同一G两次，特征硬CE平均 | .05941532657537612 |
| GGCLMulti | RestorationBridge/BridgeGG | G_k/G_(k+1mod3)，跨视图跨RX CL | .04677712061808677 |
| HGCLMulti | RestorationBridge/BridgeHG | H_k/G_k，同原跨视图跨RX CL | .04677712061808677 |
| GJointMulti | AsymmetricHG/GJoint | H_k/G_k，原G CE＋CL | .04221045401136996 |

复用原 task_extra、relation_extra、joint_extra，不改损失、梯度路由或系数，不重做V/目标剂量选择。原冻结三E30 raw全同类且各confidence≥.95的q/接受集合保持；硬CE取argmax，CL同伪类不同RX正对、不同伪类负对，joint160受理集detached均值/std floor1e-4、L2eps1e-4、temp.2、除2*完整U batch，全部沿用原primitive。

原完整T1 L、raw selfPL.95/weight1、默认头联合保持：SIG_TIME_X3/CENTER_25K、ordinary/raw RDrop、As真实CE且KL0、LS.01、浅中层ORD128跨RX.1。附加分支排除RX1、分类weight附加梯度0，featurefusion norm/gain仍训练。无EMA、新头、G/teacher/reference更新，U标签−1/tx_label_visibleFalse/metadata无y。RX1只是额外分支留出，原L/rawPL/E30/G见过它。

追加U epoch e=0..19选r=e%4，scene k沿原cumulative step%3；日程不消费原生RNG。r0只读旧fixed，r1..3只读已完成G-only缓存的新paired H/G。MultiCache检查原q/physical索引与每fresh IDs/顺序/合约/G/scenes一致，不重建数据或缓存。无G输出后的额外归一化、白噪声或IQ插值。

每row新4440步、U1134000、held226800、额外接受838780、额外学生FW2268000；每实现U283500/受理209695/calls1110。GTask的CE_G受理前向数1677560（两次平均），GJoint为838780（一次），另两条0；都不是额外独立物理U。GTask无CL，CL有效anchor/正负pair不与它强行等量。

首GPU阶段一个共享smoke，逐个真实own E30/defaulthead/fresh3执行L128/U256联合backward和直接aux-head-shield核实，optimizer/目标0，原L768/AsKL0和主头有限非零梯度。PASS立即四条训练，无重新校准。各row final后自动完整V27000八视图；旧PLControl和对应fixed控制只用冻结预测，保留逐包净纠错/TX/RX/day/held/trained及完整20轮曲线。native eval-only V健康检查允许，objective V读取/更新0。

全四条完成后仍用heldRX1 G三景≥+.3pp、overall V raw≥−.1pp；最高合格heldG，精确同分名字母顺序；无通过不扩种子/目标。本控制器无target输入/真值或自动目标访问。通过也不是原三个学生seed raw clean/raw六景/同G六景目标达成。

N607普通账户、既有Python/Torch2.1、FP32/TF32off、每worker CPU2线程，无安装。CWD PROJECT/releases/20261026-phase1-PAMF-URealizations-sourceHold-release-r01；命令 PYTHON -u -m experiments.pamf_channel_pl.u_realization_training dispatch --config该release/runtime.json。不可覆盖 PROJECT/runs/20261026-phase1-PAMF-URealizations-sourceHold-m4-r01，精简source_context无目标句柄。实际commit记发布runtime/submit/完成记录。

GPU0..7全局CUDA+pending<2且free≥10000MiB，最多4条本run train_eval；保留所有owner。一次archive传输比较/remote compile及独立PID/CWD/argv/GPU/stepgrowth。首技术/协议/非有限/coverage/budget/smoke故障停新派发，已启动健康任务完成；不热改、自动重提或干预他人。科学负结果跑完四条。

4项新聚焦检查/local compile及唯一一次独立P0/P1 PASS。代码/配置/登记/公共聚合入Git；IQ/权重/逐包预测/日志/runtime不提交。另计原三E30、旧H/G/q与新G-only缓存成本，无推理多G ensemble。

# RXTrust-P：按 raw 可信度和 H 分歧分配单标签/候选集合监督

状态：RUNNING。三seed真实原E20/fullPAMF/SIG1/defaulthead检查与12批源梯度校准全部PASS，三条CUDA轨迹正在自身E20+30训练。尚无完整P性能结果。下文排队记录为历史发布节点。候选在当前 RXMeanMax 任何目标分数尚未读取时固定，不依据其结果决定是否运行。三模型 seed 2026092701/02/03，已有 PL/Sup 和即将完成的 Max 控制全部复用。

原完整 PAMF、单 SIG_TIME、默认头、fullT1 L 与普通 raw PL 阈值0.95/权重1保持。每 seed 只从原合法scratch E20独立+10+20至E50/11100、13L37LU；不加载E50教师。训练仅原clean及同包星地H、源U已知RX，0 DURA。推理原始IQ单F，不增参数/头/视图/集成。

新增目标来自FastTrust的“证据决定监督类型”思路，不是照搬历史完整FastTrust。raw阶段仍用现有RXMeanMax全U软一致性；H阶段先复用普通PL实际raw接受掩码。若raw未达到0.95，继续原同H软一致性。若raw达到0.95，记其判决a，当前同包H判决b：a=b时集合S={a}，否则S={a,b}；对选中RX风格H的学生输出计算 -log(sum_{k in S} p_k)。不给歧义H强制唯一身份，也不把候选集合称为已纠正唯一TX。门控和集合只读当前eval/no_grad预测，无U真值/EMA/跨包重排。

四种源RX均值差物理作用的强度0.1、±0.5dB/±0.025rad，以及逐包同base最大KL选择保持，与Max相比只改H的新增目标。RX均值仍是混合响应，非纯硬件或身份保护保证。原U普通PL不改；额外64L软一致性仅detach监测，原L fullT1照旧。因此H损失变化不等于改变L/U全部原监督。

H仍按全部U分母求平均，不按接受池大小放大。每seed12训练L/U批、6raw6H、0optimizer，用同一公式alpha=min(1,0.1*median(primary_norm)/median(aux_norm))后444步ramp；无V/target调权。P的实际alpha可与Max不同，这不是严格等剂量归因实验，不能把任何收益全部归因于集合设计。

源先验检查复用RXMean-U/LU各三seed的原V27000×4已存logits，0新F/训练/U真值/target：候选集合包含真TX的总体三seed均值约99.883%，所有sourceRX条件的最低观察值99.565%；low urban的平均解析logit梯度范数约0.3174，对raw单标签硬CE约0.5170。它说明集合能保留当前源raw的可信信息并降低歧义H的局部压力，不是置信保证、实际网络梯度或U/目标成绩；这些是E50代理，不保证训练早期E20。完整108行条件统计见source_candidate_analysis.json，未据此新增风险阈值或拟合门控。

每模型30次完整U访问，raw/H各850500；H的singleton/pair/fallback三路总数必须850500、每源RX各170100，按真实raw伪类统计但不配额。与Max相同6660新增更新、候选eval6804000包、teacherU9355500包、studentU3402000包、原L5114880包与额外L852480包，合计18724860个F前向包；P不增加F前向，也不因此宣称训练免费或推理加速。

全部GPU任务等待当前Max整个run完成，纯资源依赖、不读其指标；发布CPU等待控制器不占GPU。随后own≤4/每卡own≤1且总CUDA+pending≤2/free10000MiB、总CPUscorer≤1，保留其他owner健康任务。首个实际E20/fullPAMF/SIG1/defaulthead和U-only梯度检查PASS后直接完成校准并续训，参数保持、0校准optimizer。技术失败停止新派发、健康子进程结束后保留全部输出，不因科学负结果取消seed。

每行E50后自动同V诊断、rawclean＋六practical/cross_rx126000＋cross_both42000，全部14组预测固定后独立truth-last。3模型×168000×7=3528000新正式判决；原默认头冻结读取，逐seed/均值/样本SD/同seed配对/逐景/TX/RX/day完整报告。原绝对rawclean≥85.98174603174603%、六景>72.0961970899471%及全部声明范围严格优于PL/Sup仍是目标，本F0小矩阵或任何源P指标均不构成完成。精确命令、数据/继承、输出根和停止规则见experiment.json。

发布前本地11项数值/梯度/路由/输入负例/三seed管道与原Max5项回归均PASS，5模块编译PASS；本候选唯一一次独立P0/P1审查PASS、无直接缺陷。N607普通账号一次preflight核实新root未存在、3份scratchE20无祖先且源契约/缓存匹配、原fullU bank复用、剩余磁盘充足；新P真实checkpoint梯度smoke仍待首GPU步，不能把本地结果称为训练成功。父完整checkpoint状态将在首步既有load_prefix严格核验，无重复数据hash/receipt/审批。当前Max健康任务和旧远端release均不改动。

执行commit `31d6ecbf8b2da46cfd70e7ca462093eee0e4b4df` 已推送并独立对照上游OID；一个archive传输校验VERIFIED，remote compile PASS。实际控制器PID/CWD/argv/UID与等待状态已核实，当前不占GPU，12job均等待前序Max整体完成。source缓存/均值bank/已有PL-Sup控制继续复用；没有重训对照、停止旧worker或修改Max release。见launch_verification.json。

等待控制器第二次独立观测确认日志由62增长至248 bytes，pipeline时间同步推进；排队状态不是死锁文件或提交返回值代替。GPU worker仍0，真实P-smoke未开始。

结果工具已就绪：`rx_trust_p_results collect/analyze` 仅在3行/12job完整结束后读取固定预测和既有评分，复用PL/Sup/Max共9参考，核查18组同seed物理顺序、3528000新raw判决。`rx_trust_p_audit` 逐条核算全部完成行step/epoch/stdout、H单标签/集合/fallback累计与当前计数、原raw门控、尾批124及每RX全部H覆盖。4项针对性数值/缺seed/错参考/未完成管道/路由计数检查PASS，见analysis_tool_verification.json。此为本地分析工具补充，不改变远端31d6ecb神经执行、冻结方法或资源依赖，不代表新P真实smoke或性能通过。

当前P三seed均已实际通过校准和有梯度训练；actual PID/CWD/argv/UID/CUDA、非零连续step和H单标签/集合/fallback计数已核实，见launch_verification.json。系数依次0.37245122389054863/0.38352031095117256/0.4053865999870067；每seed6个H校准批共1536U的三路计数完整，0optimizer/参数buffer保持。正式训练已完成至少一整H epoch并继续raw，H三路每行合计56700；前缀约646至657步仅作执行证据，不冒充全E50日志或性能。当前Max目标结果不回流P。

已存在36批校准的同seed/step/raw-H配对分析见paired_calibration_analysis.json；两版实际E20父路径一致，主梯度范数相对误差最大2.93e-9，选择KL差0，raw未加权辅助梯度近乎相同。P full-ramp参考比例raw1.66%至2.72%、H24.75%至26.96%；H反向余弦10/18（Max6/18），raw8/18相同。P同时改变H目标和校准后的raw/H剂量，不是严格同剂量的目标归因实验；只凭本校准不能预测或解释目标表现。原P不修改。独立PM只改变系数规则，三seed固定在本P任何target访问之前，并等待本P完整结束；它复用已有源梯度、0新校准F，无P E50权重或目标指标输入。

三seed E20→E30完整段已全部核验：6660条step/30模型epoch，原普通PL/fullT1 L/四候选最大KL/实际alpha/ramp/尾批124/P的三路H门控逐条符合；每模型56700U各10次，raw/H各283500，H三路合计283500且每源RX各56700，所有原calibration和真实parent smoke保持。见first_segment_audit.json。当前继续各自E30→E50，前段完成不等于完整P性能。PM独立排队只改系数、0新校准F；不热改本P。

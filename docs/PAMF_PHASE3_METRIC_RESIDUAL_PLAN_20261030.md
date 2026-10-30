# PAMF 独立度量残差与OE软结构：源侧候选

固定run：`20261030-phase3-star-metric-residual-s2026092701-r01`。唯一主M_LOE，匹配embedding控制M_L。原F0/default6head冻结，另学输出160维的开放embedding；没有新身份分类头、最大成员关系头、EMA或目标状态。原F0 density_ratio作为固定参考仅复用，不按本轮成绩换主。

## 数据与来源

训练只用已有source L6300和OE6134的原F0特征。L旧六类各1050、每类×RX×day70条；OE仅outside-six/publicRXday可见，真实OE TX/K隐藏。L/OE原rawfused160与四贡献来自protected-cache实际code7dd90671968ddf58aab5a5b302b4d56371c8b9c9。共同预处理为原fused160单位向量u，以及逐分支FP64单位化/拼接整体单位化后FP32的c640，零分支保0、全零报错。派生缓存不产生新物理样本或新的rawdata验证。

knownV27000（旧六类各4500）仅冻结后的固定known风险校准；source novel3150（19-15/19-16/7-1各1050）仅完整预测后独立truth-last。它们的原u160/c640/defaultlogits直接复用已完成contribution-density codeedc28ad00baace5e4b9567fbf3391154bf6c34b4的同次特征/输出。父固定fused160 control模型/τ/预测原样保留；没有R1或640密度模型权重进入新embedding。原Us56700、target U/Q、正式RX24-18不读、不回收历史query训练。

源RX1/3/4/6/8对应1-19/18-2/19-2/2-19/3-19，day1/2/3为2021-03-08/15/23，均已见；novel有历史开发曝光，不是新RX/TX盲测。全部postPhase1输入沿既有固定G10MIX01/P1：600km/25MHz/2.45GHz/residual/post_sync/noEQ/refSNR30dB/reflect，receiver2027/scene2026103001/channel2026104001，原一次unitRMS/单received保持。P0不是clean，本轮不新造IQ/信道、不读P0或clean实值、不加载或运行G。

## 网络、损失与采样

开放网络为`640→128 Linear→GELU→160 Linear`，102688参数，seed2026092701。最后Linear权重/bias精确0初始化。`z=unit(u+g_theta(c))`，u与c在forward统一FP32单位化；无0.1硬残差上界、无额外身份头。原F0仍E50/11100/232993、新native更新0。新网络只产生开放特征，所有部署旧身份仍来自原F0默认head，不能用新embedding类别读出代替。

L每步192条：六类×15公共RX/day格各2条，每类另加两个不同轮转格各1条，合每类32；格内无放回，同batch无重复。extra域为`(2*step+c)%15`和下一格，step从0计；L RNG2026103017。OE仅M_LOE训练读取，每15域4条＋连续四个轮转域各1条，共64；extra从`4*step mod15`开始，OE RNG2026103018独立。M_L不打开OE训练值，也不推进L以外随机流。训练L indices两arm完全一致；标签只为已知L，OE不按隐藏设备补齐。

两损失T=.1。对任一anchor，仅允许同时不同RX且不同day的candidate；自己自然排除。candidate基础质量为该candidate所在域样本数的倒数，加入softmax logits的log权重，使域先等权再按相似度分配。所有mask/candidate仅用公共元数据和合法L标签，不用OE TX/K。

L监督度量项：正样本是上述合法candidate中与anchor同旧TX的记录，目标概率先在有正样本的域等权，再域内均分。学生概率由`cos(z_i,z_j)/T-log(n_domain_j)`在全部合法candidate归一化；loss为目标对学生的KL（交叉熵减目标熵，与有监督对比同梯度）。anchor先在六类×15域格内平均，再格等权；生产采样保证各格有样本、各anchor有合法正样本，缺覆盖明确报错。

OE软结构项：teacher概率用冻结原u160 cosine/T加相同域质量，student用z cosine/T。teacher stop-gradient，KL(teacher||student)，anchor按15公共域等权平均。没有硬同设备正对、未知类别估计或“其他OE均为负类”的假设。teacher图只是一种软结构约束，不把它当真实未知身份。

M_L只优化L_metric；M_LOE优化`L_metric+OE_KL`（权重1）。两arm同结构/init/L采样/2220步，fresh AdamW lr1e−4/wd1e−4、clip5、常量LR、无早停。全部新增2220是开放网络步骤，不写成原F0续训或新训练epoch。第一/二步记录梯度路径、preclip范数与真实loss/特征漂移，完整2220步及索引保存。M_L是“embedding训练无OE”控制；两者后续背景统计均会使用OE，不能称全流程无OE。

## 首步、冻结预测和评分

共同prepare首步严格加载实际原F0，执行2条合成G10无query smoke，PASS立即导出既有L/OE特征对缓存；真实新增F0/G前向0。两arm在读取训练特征前对开放网络做不更新参数的合成梯度smoke，clear grad/状态不变；不把smoke算入2220更新。训练终点严格保存完整新网络和实际来源，原checkpoint/输出不覆盖。

每arm终点冻结后立即自动预测，不等待另一arm。严格加载实际新checkpoint，先做2条合成特征推理smoke；再映射全部L/OE到z160，按同一D160 Gaussian density_ratio规则拟合90known/32background/25Lloyd/seed3019/covshrink.1及径向收缩，均只用L/OE。分类器/骨干不重新训练或换头。

统计全冻结后读取复用V特征/defaultlogits，V仅known42/common-tail六predicted-class阈值；τ冻结后才复用novel特征并输出。固定F0 density参考列原样保留，原身份/ID/metadata逐条一致。部署仅102688新参数＋聚合密度/τ，不携带source逐行bank。原F0活跃参数232993加开放网络算术为335681；是否实际端到端加G、时延/显存须实测，不能由参数推定。

每row全部预测固定后由独立CUDAempty CPU scorer唯一读novel truth，报告宏A_K/R_U/H、C_K、同P1原F0损失、AUROC/OSCR、完整TX/RX/day/scene与固定参考配对。唯一主M_LOE必须known42＋整体source R_U≥90%早筛才另预登记跨未见RX星地，控制不能替主，source通过不代表最终目标。两arm结果不回流健康训练、预算、阈值或矩阵选择。

N607：共同prepare一个GPU worker，完成后最多两个train/eval worker，每卡新增本任务≤2、启动实际free≥4096MiB；单独CPU scorer一次最多1个。可以与其他用户共卡，不停止/修改/重启其他owner。一次代码commit、聚焦负测/实际checkpoint首步smoke、唯一直接P0/P1审查、最小登记/单preflight/archiveSHA/compile及独立PID/CWD/argv/GPU/log/终态，保留不可覆盖root。非法来源/角色、非有限、无合法预测、碰撞或系统技术失败保留，不按低性能停健康训练，不自动重启；UNKNOWN先核原PID/root。

本轮没有目标适应/发现/注册或未来AN。完整跨RX星地、最新v3 known保护及高拒识研究目标、无标签DA与匿名持续注册保持ACTIVE；source和代理软邻域不替代最终验收。

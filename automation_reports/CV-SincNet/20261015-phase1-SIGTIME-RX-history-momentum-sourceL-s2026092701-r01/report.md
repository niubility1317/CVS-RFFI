# 源 RX 对比过滤历史特征动量：固定源原型

状态 **PLANNED**。根据同父、同梯度的纯 CE 状态实验，保留尺度历史有利于默认头读出，但保存矩仍使 raw 跨 RX 身份关系下降。本轮只研究一个新操作：**保留二阶尺度和原当前梯度，用源 RX 梯度对比过滤历史一阶特征动量**。这不是清空全部状态、旧完整步风险投影或已关闭损失的系数调整，尚未证明泛化收益或完整文献原创性。

对同一原 L180 CE 前向，求源 RX1/3/4/6/8 各36包的平均梯度。main仍是原whole180平均CE梯度。按完整188梯度的原clip5系数c和各坐标saved t/v计算下一步prospective AdamW分母d；历史贡献u=beta1*m/(bc1*d)，对比h_r=(1-beta1)*c*(g_r-mean(g_r))/(bc1*d)。只将u投影到五中心化h跨度的正交补，再映回m。原AdamW随后正常使用当前g更新；v/step/weightdecay/W/其他参数状态保持。五行Gram用FP64，固定eigen cutoff max(1e-30,1e-10*maxeig)，rank至多4。记录理想投影和实际FP32状态映射残差，不把数值内积当身份收益。该坐标度量只适用于固定CENTER25K chart。

对照KEEP（保存状态）、M_ZERO_F（只清feature一阶m）、RX_M_FILTER（结构过滤），全部LR2e-4及同父状态/同梯度。f_theta含原Sinc/时频/PA/reference/归一化/3融合gain的71tensor/195041coord，实际classW不滤且正常联合CE。临时188参数after完整复制，真实joint160保留after3gain；ACTUAL用afterW、PRE_W只用beforeW分析同一z，无分类器拟合或替换。

固定own BASE_CE第一seed2026092701 final20，父执行6693e4480ef990bc23c5021ff1440b547fe5fcaf；source_proof先核对实际数据角色、完整scratch来源及fixedfinal20冻结，再反序列化，不因同名权重直接继承。原L6300，7/63/30角色，6TX/5RX/3day。四currentL180采用原90cell的ordinal0/1、17/18、35/36、52/53；held180采用68/69。900包native5ID和全局baseindex互不交，held曾被父训练见过，只是独立当前步的源评价，不是盲测或未见RX。U/V/target IQ不读，U真值不读。

固定晋级要求至少2/4 cohort同时满足：FILTER对KEEP及M_ZERO_F各有实际F差norm>1e-6、heldjointRMS差>1e-5、至少1浅中层tensor变化、raw身份关系间隔增益>1e-5；FILTER自身间隔比before提高；ACTUAL与PRE_W meanCE相对before及两对照均不得恶化超过1e-6。rank须非零。raw关系定义为同TX跨RX同day余弦减异TX同RX同day余弦，不等同classmargin。只缩小步、只让投影成立、只有默认头置信度改善都不能晋级。阈值和rank规则在真实源结果前固定；失败关闭此固定机制，不扫LR/rank/强度/频率或选对照替代。

预算900uniqueL/6300metadata/3062packet网络前向（含2零IQ真实checkpoint烟测）/24临时CPUAdamW、0实际训练/头拟合/EMA/U/V/target。真实CP noquery smoke PASS立即继续，所有live模型/buffer/grad/payload/RNG保持。技术不一致或异常保存输出并仅停本run，其他owner不变。

run `20261015-phase1-SIGTIME-RX-history-momentum-sourceL-s2026092701-r01`，owner `codex/root/RX-history-momentum-source-20261015`。实际GitHEAD写release_commit.txt；普通N607账户/CVS-RFFI Python/CPU2线程/GPU0，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_RX_history_momentum_sourceL_20261015_r01`，命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_RX_history_momentum_sourceL_20261015_r01/experiments/cvs_rx_momentum20/runner.py /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_RX_history_momentum_sourceL_20261015_r01/resolved_request.json`，唯一输出 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-SIGTIME-RX-history-momentum-sourceL-s2026092701-r01/source_diagnostic`，日志 `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_RX_history_momentum_sourceL_20261015_r01/source_probe.stdout.log`。完整环境、stop和artifacts见[experiment.json](experiment.json)。一次归档传输SHA比较、远端编译、进程/CWD/命令/日志增长核验及独立终态读回；没有新增seal/receipt/hash链。

如果源规则通过，才另行固定正式训练频率与E20+20纯CE同父同增量控制。当前没有发布正式训练或clean预测，目标测试不能反馈调参。SIG_TIME、CENTER25K、COMPACT_LINEAR、LONG_SILU历史正路线分别保留；84%默认头clean三seed目标仍active、未完成。源RX梯度差可能包含有用身份交互，过滤未必是去接收机因素，也不提供有限步风险保证。机制依据见[设计](../../../analysis/phase1_receiver_contrast_momentum_20261015.md)。

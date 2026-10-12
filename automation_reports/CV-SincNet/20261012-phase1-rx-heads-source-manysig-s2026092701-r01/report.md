# 源RX分类边界专门化：匹配随机分组的30轮源检验

前述waveform、pool/phase源消融均未过固定阈值，保留全部完整负结果，无target测试。新的假设是将条件信息用于训练分类边界，而不是消除身份特征或替换已有效的native骨干。此为源端待证假设，不按目标成绩选择网络。

每source heldRX1/3/4/6/8有RX_HEAD及RANDOM_HEAD，两组均从零native+4个cos专家头，joint160共用一次forward，全部原身份时频/PA/参考/内部门控/cos结构保留，无显式CFO标量、U、增强、teacher、EMA或其他辅助网络。fitL5040/heldV5400、整heldRX排除训练、原20%角色不变；固定30×222、L采样seed+101、原coslr/AdamW WD0.0001、label_smoothing0。

原global身份CE保留，再加0.25指定专家CE。RX按四fitRX分组，Random不看TX/RX/day，固定种子+713打乱fitL索引后均匀分成4组；每组1260、相同head/readout/loss/采样更新成本。专家初始用独立generator seed+161803、shape4×6×160、cosscale16，两arm初始状态对应相同，不改native初始化。180865可训练参数，较native177025增加3840。global与aux CE及梯度全日志保留。

推理不要求任何RX/域标签或其他query元数据，固定0.5 globalnative概率+0.5四专家softmax温度1的均值，所有六类/packet同规则。模型冻结后再与已验证native3固定概率按0.5/0.5混合；概率ID对齐后先持久化、再sourceV评分。新模型单独和混合都报告，不fit融合权重、温度，不做query更新、配额或全局重排。未知RX的专家推理平均是待证规则，不声称已识别接收机。

候选RX_HEAD固定资格：混合source均值>=native3+1pp，>=RANDOM混合+0.5pp，各至少4/5RX非负，两对照最差RX均非劣1pp。完整10row后才判，无合格者不访问目标。若通过，全源30同机制每sourceRX一专家，共5头每头1260 L；这个全源扩展在target之前固定，三primaryseed2701/02/03分别从零、每模型181825参数，复用其合法native3自身推理成员，总712900参数，成本另列。逐row固定后自动clean+六practical预测，全视图每168000完成后独立truth-last评分，不根据目标改权重、分类头或重跑。84待实际三seed完整证明，现有benchmark已暴露，不称新盲测/真实卫星部署。

新10学生66600更新，原native3历史成本另计；同参数/同readout控制用于隔离真实RX分组与随机分组，不把增加参数当机制有效。一次P0/P1、聚焦负测/真实matchingcheckpoint noquerysmoke、一次实际code/config Git提交、一次不可覆盖run/release/output、N607preflight/单archiveSHA/compile、启动PID/CWD/argv/GPU/loggrowth核查。普通账号唯一owner，总GPU实验进程<=3含其他owner/free>=12000MB，不改旧健康任务；仅系统技术/权限/输出碰撞失败停止所属run、不按分数停止。完整矩阵/环境/命令/CWD/路径/stoprule见experiment.json，无额外review/hash门槛。

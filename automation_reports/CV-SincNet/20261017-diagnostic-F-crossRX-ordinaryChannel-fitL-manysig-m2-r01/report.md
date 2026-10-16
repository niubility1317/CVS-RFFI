# F＋跨 RX：普通 AWGN／multipath 源因果矩阵

PLANNED，尚未启动。固定逐 row 矩阵和实际来源见 [experiment.json](experiment.json)，依据与边界见 [研究说明](../../../analysis/cvs_ordinary_channel_source_20261017.md)。四格 full／no_AWGN／no_MP／no_both，只以 no_both 为主候选，单项格仅归因，不回退晋级。

原生 DataAugmentation、build_aug_base_cfg 和 configure_augmentor_for_epoch 实际 E13；所有其余旧 IQ 操作／PA 保留、DAC／class-signature 关闭。原语执行后返回输入以移除效果，仍消耗原抽样；每格恢复同一 batch 起始 Python／Torch RNG，核对原语参数／前后 RNG／PA 强度，结束推进一次 full。私有 generator 重建 MP 随机参数只用于观察，额外原语标量实际计数；不修改源实现。无后置 RMS、PA 条件、gain clip、新信道或噪声。

MP 后是按当前功率计算的 AWGN、再是 PA；移除 MP 后其功率依赖自然重算，是总链效应，不声称固定噪声直接效应。既有 As 本来读取 raw，不接 ordinary；本轮无 As，不声称消除了同视图重复星地叠加。源四格交互项不等于最终 Ag／As 四格。

两份原合法 source-only scratch E20 final20 B00 默认头，在各自不可变后端加载一次，匹配 L5040／U45360／V21600、RX1 held、final-only／4440 更新及无祖先／target／头拟合。只用 fitL5040，RX3/4/6/8×day1/2/3×6TX，72 格各70包。源载体 L6300／未用V及历史特征反序列化，RX1先排除；U仅角色ID、无IQ/TX标签。不用 heldRX1／U／V／target 推断，不混同完整L6300或L_train5580。

四命名增强流 202610171701 至04，独立 shuffle 202610171710 至13，每抽样全5040一次。两模型每个 raw5040＋四抽样×四格×5040＝85680源NN；总171362 NN含首模型真实CP合成2包no-query smoke。161280源ordinary生成，另320合成canonical ordinary；kernel私有随机重放和影子原语生成／实际施加包数分开统计。新训练／更新／教师／另训头／target均0；参照历史40model-E／8880更新／9373356NN单列。

固定科学条件：两模型 no_both 相对full accuracy≥+0.2pp、macro-F1不降；至少3/4抽样正增益，每抽样≥−0.2pp；相对raw accuracy/F1≥−0.5pp、raw正确保留≥99%；每RX/TX/day相对raw≥−1pp、相对full≥−0.5pp。所有样本、格和分层保留。科学FAIL完成全部固定矩阵，再关闭 joint普通AWGN/MP移除，不挑单项、PA条件、系数、clip、抽样、RX或seed救跑。PASS只提供源身份依据，后续匹配scratchE20另登记，不自动训练，不冒称新增Ag、跨RX clean或As收益。

唯一owner codex/root/F-ordinary-channel-source-20261017；普通N607现有CVS-RFFI Python，CPU／OMP／MKL／OpenBLAS2，CUDA_VISIBLE_DEVICES为空、GPU none、两worker顺序，不干预旧健康任务。CWD PROJECT/releases/cvs_F_ordinary_channel_fitL_20261017_r01，命令 CVS-RFFI/bin/python -u -m experiments.cvs_ordinary_channel17.dispatch；逐row runner命令在experiment.json。实际私有账户／路径仅runtime保存，不入Git。

本地发布命令 F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_ordinary_channel17.publish，CWD当前managed工作树。Git commit为代码身份，实际值写release_commit.txt/resolved_config。唯一不可变输出 PROJECT/runs/20261017-diagnostic-F-crossRX-ordinaryChannel-fitL-manysig-m2-r01；各seed.stdout.log及release/source_ordinary_channel.stdout.log。一个N607预检、一份archive及一次传输SHA比较、一个remote compile、launch后PID/CWD/argv/CUDA/日志增长独立读回。首worker真实CP smoke PASS立即继续，不增加授权token或hash链。

错输入／来源／数据角色／实际配置、RNG或参数不匹配、非有限、输出碰撞或worker异常保存failure/partial后停止，不自动重试或修改旧进程。local17输入/配置负例＋7科学反例、480合成ordinary／72私有kernel随机标量PASS，0源IQ/CP/NN/更新；原生canonical、AWGN功率依赖、PA匹配及函数恢复已验证。NumPy桥警告未使用、不装包。一次独立P0/P1及直接缺陷的一次定向复审（如需）见review.json，不新增审查层。

保存 root request/active/launch/result/completion、逐模型实际native args/调用文件/来源、raw与四抽样四格完整logits和hard预测、320源batch的逐包PA/尺度/参考变化与随机原语参数/功率依赖、真实CP smoke、完整日志、独立全171360源判决/CE/F1/分层/72格/交互项/修复新错/成本复算、图与同row报告登记。IQ、权重、原始判决／物理ID／运行态不入Git，聚合元数据可复核。没有target预测或truth评分。

完整联合目标ACTIVE。RMS、PA bypass、shift bypass及其他已关闭固定配方保持；本路线区别于另一对话practical正负噪声训练。

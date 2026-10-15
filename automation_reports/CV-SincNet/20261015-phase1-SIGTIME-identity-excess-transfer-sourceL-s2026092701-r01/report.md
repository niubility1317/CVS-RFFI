# 异 TX 接收变化的相对身份风险迁移

状态 **PLANNED**。当前目标是学习能跨 RX 使用的实际 joint160 身份特征，保留原默认分类头正常联合训练；84% clean 三种子均值尚未达到。这是本轮第二个核心假设，失败后不再增加第三个假设。此前绝对概率迁移 INITIAL 和 FINAL20 各只有 1/4 通过全部源条件，固定候选关闭，不改变阈值。它仍留下明确证据：FINAL20 跨身份 RX 差分产生 316 至 462 个 active 判定、81 至 107 个不同 origin，辅助与主 CE 的 F 梯度余弦 .407 至 .688；初期则接近 .99975，主要放大普通分类方向。

本候选改变监督内容：**只修复借来的接收变化比真实目标 RX 同身份样本新增的分类风险。** 同 day、真实有标签 L，a 是 RX-r/side0 六 TX，b 是 RX-s/side1 六 TX。对 origin y 与每个 c≠y，正向 virtual=a_y−a_c+b_c，实际参照 factual=b_y；反向 virtual=b_y−b_c+a_c，factual=a_y。c 遍历全部 5 个异 TX，不先平均差分、不按结果选参照。目标来自 origin 真实 y，factual 也为真实同 y。物理包不假设相同报文。类相关接收作用、非线性及 packetnoise 都可能使迁移失败，所以这只是可检验的特征训练任务，不是硬件反事实或精确 RX/TX 分解证明。

用原 cosine W.detach、scale30、eps1e-4，固定 loss=.1×mean relu(CE(virtual,y)−stopgrad(CE(factual,y))−1e-6)。1e-6 是数值容差，不扫参数。ReLU 在相等处梯度为零；停止风险后不继续要求置信度达到 .9。virtual 的三个实际 joint 端点与三个实际增益完整反传，factual 风险和 W 停止梯度；mainCE 正常联合更新原 W/骨干/增益。factual 在其他 virtual 角色仍可能收到梯度，不能声称它的参数不变。没有新头、拟合分类器、生成器、教师关系复制、原型或部署特征变换。FP64 解析判别及同 view FP32 比较单列，不称额外模型前向。

采用既定三控制 ZERO/SAME_RX/CROSS_RX。CROSS 是 20 个有序跨 RX 组合×3 day=60 个唯一组合；SAME 是 15 个同 RX/day 两 physicalside 组合重复 4 次，披露只有 15 个唯一组合。每 pair 双向×6 y×5 c=60 virtual，共 3600/control；同数 factual 判别逐项计算，不复用 180 个 factual readout，因此 7200 解析判别/control、两非零控制每 call 14400，FP64 与 FP32 各计算一次。固定 loss 分母 3600，不随 active 数变化。每个 packet 作 origin/source-reference/destination-reference 各 20 次，groupendpoint 各 4 次；active_origin 在 180 个真实物理包中去重，重复视图不冒充覆盖增加。RX、targetTX、referenceTX 的 active、风险、错误与真实参照 CE 分层保存。

源矩阵 INITIAL/FINAL20×4 currentL180×3control，共 24 单元。复用此前 900 L：current ord2/3、19/20、37/38、54/55；held66/67，物理 ID 不交。原数据角色仍 L6300/U56700隐藏/V27000，RX1/3/4/6/8、day1/2/3、6TX。父 FINAL20 见过 heldL，这不是未见 RX 或新的独立盲测；V 不参与梯度，U 不读取 IQ/标签，target 不读取。复用已验证数据 handles，不新增数据重验或哈希链。

固定父为 own BASE_CE-s2026092701，执行6693e4480ef990bc23c5021ff1440b547fe5fcaf，原随机 initial_smoke 与 final_ssdg epoch20。source_proof 在两个加载前核对完整实际角色/ID、scratch ancestry、固定final冻结规则，初始 scratch_only 和 finalactualargs 严格验证。没有按 target成绩选择权重或回退其他父。INITIAL 保留原 optimizeroptions、moments 为空；FINAL 保留 savedm/v/step。两者同预定 LR2e-4/globalclip5/188parameters。各真实 CP 做 2 个全零 IQ noquery smoke，PASS 立即继续。eval 模式，原模型/optimizer/CP/grad/RNG不更新；临时 AdamW 副本执行对照一步，after 全188参数进入真实模型，ACTUAL afterW 和 PRE_W beforeW 分析同一 afterjoint/after3gains，不换头、不拟合。

**真实结果前固定资格**：每 state 至少 2/4 CROSS 同时相对 ZERO/SAME 满足以下全部条件。

- current active≥180/3600，distinct active origin≥18/180，辅助梯度非零，abs(auxF/mainCE cosine)≤.95，避免仅放大原方向。
- held PRE_W 的 mean positive excess risk 差≤−1e-4，且 raw virtual meanCE 差≤−1e-4，transfer 错数不增。后一条件防止只降低 factual 身份置信度就让 excess 变小。
- 原 held ACTUAL/PRE_W 准确率不降，CE 相对各控制的增加≤1%；门槛与旧候选不混用。
- actualF 差 norm>1e-6、heldactualjoint RMS 差>1e-5，至少一个浅/中层 tensor 改变。

FINAL20 合格优先固定同父 E20＋20；仅 INITIAL 合格才 scratch20。该优先级依据已经完成的前项初期近共线、晚期新方向证据，在本候选数据前固定。source PASS 只说明值得真实训练，必须继续发布固定预算的 CE 纯基线、原隐藏 U 旧门控 SSL、SAME/CROSS，原默认头 joint、final-only、每row自动clean预测和独立 truth-last。续训必须同父同20增量 CE 对照、完整继承核对及真实 optimizer状态说明，累计40不能冒充scratch20。此前绝对概率候选不按本规则重判；源不通过则关闭固定 loss，不扫容差/λ/LR/参照/强度。

预算 900uniqueL/6300metadata/6124 网络 packet 前向（4 smoke＋2×[180before＋4×(180current＋3×180after)]）、48 临时 CPUAdamW、60loss calls、864000 FP64＋864000 FP32 解析判别；0 实际训练更新/头拟合/EMA/U/V/target。风险降低不等于原身份识别提高，局部一步收益不等于完整20epoch，更不等于未见 RX 泛化。Sinc kernel 比较是 whole CE+aux proposal versuspre，不冒充纯 aux effect。

唯一 run `20261015-phase1-SIGTIME-identity-excess-transfer-sourceL-s2026092701-r01`，owner codex/root/identity-excess-transfer-20261015。普通 N607/CVS-RFFI Python/CPU2/GPU0，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_identity_excess_transfer_sourceL_20261015_r01`，命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_identity_excess_transfer_sourceL_20261015_r01/experiments/cvs_identity_excess_transfer20/runner.py /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_identity_excess_transfer_sourceL_20261015_r01/resolved_request.json`，不可覆盖输出 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-SIGTIME-identity-excess-transfer-sourceL-s2026092701-r01/source_diagnostic`，日志 `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_identity_excess_transfer_sourceL_20261015_r01/source_probe.stdout.log`。actualcode/config Gitcommit 写 release_commit.txt；一次 focusednegativecheck/唯一 independentP0P1review、一次 preflight/单归档 SHA比较/remotecompile、一次 PID/CWD/argv/GPU/loggrowth核对。权限/来源/非有限或技术错误保存产物并只停止ownedassay，先核对原状态再恢复，不触碰健康任务，不新增审批/签名/receipt。matrix/stop/artifacts 唯一登记见[experiment.json](experiment.json)。

跨类别借用 delta 已有[Delta-encoder](https://arxiv.org/abs/1806.04734)，relative risk 与本仓库 sharpness gap 思想相关；不能把这些概念本身称首创。这里待检验的组合是同day真实 RX差的跨身份功能迁移、真实同身份目标RX风险作为停止门槛、实际加性 joint 三端梯度和原默认头。跨 RX 射频指纹分离也已有[RIEI](https://arxiv.org/abs/2411.03636)；本任务不声称证明可分离条件。前述报告的集合监督、观测预测、PA重组后续完整负结果保留，不重新打开；结构/CENTER25K等历史正收益也按各自同row/预算保留，不能相加或按target最高挑父。

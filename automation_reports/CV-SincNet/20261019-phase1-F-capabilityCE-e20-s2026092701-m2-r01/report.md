# 纯 F E20：教师判对信号分配 As 真实标签 CE

状态：PLANNED，尚未发布，没有新性能结果。目标保持 low urban >60%、clean 相对同预算 NoAs 可重复正收益，分别报告各 suburban 与同一模型 clean 的差距。

唯一假设：对同一受损 IQ，冻结 CSIL 能正确识别、而 F 尚不能识别的样本，可提供“哪条观测已有另一模型的判别见证”这一位训练信息。固定全部 As 和 CE 总份额，把部分 CE 权重分给教师判对观测，检验它是否优于同数量随机分配及普通 As。教师答错不代表物理不可恢复；教师答对也不保证 F 能学会。所有合法 As 始终保留真实标签 CE，不设学生正确/高置信前提。

源依据来自已保存的 V360 logits，0 新 IQ/前向/更新/拟合/目标访问。两 draw 的 mid/low urban，CSIL 判对而 F B1 错的样本数分别44/72/65/76，教师错而学生对18/21/11/16；观察总数1440，教师准确率优势13.263889个百分点。它只说明二者判别能力存在差异，不证明本权重有效。另一已完成源诊断 `20261019-diagnostic-F-observedChannel-e20-s2026092701-m4-r01` 的实际 complete/analysis 产物为180个合法L、5761前向、0更新/0target；B1训练命名流确定性重建 low urban 45/69=65.217391%，同L新draw配对89/138=64.492754%。这是有限子集和确定性重建，不是独立历史IQ字节复核、全L拟合统计或物理上限。

区别与去重：历史 CSIL 全 posterior、非真类 posterior 与输入 Jacobian 方案保持关闭；本轮不匹配概率、confidence、特征、Jacobian或某个非真类关系，只用 argmax==合法L真TX 的 bit 重新分配原真实标签 CE。旧 recoverable RX6 课程通过学生进展决定 physical-state 有限提案，改变了观测分布；本轮不改任何 As IQ、场景/状态分布、提案或日程。scene-risk 把权重集中到高 CE 场景后仍未满足源判据；本轮使用外部固定教师的逐观测能力信号，不重开该风险更新。旧 CE 温度、LS、SAM、权重/视图日程不扫描。关联聊天的 U practical 路由、额外 HQ 标签视图和待发布噪声一致性不纳入本轮。

固定纯 F=SIG_TIME_X3＋CENTER_25K＋CENTER_UNITS，900722参数、实际joint160、既有cosine6×160/scale30头，骨干与默认头联合训练。0 CROSS_RX、0 RDrop、0额外部署分支/头。两新行均同modelseed2026092701从零累计E20=13L+7原U，222步/轮、4440更新；不继承任何学生权重。普通增强/原native U及ownEMA/AdamW和cosine20、smoothing0.01、NoExplicitCFO及原内部补偿均不改。

完整 L6300/U56700隐TX/V27000，源RX1/3/4/6/8、day1/2/3、split392005、256点IQ及既有物理角色不变，无Ldev/heldRX。As仍从raw单次完整 practical residual/post_sync/noEQ生成，不先叠ordinary、不弱化信道、不筛除失败/低置信观测；训练场景与原B1相同（low suburban/mid urban/low urban），测试仍clean＋六practical。物理硬件session和训练随机流保持，V及target不参与权重拟合。

逻辑参照 B0=ordinary NoAs、B1=ordinary＋普通完整As，均只读复用 `20261019-phase1-F-nontrue-CSIL-e20-s2026092701-m4-r01` 的同seed纯F FINAL20、源读出及七视图预测/评分，不重复训练/预测/评分。旧B1已执行同样568320次冻结CSIL shadow前向，未改变学生loss/状态/RNG；本轮两个新行的模型前向预算与B1相同。B0只有一个L前向且没有外部teacher，额外As训练成本单列。

每个真实L128步骤先按原生方式计算 ordinary CE，再生成同128物理样本的完整As；学生与冻结CSIL都接收同As。令b_i=1[teacher argmax==source L真TX]。CAPABILITY使用原b；SHUFFLED在真TX×sourceRX每组内部，用独立CPU generator(seed20261019621+step)置换b，保持每组及全批判对数，不改变训练全局随机流。两组均使用教师，SHUFFLED控制其类别/源RX级权重与计算，CAPABILITY−SHUFFLED检验逐观测关联是否有价值。

权重 w_i=0.5+0.5×b_i/mean(b)；全0时w_i=1。每个出现样本w_i≥0.5，全128平均w=1。总 L=0.5×ordinary CE+0.5×mean128(w_i×As CE)，CE temperature1、smoothing0.01，分母始终128。权重/教师输出停止梯度；不要求学生先答对，不做接受样本专属分母的loss。b全1或全0时准确退回普通As目标。记录全步teacher argmax、原bit/置换bit、TX/RX、逐包CE/权重、种子、实际总loss与真实更新供完整复算。

教师固定为已核实同数据契约的历史CSIL last200/defaulthead；现有 teacher.load 在模型加载前检查source角色、初始化和完整来源、actual config、类别与checkpoint payload，并要求原source资格。学生从零，不加载教师权重，教师eval/requires_gradFalse，放在学生非注册属性中，不进入学生state_dict/部署/EMA；结束时检查教师逐项冻结。教师只处理合法L As，U继续原ownEMA，不读取U TX或目标真值。历史教师200模型轮及其准备成本单列，不能把整个方法说成总共20轮；本轮不新训教师。

固定源推进规则：FINAL20的原fullV27000 clean与既有V1800两draw读出；CAPABILITY相对SHUFFLED和B1的mid/low urban两draw平均各≥+0.5个百分点，且相对B0的fullVclean≥−0.2个百分点。判定先于新正式分数读取，不根据target改变参数。任一失败关闭固定capability-label-CE allocation，不扫混合份额、教师、seed或阈值救援；健康行仍按预算完成并自动评价。PASS只获得自身E20＋最多30轮的源资格，追加前固定optimizer/EMA/U/采样器/信道衔接及匹配控制，不自动追加。

每row FINAL20冻结后自动源读出与168000物理样本×7固定视图单包预测；全部7预测固定后独立exec scorer才连接truth。报告acc、macroF1、最差RX、CAPABILITY相对B0/B1/SHUFFLED差、修正/新增错误、各suburban距自身clean的差。n=1模型seed，SD不适用、draw不算模型seed；暴露测试基准不称新的独立盲测。不得以权重转移、源资格、梯度或单seed小差替代识别与可重复性。

实际入口 `CVS-RFFI/python -u experiments/cvs_capability_ce20/runner.py`，普通N607既有FP32/TF32off环境、CPU2/worker；CWD `PROJECT/releases/cvs_F_capabilityCE_E20_20261019_r01`；不可覆盖输出 `PROJECT/runs/20261019-phase1-F-capabilityCE-e20-s2026092701-m2-r01`、日志 `<run>/logs/<row>.log`，唯一owner `codex/root/F-capability-CE20`。每GPU含其他owner最多2进程、至少12000MB空闲，CUDA init在共享capacity锁中；不停止、重启或热改健康旧任务。仅实际协议/来源/数值/预算/接口/输出问题技术停止并保留现场，不自动重试。

预期每行L训练前向1136640包、As568320包、外部教师568320包、原U学生396900包/ownEMA4440更新；源面板12600包、正式预测1176002包另计，实际耗时和NN在完成后报告。每行原生checkpoint无query smoke是首步，PASS后直接训练，无额外token/批准。精确row/config/seed与artifact路径由experiment.json维护。

已通过：公式与logit梯度、全部正CE和平均权重1、教师脱图、TX×RX数量保持、全0/全1回退、类别重标记与全局RNG不变、5协议负测；两实际F同初态、原native L hook、置换mask实际变化30/原mask变化0、Sinc/签名/时/频/PA/默认头均非零梯度、joint160；这些本地测试0真实sourceIQ/0真实教师CP/0target/0优化器更新。一次独立P0/P1审查PASS，无缺陷。代码/config身份在实际commit后写入登记；无额外设计SHA、签名或审批门槛。

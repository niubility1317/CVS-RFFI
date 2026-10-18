# P72：完整 practical 下 raw—As 的既有融合 Dropout 相关性

唯一假设：在固定完整 As、原六次 L 前向和原目标下，只让同一物理包的 raw 与 As 在已有 base160 上接受相同随机坐标支持，是否比独立掩码更有利于源 mid/low urban 联合更新。它只改变随机梯度相关性，不增加物理观测、身份信息或损失，也不保证降低梯度方差。不是全特征/分支对齐。

P71 已 ANALYZED，但 source 两条件 FAIL，固定风险权重关闭，不重开、不叠加。另一对话“审计并设计 CVS 联合增强”正在实现 U practical 三行，本侧不重复。PROP_GROUP 共享传播已有完整训练且源FAIL，不重做。旧dropoutCoupling的 ordinary↔As ANTITHETIC作用13个encoder随机层，此target也在旧prefix内；本轮关键变化为只选既有fuse.2、raw↔As IDENTICAL正相关，而不是声称该层从未受作用。As dropout旁路/E50改变边际，亦不能代替本对照；不重开旧旁路。

experiment.json是唯一逐row矩阵。两行I1/S1同modelseed2026092701，均严格加载原P43合法 scratchAs20学生/自身EMA，独立重置AdamW、lr2e-4/wd1e-4/cos10/CENTER_UNITS，新增E10/2220更新，总E30/6660。原As30、NoAs30冻结预测和评分复用；NoAs来自自身合法NoAs20父，不能称四行同父。禁止加载P70/P71已测试final做初始化。

固定F＋跨RX同TX SupCon、SIG_TIME_X3、joint160、默认cosine30头与正常联合更新；900722参数、206state。L6300/U56700(TX隐藏)/V27000、全部五源RX1/3/4/6/8与day1/2/3、split392005，未留出训练RX或Ldev。native U batch_neighbor/teacher/strong-agreement/熵目标保持，跟另一对话的共同取消学生门控不同。

准确目标模块 id_backbone.encoder.core.id_backbone.fuse.2，nn.Dropout(.45)、[B,160]。它只作用于TF融合后的base160，之后才添加PA/reference/SIG_TIME残差，不是最终完整joint160随机层。ordinary0/1原样；As0↔raw0、As1↔raw1以当前L128的smask物理位置配对，两RD pair仍独立。所有原Dropout先完整执行以保留global RNG；用pre-dropout输入×私有scaled mask覆盖目标行，不能再次对已dropout结果相乘。两行都抽相同As和独立raw私有mask；S1仅将raw对应mask替换为As mask。私有mask seed20261019072按step/pair/role命名，不含TX；不加参数或持久mask state，U/EMA/eval与其他随机层不改。

P70 nominal池142080条是在任何该池训练更新前按原P43尾段L物理序列/命名流产生。本轮两行只读按原生成顺序64*(step−1):64*step加载，每条观察只在一个更新中选取一次，既有两次RDrop网络前向仍保留。每一步先核当前raw[smask]的五项物理key、真实标签、compact domain、原step/slot，再消费对应As包。不是P70随机重排或质量分层，不做质量加权/筛选/补取/重画；不允许不同物理包错误配对。跨epoch/view的观测刷新已包含在原完整池中；训练/源评价/正式流分离，硬件session按RX/day/eq而非TX。所有合法L As继续参加CE，不按预测门控。

CE总系数1.25，As CE=.25、As sameIQ KL=.25；.5 RDordinary128+.5 RDmixed128+.25 RDraw128+.1 ORD128 SupCon原样，L每步768样本前向。不存在额外迁移约束、部署分支或诊断头。保持per-view Dropout边际，不称新风险目标；只有相关结构变化，不能把mask相合或训练CE下降当作身份收益。

源推进只两项：S1全V27000 clean−NoAs30≥−.1绝对百分点；两固定draw pooled mid/low S1−I1≥+.5个百分点。任一FAIL关闭固定配方，不扫p/site/maskseed，不扩E50或扩seed。源分层/parent正确集合/margin÷30/梯度/有效覆盖为解释，不新增门槛。无论source PASS/FAIL，两行FINAL30冻结后逐row自动clean＋六practical全预测，七预测固定后独立CPU scorer连接truth，结果不反向改参数、选模、晋级或重跑。

实际输入与物理路径沿原pinned20261015_r02的experiments/adv3b02_xuc/code/leo_practical，25MHz、float32 B×2×256、RMS center=false后从raw生成完整practical；residual/post_sync/EQfalse(G=1)/reference_snr_db30/reflect，质量适配/phase tracking/IQ补偿/AGC保留。训练fc2.462GHz、正式固定fc2.45GHz沿历史约定。固定正式received IQ与六信道配置/SNR/困难样本不变；不是总SNR的新增噪声比仅分层解释。

新增成本预计两E10共4440更新；每行native3162962+source12600+formal1176002，共8703128，加一次共享真实父student/ownEMA无query smoke4=8703132新远端样本前向。本地/分析实际CVS NN=0；mask运算不是网络前向。共用As20父成本4797722 NN/1810.249秒单列，原参照不重训/重测。池历史生成142080次/287.656秒，新行0新增模拟，加载与训练墙钟实记。所有seed逐row报告，不挑最好；仅单modelseed，不能证明可重复双提升。预算耗尽不等于收敛。

N607既有普通非特权CVS-RFFI环境，CPU2/FP32。CWD=PROJECT/releases/cvs_FcrossRX_rawAsMask_E30_20261019_r01；执行python -u experiments/cvs_raw_as_mask72/runner.py。唯一owner codex/root/raw-as-mask72，每GPU最多2总进程含其他owner且free≥12000MB；输出PROJECT/runs/20261019-phase1-FcrossRX-rawAsMask-e30-s2026092701-m2-r01不可覆盖。第一步真实checkpoint smoke PASS立即继续，不增审批。

技术停止限当前行的输入/父源/物理配对/输出冲突/错误mask范围或call数/非数值/预算/法律prediction错误；保留partial，不自动重试或重模拟，已健康同伴继续，不碰其他owner进程。发布按一Git代码/config提交、一次独立P0/P1、一归档一次SHA比较/compile、一次启动PID/CWD/argv/GPU/log增长；不加hash/receipt/报告审批。

预期完整产物含resolved config/真实模块与源契约/初始化、完整step与epoch序列、mask_steps与ordered_steps、frozen CP、固定源panel/两规则决定、all7prediction/独立scores、accuracy/macroF1/最差RX/RXdayTX/净纠错/normalized margin/parentAs20 clean正确集合保持、实际成本和四组PNG/PDF。代码起点7c9271cb1f1375b4064830aa136131820d552c2b；实际代码/config提交待本地实现检查与唯一审查通过后记录。

状态：PREREGISTERED，代码与分析实现完成；20协议负例、8mask CPU toy、2原序池/实际六前向hook toy、8分析CPU toy、21py语法PASS，唯一独立P0/P1审查PASS。0真实CVSNN/CP/IQ/新模拟；本地未测CUDA，真实父CP smoke4与实际GPU训练仍待launcher执行。尚无P72性能结果。

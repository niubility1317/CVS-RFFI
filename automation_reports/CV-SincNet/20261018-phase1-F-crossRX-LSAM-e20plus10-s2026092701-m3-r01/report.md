# F＋跨 RX 同 TX：有标签 SAM 与同 IQ 重放 E30

状态：RUNNING／VERIFIED，实际执行提交 a71b0094dba1264b7bacfc742dc03ab3f3a59aa2；三行已开始真实训练，尚无终轮或性能结果。唯一问题是在保持完整 practical、现有监督份额和原 U 规则的条件下，L 目标的有界参数邻域更新是否相对完全匹配的中性重放改善 practical 识别并保护 clean。本项是 A_s 训练组织，不是 IQ A_g；不构成四因子或联合成功。

|行|原始 E20 父|新10轮的L更新|每步L前向|额外L VJP|
|---|---|---|---:|---:|
|NOAS_REPLAY|原REPEAT20 student＋自身EMA|原ordinary目标，中性重放不改梯度|8|2|
|AS_REPLAY|原CONCAT20 student＋自身EMA|ordinary.5＋完整As.5，中性重放不改梯度|8|2|
|AS_LSAM|与AS_REPLAY完全同一CONCAT20 student＋自身EMA|上述同一L目标，固定rho.05 SAM替换L梯度|8|2|

三行使用完整L6300／U56700隐藏TX／V27000、split392005、RX[1,3,4,6,8]／day[1,2,3]／6TX、EQ1／25MHz／256点／逐包原RMS。没有训练RX留出或L_train/L_dev分折。F三路SIG_TIME／CENTER25K＋CENTER_UNITS／真实joint160／既有cos6×160 scale30／NoExplicitCFO与既有补偿保持，骨干和原头共同更新；无新分支、头或拟合诊断器。原U伪标签／教师／门控及额外前向保持，不使用U真值，不让urban反向生成clean监督。

父来源为PROJECT/runs/20261017-phase1-F-crossRX-fullsource-concat-e20-s2026092701-m3-r02下原REPEAT/CONCAT两行的source/final_ssdg.pth；本次只读d.parent_proof确认两者原从零、固定final20、4440更新、原5RX/3day/全部roleIDs一致、无目标训练或选模、默认头单独拟合0。两种treatment父不是同一权重，但源契约和初始化规则一致；As两行严格同父。来源依据与当前只读结果见parent_inputs.json。不存在已失败候选继承或因target成绩挑选父模型。

每行原20＋新10＝E30，追加2220次更新，累计6660；重置原AdamW及cos10，严格恢复自己的原EMA。共新增30 model-epochs/6660实际更新，两份原20父共享一次为40 model-epochs/8880更新，唯一累计70 model-epochs/15540更新；不把每行30轮和实际NN成本混淆。L-loader20261018131／U-loader20261018132及原6批buffer／37池每epoch保持。

设原L目标L=.5RD(ordinary)+.5RD(secondary)+.1SupCon，其中NoAs的secondary是同一ordinary，As的secondary是原完整practical(raw)。捕获gL，原生总反向得到gNative。在原clip前计算epsilon=.05*gL/||gL||，固定F存储坐标含CENTER25K q，不声称物理Hz或重参数不变；零梯度则epsilon0。四次重放使用四份实际encoder输入及各自forward入口的Python/NumPy/Torch/CUDA RNG，两个ordinary和两个As分别复用原已生成IQ，不重抽增强、传播或噪声。SAM在theta+epsilon计算gLnew，然后精确copy回原权重并恢复全局RNG；总梯度改为gNative−gL+gLnew。中性REPLAY也完成相同重放及求导，但完全保留gNative。原clip5／AdamW／EMA每步各一次，U及其余原目标在原theta的梯度不变，没有额外标量损失系数。该更新不是对真实接收信号做扰动，也没有有限AdamW步clean必降保证。

方法动机及固定rho.05取自[原SAM论文](https://arxiv.org/abs/2010.01412)与[作者实现示例](https://github.com/google-research/sam)，不从CVS target选择半径。原论文的通用泛化结果不能替代本项目星地识别证明。本次是限定L的SAM，区别于对全部U伪标签损失做扰动；也区别于已失败的梯度投影、dropout配对、MC2与噪声正负配对。关联对话当前研究跨视图SupCon覆盖，本行不改变SupCon视图、层、系数或U规则。

源资格固定用原V中按opaqueID排序、每TX/RX/day20包的V1800（90单元），clean一次、六practical各2draw，完整原residual/post_sync/noEQ/refSNR30/reflect保持，源随机流20261018303，原receiver session键按RX/day/eq独立于TX。每行23400默认头决策、21600完整物理处理、1个源缓存和1个最终CP，0更新／teacher／U／target。每行资格产物先于该行正式预测；矩阵资格函数只读三行源产物，不消费target，无论source通过与否，三行都按固定对比矩阵自动正式预测评分。主Agent在完整源资格冻结前不读取本run目标指标。

唯一候选AS_LSAM相对AS_REPLAY的预登记来源条件：clean和midurban不降、lowurban至少+.5个百分点、每suburban不低于−.25个百分点。任一失败即关闭这组固定L-SAM配方，不改半径／坐标／子集／父／seed／optimizer／预算挽救。通过仅是源资格，不是Ag有效、跨RX泛化或联合成功，也不据目标自动扩种子。正式报告列三行clean＋六practical的accuracy／macroF1／worstRX／TX/day、配对修复与新增错误、clean∩scene联合正确与三个suburban绝对值及clean差距；joint仍要求严格B00 clean净正、low>60及suburban探索差≤2pp且不靠降低clean。当前无新增A_g，因此没有四因子交互可报告。

执行：N607普通账户／既有固定F依赖／FP32／CPU2；CWD PROJECT/releases/cvs_F_LSAM_E30_20261018_r01；命令CVS-RFFI/bin/python -u experiments/cvs_lsam30/runner.py；唯一owner codex/root/F-LSAM-E30；不可覆盖输出PROJECT/runs/20261018-phase1-F-crossRX-LSAM-e20plus10-s2026092701-m3-r01，逐row/source、source_qualification、prediction、scores，日志root/logs。由既有容量锁选择GPU，每GPU至多3个总实验进程、空闲12GB以上，计入其他owner，不停止或热改健康任务。Git commit为实际代码身份，发布写入release_commit.txt，单归档一次传输对比、一次远端compile；launcher先一次两父真实CP无query smoke，PASS立即继续。

停止仅限原source/parent/hiddenlabel/非有限、重放范围/权重或RNG恢复错误、输出冲突及无法产生合法prediction等系统技术失败；保留失败与所有健康产物，不自动重启或改配方。低分不作技术停止。所有七视图prediction固定后独立CPU scorer连接truth，不回流参数、模型或重跑。REJECTED_EXTRA_GATE：通用skill的额外设计审批、spec/plan链超出当前用户明确授权的最小流程，不实施。

本地23负例、闭式方向／剩余梯度控制和完整F随机合成9216 forward通过；两个REPLAY相对原更新梯度最大差0，实际SAM范数0.05000000235，6次捕获/重放生命周期、权重／RNG／buffers恢复与teacher deepcopy归属检查通过，3次合成AdamW更新，12次额外F VJP。首次检查只因NumPy seed>2^32在任何F前向前失败，已修正测试seed模2^32；额外2个随机模型构建、0NN/CP/真实更新，记录保留。旧本地NumPy2/Torch桥接警告未被本检查路径使用，不修改环境或依赖。唯一一次独立P0/P1审查PASS，未发现问题；审查者仅跑23负例与配置一致性，不重复完整F或读取真实IQ/CP/远程/target。真实训练已开始，终轮及评分尚未完成。

预估每行新增训练L forward2273280（含重放1136640）；extraVJP4440／backward包导数2273280；原U/teacher保持。实际student-train/eval、teacher、资格、正式1176000决策、父成本与墙钟均从真实counter分别记录，不据预估宣称成本相同。

## 实际启动

三行真实PID/CWD/argv/CUDA及nvidia-smi进程映射已独立核实，分别GPU3/4/5；两份真实父CP smoke（4NN、0query）、单归档传输／compile及远端完整合成检查PASS。初次训练日志可完整读到NOAS_REPLAY 154步、AS_REPLAY 105步、AS_LSAM 104步，均无非有限跳步、U真值标记0；控制台按epoch输出，逐步gzip已有记录，不把安静控制台误判为停止。详见launch_verification.json。这里只是运行正确性证据，未证明源资格或正式识别收益。

开发完整F检查实际本地与远端各9216NN／3合成更新／12额外FVJP，总18432NN／6合成更新／24额外FVJP；真实父smoke4NN单列，后续真实训练／资格／正式预测／教师／父成本另记。没有热修改远端代码，没有更改旧健康运行。

三行均已完成E21回调／各222新增更新，并继续原进程推进；回调已验证每行基础L113664＋重放113664＝实际L227328NN、额外444次L VJP以及原单次clip／AdamW计数，零未消费图。该旧进度文件的L_packets只计基础4次前向，不能将其当实际8次总成本。逐步日志已进一步读到386／271／253步，三行原PID身份一致、gzip日志增长、无技术失败；没有完成E30或读取本run目标成绩。

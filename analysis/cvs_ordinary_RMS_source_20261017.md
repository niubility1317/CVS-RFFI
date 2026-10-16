# 同一普通增强 IQ 的尺度恢复：源反证

2026-10-17。当前固定候选：对已生成的原ordinary IQ恢复对应原始包RMS。已实现登记，19输入/配置负例＋8科学反例和96合成RMS操作PASS；原两P1修复及定向复审PASS，尚未启动真实源检验。基线仍F＋跨RX同TX、3SIG_TIME/统一Sinc/joint160/defaultcos/NoExplicitCFO，未来单模型训练预算E20。旧edge/cycle/IQmix/MC2/additive等固定失败继续关闭。

## 依据和区别

实际DataAugmentation普通路由与[既有审计](cvs_joint_ordinary_IQ_audit_20261016.md)确认：数据入口逐包RMS，ordinary出口仅nan_to_num，不恢复包功率。普通multipath先归一化独立tap，再将相同整数延时tap相加，不固定实际输入或组合核功率；AWGN与可选PA随后作用。这里不把合法随机多径增益称为程序错误，也不能据源码认定它造成识别失败。

[日程更正](cvs_ordinary_PA_schedule_correction_20261016.md)确认native使用min(epoch,13)，E13至20保持最强有效日程。新源检查固定effectiveE13、四个命名抽样，区别于历史PA探针E14至20较强外推，不冒称训练重放。旧constantgain的合成真实CP审计说明网络对包增益并非严格不变，恢复原输入RMS后原函数可复原；它没有证明现在恢复ordinary输出有效。

旧随机constantgain(raw) E20失败保持。本问题是去除现有ordinary生成的整体尺度差异，不是在raw上再次抽随机增益，不改旧范围或seed救跑。它也不删除PA/shift/multipath/其他既有增强；归一化不能撤销这些已产生的形状损伤。

## 数据前固定的算子与最小检验

x为原生RMS的raw，o=A_old(x)只生成一次，n=o sqrt(sum(x²)/sum(o²))。功率比用FP64求和，再输出原FP32；每包单一正实数，不读TX/RX/day或置信度、不夹增益、不重抽增强、不改相位/时序/相对波形、没有再次传播或加噪。零能量或非有限输入保存技术失败，禁止静默删包。仅恢复已生成ordinary的功率，不修改raw、As、U或正式测试IQ，不称为提高SNR或更换RX。

两份已核实source-only scratch final20 B00默认头（seed01/02）独立冻结评估，源角色与其原契约匹配，仅fitL5040、RX3/4/6/8×day1/2/3×6TX、72格各70包。原缓存L6300/未用V/历史特征会反序列化，RX1原L1260先排除；U仅角色ID元数据，无IQ或标签。两个模型原合法U规则不改变，本轮不训练，不用V/U/heldRX1/target，不把该折冒充完整L6300或L_train5580。

每模型先raw5040；四个固定命名抽样各生成原ordinary5040，与同一IQ恢复RMS后的5040配对，共45360源NN。两模型90720源NN＋首模型一次真实CP合成2包no-query smoke=90722。原ordinary生成40320个包，归一化40320个view，两个模型分别重建相同20160观察；另每模型64个合成ordinary生成用于原生输出/随机状态的canonical检查、0NN。0新epoch/更新/教师/另训头。历史两参照40modelE/8880更新/9373356NN单列。

四个AUG种子为202610171701至202610171704，单独shuffle generator的种子为202610171710至13；每抽样全L只出现一次，不挑样本/抽样。两模型使用相同source order和抽样。精确种子与矩阵只在experiment.json/design.config维护。最强日程是代码确定的实际有效日程，不从source/target分数选择。

两模型均须：RMS相对原ordinary accuracy≥+.2pp、macroF1不降，至少3/4抽样accuracy正增益且每抽样不低于−.2pp；相对raw accuracy/F1≥−.5pp、raw正确保留≥99%；每RX/TX/day相对raw≥−1pp且相对ordinary≥−.5pp；至少5%观察的校正因子偏离1超过1%；包功率误差≤1e−5、正实数形状保持误差≤1e−6。保留全部原ordinary错误和所有分层，统计修复/新错。非空、尺度和RNG检查不能代替身份条件。

任一模型失败，关闭这个完整ordinary后置RMS/E13/四抽样问题，不换归一化位置、部分系数、clip增益、禁用某个旧增强、抽样、RX或seed救跑。通过也只提供源尺度与身份依据，不自动训练或访问target。

下一匹配E20若获得源依据，须另登记三组：共同raw.75＋ordinary.25＋原SupCon保持，extra.25分别raw／同一ordinary／同一ordinary恢复RMS，均8L前向/1.25总权重、原U规则、默认头联合训练。恢复RMS不能隐式多抽一个ordinary，RDrop必须同IQ。须同时超过两个控制clean≥+1pp、F1不降、每practical/seenV≥−.5pp；单源种子通过再独立确认。此训练尚未实现、登记或启动。

完整联合目标ACTIVE：新Ag、As稳定、完整契约四格/交互项、跨seed与正式clean＋六practical、lowurban>60%及suburban距自身clean≤2pp探索目标仍未完成。源fit修复不是跨RX clean收益，普通增强尺度修正尚不能称Ag有效。


普通增强输出RMS r01 20261017-diagnostic-F-crossRX-ordinaryRMS-fitL-manysig-m2-r01 FAILED_TECHNICAL_VERIFIED：SatViewStage日志序列化失败，2合成NN/1CP加载/64合成ordinary，0源NN/训练/target；seed02未启动，全部own进程自然退出。源假设尚未检验，不把技术失败作科学FAIL，不覆盖或自动重试旧根。


r02 20261017-diagnostic-F-crossRX-ordinaryRMS-fitL-manysig-m2-r02 已登记未启动；仅修复原生dataclass日志表示，原科学矩阵和条件保持，r01失败成本单列。
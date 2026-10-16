# F＋跨 RX 同 TX：旧循环移位的同抽样身份反事实

状态：PLANNED，未启动。唯一假设：继承的普通 IQ 循环移位对默认头带来可避免的源身份错误。这是旧增强审计和可能的普通配方修正，不是已成立的新 A_g。固定微频移 E20 已关闭：clean 无净提升，六个 practical 均下降；不换幅度、seed、RX 或阈值救跑。

## 固定最小源矩阵

唯一基线 F＋跨 RX 同 TX 对比学习，三路 SIG_TIME、统一 Sinc、真实 joint160、默认 cosine 头及 NoExplicitCFO/必要逐包参考补偿保持。本次只冻结自有 B00/seed2026092701/final20 做默认头诊断，不拟合任何分类器，不初始化新训练。checkpoint 来自 `20261016-phase1-F-crossRX-satmix25-sourceRX1-e20-s2026092701-m2-r01/B00-RX1-s2026092701/source/final_ssdg.pth`，对应 scratch E20/4440 更新，无父权重、目标接触或目标选模；严格复用已核实的原始数据角色和 checkpoint_model 加载路径。父训练 4686678 NN 包前向另列。

实际四 RX 训练契约 L5040/U45360/V21600；冻结诊断仅用 L5040 的 RX3/4/6/8×day1/2/3×全部6TX，每单元70包。复用 source_contract 与 SOURCE_GEOMETRY_AUDIT_COMPLETE/EXACT_MATCH 缓存；在统计、变换和推理前排除 heldRX1，fit ID 必须严格等于模型的 L_s。缓存会反序列化其他源角色/旧特征，随后丢弃，未作统计、推理或更新。U 真值不读取，U/V/heldRX1/target 不使用。不得把此契约称完整 L6300 或 L5580/Ldev720。

普通实际顺序：DC/taper→循环移位→gain/phase/CFO/Wiener/multipath/AWGN→DAC→PA。DAC p=0，class_signature=false，旧 PA 保留。循环移位是 modulo256 gather，批级概率、逐包整数，实际 configure E1..13 的 max_shift 为3至8；接缝不落在参考窗80:160内，但窗内内容被平移，是否损伤身份待测。训练 E14..20 继续使用 capped E13，而非外推日程，本次只逐个覆盖实际 E1..13 调用参数。

raw 只推理一次。每个日程 Python/Torch seed=20261016511+epoch，固定缓存顺序及批128/末48；这是新诊断抽样，不伪称重建训练轨迹。每批从同一个随机起点运行完整原生增强两次，full 返回原始 shift 输出，bypass 在调用原始 shift 并保留整数抽样后返回 shift 前输入。后续链完整执行；所有外生抽样和 PA strengths 相同，AWGN 功率根据各自信号自然重算。两条末 RNG 须一致，并恢复 canonical full 末状态。记录 shift 整数、nonzero 与最终 IQ changed，不用 changed 集合替换预登记 nonzero 集合。无额外 RMS。每轮首128包另跑未包裹的 canonical 原生增强，须输出和 RNG 末状态逐位相同；增加1664增强生成，0额外NN。

raw5040＋13×5040×2=136080源默认头预测，加2合成真实CP无query smoke=136082实际NN；paired普通生成131040，加canonical1664，总132704增强包生成。每次shift调用额外观测重放整数draw并恢复状态，单独累计scalar draw数。0训练、头拟合、teacher、As和部署模块。局部验证1440合成增强包/14负例PASS，0真实IQ/CP/NN；前次错误测试条件消耗1248合成包，亦单列保留，不计作训练或真实样本。小批自然抽样未触发shift，因此另设强制触发的纯合成对照；正式实验不强制或重抽。

## 预登记判断与边界

全部13日程汇总的实际非零shift子集，bypass 相对 full 的默认头净修复至少2个百分点，并且至少3个 fitRX 净修复为正，4个RX均有非零shift覆盖。覆盖为零即FAIL。记录全包及子集accuracy、Macro-F1、CE、CM、repair/new break，逐轮和RX/day/TX全部报告。不能把同一物理包的13次观测视为独立样本显著性。

不满足即关闭固定循环移位修正假设，无draw/seed/RX/阈值救跑。PASS只支持另行登记的同预算 scratch E20 旧普通配方修正，需要 clean 至少+1pp、Macro-F1不降、每个practical及源V不低于−0.5pp和训练健康全部成立；这里不自动训练，不把冻结fit修复当跨RX泛化。PA先前冻结fit修复没有转成 E20 clean 收益，因此这次也保留同样的证据边界。联合 lowurban>60%、suburban接近自身clean和 B00/B10/B01/B11 仍未完成。

## 发布与产物

唯一owner `codex/root/F-CROSS_RX-ordinaryshift-20261016`。普通N607用户，CPU2线程、CUDA_VISIBLE_DEVICES为空，不占GPU槽，不停止或改动健康任务。精确CWD为 `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_F_crossRX_ordinaryshift_fitL_20261016_r01`，命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u experiments/cvs_shift_counterfactual16/runner.py`；run root为PROJECT/runs/20261016-diagnostic-F-crossRX-ordinaryshift-fitL-s2026092701-m1-r01。完整数据路径、六类seed、矩阵及输出见experiment.json。实际代码身份以一次Git commit/release_commit/resolved为准，提交后发布时回填。

一次独立P0/P1审查PASS，无P0/P1。一次N607 preflight、单归档本远端SHA比较和remote compile；launcher第一步真实匹配CP合成无query smoke，PASS立即继续；随后独立PID/CWD/argv/CPU环境及log增长读回。协议/CP/非有限数、native draw/canonical输出/RNG/冻结状态/成本不匹配、输出冲突为技术失败，保留产物，不自动重试；科学负结果完成全部13轮，不提前停或热改。

保存fitIDs、resolved、smoke、raw及13轮logits/predictions、实际draw/changed/MSE记录、epoch汇总、result/completion或failure及完整stdout。预测完成后独立 NumPy 重算136080默认头预测及logits/CE/F1/CM、936源单元及全部配对判断。IQ/CP/逐包预测/原始日志留在私有产物，仅小型汇总与代码配置登记入Git。

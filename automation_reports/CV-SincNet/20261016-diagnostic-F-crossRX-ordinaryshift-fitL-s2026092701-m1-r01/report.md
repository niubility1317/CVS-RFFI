# F＋跨 RX 同 TX：旧循环移位的同抽样身份反事实

状态：ANALYZED_SOURCE_ONLY／VERIFIED。全部固定日程已完成并独立复算。唯一假设：继承的普通 IQ 循环移位对默认头带来可避免的源身份错误。这是旧增强审计和可能的普通配方修正，不是已成立的新 A_g。固定微频移 E20 已关闭：clean 无净提升，六个 practical 均下降；不换幅度、seed、RX 或阈值救跑。

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


实际执行commit `b43ebc53c54b50735a06541f6c0c04d1f4815ba7`；archive/compile/realCP smoke、PID/CWD/argv/CPU环境与log增长VERIFIED。全部13日程继续，无目标访问、重启或热改。


## 实际完整结果

固定判定 **PASS**，全部136080默认头预测及logits/CE/F1/CM、936源单元及全部RX/day/TX配对已独立复算。判断条件：{"nonzero_coverage": true, "identity_recovery": true, "RX_coverage": true, "RX_positive": true}。

原raw源fit准确率99.841270%；非零shift3946观测，最终IQ变化3946，涉及2853个唯一物理包；修复150、新增错16、子集净恢复+3.395844pp。全65520观测净恢复+0.204518pp。重复物理观测相关，未做独立包显著性宣称。

| 日程 | shift概率/最大样点 | 非零/变动 | full准确率 | bypass准确率 | 非零子集恢复 |
|---|---:|---:|---:|---:|---:|
| 1 | 0.035000/3 | 223/223 | 98.690476% | 98.928571% | +5.381166pp |
| 2 | 0.035000/3 | 675/675 | 97.361111% | 97.599206% | +1.777778pp |
| 3 | 0.035000/3 | 103/103 | 98.095238% | 98.333333% | +11.650485pp |
| 4 | 0.037964/3 | 0/0 | 97.380952% | 97.380952% | 无覆盖 |
| 5 | 0.042050/4 | 226/226 | 96.309524% | 96.468254% | +3.539823pp |
| 6 | 0.046703/4 | 458/458 | 98.809524% | 98.948413% | +1.528384pp |
| 7 | 0.051768/5 | 123/123 | 97.777778% | 97.876984% | +4.065041pp |
| 8 | 0.057162/5 | 110/110 | 96.230159% | 96.329365% | +4.545455pp |
| 9 | 0.062834/6 | 468/468 | 97.480159% | 97.996032% | +5.555556pp |
| 10 | 0.068749/6 | 349/349 | 93.888889% | 94.265873% | +5.444126pp |
| 11 | 0.074880/7 | 123/123 | 97.658730% | 97.718254% | +2.439024pp |
| 12 | 0.081206/7 | 358/358 | 93.234127% | 93.313492% | +1.117318pp |
| 13 | 0.087710/8 | 730/730 | 90.019841% | 90.436508% | +2.876712pp |

| 分层 | 非零观测 | 修复 | 新错 | 恢复 |
|---|---:|---:|---:|---:|
| by_RX=3 | 1121 | 74 | 4 | +6.244425pp |
| by_RX=4 | 826 | 14 | 6 | +0.968523pp |
| by_RX=6 | 1066 | 32 | 3 | +2.720450pp |
| by_RX=8 | 933 | 30 | 3 | +2.893891pp |
| by_day=1 | 1152 | 60 | 4 | +4.861111pp |
| by_day=2 | 1360 | 49 | 5 | +3.235294pp |
| by_day=3 | 1434 | 41 | 7 | +2.370990pp |
| by_TX=0 | 405 | 18 | 0 | +4.444444pp |
| by_TX=1 | 752 | 26 | 1 | +3.324468pp |
| by_TX=2 | 790 | 22 | 8 | +1.772152pp |
| by_TX=3 | 898 | 77 | 5 | +8.017817pp |
| by_TX=4 | 727 | 2 | 2 | +0.000000pp |
| by_TX=5 | 374 | 5 | 0 | +1.336898pp |

全增强观测：full accuracy=96.379731%，bypass=96.584249%；Macro-F1=96.379427%→96.584294%；CE=0.109963143→0.103989892。

13次canonical未包裹全链输出及RNG控制PASS；shift实际调用34次（每个配对含两个相同native调用），额外观测scalar draw数8704；shift直方图：{"-8": 46, "-7": 92, "-6": 143, "-5": 188, "-4": 236, "-3": 435, "-2": 400, "-1": 412, "0": 61574, "1": 424, "2": 436, "3": 429, "4": 260, "5": 174, "6": 152, "7": 69, "8": 50}。仅移位输出不同，输入依赖噪声功率仍自然重算。

实际执行commit `b43ebc53c54b50735a06541f6c0c04d1f4815ba7`。archive/remote compile/CP smoke/PID-CWD-argv-CPU/log增长VERIFIED，进程自然退出、completion齐全、无failure。新增136082NN、132704增强包生成（paired131040＋canonical1664）、0训练/头拟合/teacher/As，参数buffer未变；墙钟81.479482秒、完整stdout16行。父E20成本和局部1440＋1248合成增强成本单列。未使用heldRX1/U/V/target。

已证实：固定冻结默认头对这些fit源包的旧循环移位存在可避免错误，达到预登记恢复和RX条件。尚未证实：训练时旁路移位会改善未见RX clean，或带来practical/联合收益。下一唯一假设是只旁路旧ordinary shift，在相同原生抽样、原raw份额、旧PA/U、六L前向及F＋CROSS_RX下改善heldRX clean；最小为RX1固定源折、seed01、从零E20的B00/B10匹配两组，逐行固定final20后自动源clean＋六practical。clean不足+1pp、Macro-F1下降、任一practical或V低于−0.5pp、训练不健康即放弃，不改seed/RX/draw/阈值救跑。此后续训练尚未实现/登记/启动。

完整逐包预测/logits/抽样与stdout保留于私有 `local_artifacts/F_ordinary_shift16_20261016/final`，两张完整日程及RX/day/TX图同目录；小型完整源汇总见[source_analysis.json](source_analysis.json)。

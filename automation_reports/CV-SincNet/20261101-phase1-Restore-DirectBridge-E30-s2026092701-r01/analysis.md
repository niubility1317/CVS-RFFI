# DirectBridge：原 Main 七景性能结果

**六景达标 0/6，合并 clean 保持。** 原地面完整 Tri-SIG-TIME SSL/default head/Main split，raw clean 86.538690%（145385/168000），各景至少 83.538690%（140345条正确）。

cross_rx clean 净变化 +163 条，cross_both 净变化 -54 条。两个总体分别报告。

六景中 0 景高于 PowerAnchor，0 景高于 DURA；低档城区距3pp线仍差 32478 条正确预测。

本版尚未达到主性能门槛，目标保持 ACTIVE。默认DURA不变，源端均值和计算成本不替代真实身份性能。

![主性能](E:/type10-7/automation_reports/CV-SincNet/20261101-phase1-Restore-DirectBridge-E30-s2026092701-r01/primary_performance.png)

| 场景 | DURA | PowerAnchor | DirectBridge | 距raw clean | 相对对照 | 相对DURA |
|---|---:|---:|---:|---:|---:|---:|
| clean | 86.615476% | 86.644643% | 86.603571% | -0.064881 pp | -0.041071 pp | -0.011905 pp |
| 高档郊区 | 84.343452% | 78.990476% | 77.663690% | 8.875000 pp | -1.326786 pp | -6.679762 pp |
| 中档郊区 | 83.016071% | 77.714881% | 76.391071% | 10.147619 pp | -1.323810 pp | -6.625000 pp |
| 低档郊区 | 81.322024% | 76.166667% | 74.903571% | 11.635119 pp | -1.263095 pp | -6.418452 pp |
| 高档城区 | 83.049405% | 77.799405% | 76.459524% | 10.079167 pp | -1.339881 pp | -6.589881 pp |
| 中档城区 | 75.705952% | 71.533929% | 70.286905% | 16.251786 pp | -1.247024 pp | -5.419048 pp |
| 低档城区 | 68.801190% | 65.433929% | 64.206548% | 22.332143 pp | -1.227381 pp | -4.594643 pp |

## 两个总体与 clean

| 总体 | 达标景数 | raw clean正确数 | 新G clean正确数 | clean净变化 |
|---|---:|---:|---:|---:|
| cross_rx | 0/6 | 111019/126000 | 111182/126000 | +163 |
| cross_both | 0/6 | 34366/42000 | 34312/42000 | -54 |
| combined | 0/6 | 145385/168000 | 145494/168000 | +109 |

all_score_groups.csv包含全部TX/RX/day/RXday、准确率与macro-F1（比例0至1）；combined_group_metrics.csv按互斥总体混淆计数合并，未平均百分比。paired_changes.csv列DURA/对照净纠错，paired_clean_changes.csv列相对同记录raw clean的纠错/损伤/判决保持。完整数据均来自冻结后独立评分，不用于改选checkpoint或超参数。

## 同条样本修复与损伤

| 对照 | 场景 | 修复错误 | 损伤原正确 | 净变化 |
|---|---|---:|---:|---:|
| DURA_v1 | clean | 958 | 978 | -20 |
| DURA_v1 | 高档郊区 | 4720 | 15942 | -11222 |
| DURA_v1 | 中档郊区 | 5973 | 17103 | -11130 |
| DURA_v1 | 低档郊区 | 7388 | 18171 | -10783 |
| DURA_v1 | 高档城区 | 5815 | 16886 | -11071 |
| DURA_v1 | 中档城区 | 10925 | 20029 | -9104 |
| DURA_v1 | 低档城区 | 14997 | 22716 | -7719 |
| MSAT_PowerAnchor | clean | 1071 | 1140 | -69 |
| MSAT_PowerAnchor | 高档郊区 | 3259 | 5488 | -2229 |
| MSAT_PowerAnchor | 中档郊区 | 3732 | 5956 | -2224 |
| MSAT_PowerAnchor | 低档郊区 | 4317 | 6439 | -2122 |
| MSAT_PowerAnchor | 高档城区 | 3921 | 6172 | -2251 |
| MSAT_PowerAnchor | 中档城区 | 6605 | 8700 | -2095 |
| MSAT_PowerAnchor | 低档城区 | 9053 | 11115 | -2062 |

## 方法检验与边界

本版在PowerAnchor的损失基础上，同时改变训练状态、结构条件和四步执行，是系统比较。估计器接收中间state、原received和时间，并预测连续160维源clean特征作为FiLM条件；F0特征只用于训练loss。四次共享同一权重，部署不需要clean或额外观测，原F0/defaulthead冻结。

训练中途、该版Main访问前发现：若z=(1-t)x+t y且估计器同时读取z/y/t，t<1存在x=(z-t y)/(1-t)的解析捷径；理想化使用该捷径会使后续四步保持第一步预测。这是合成数据恒等式，不证明实际模型一定采用它或某一Main变化由此造成。已运行版本完整保留，另建StateBridge匹配对照；没有据该限制取消测试或热改本版。[解析说明](paired_state_shortcut.md)。

训练state=(1-t)detachedsourceclean+t detachedsourcereceived，t=1概率1/2、.75/.5/.25各1/6，无额外噪声。源clean进入中间训练输入的事实明确披露；部署中无clean参考。原R07 Ls6300/Us56700、20+10draw、order2026110102/modelseed2026092701、AdamW/.001-warm247-cos2e-5/WD1e-4/B256tail24/globalclip1、单阶段30/7410保持，time RNG为globalCUDA2026110103，技术恢复保存CPU/CUDA RNG。

损失为IQ NMSE+.05STFT+.1（输出F0余弦+源方差标准化MSE）+.1（latent余弦+同尺度MSE）+.1raw/同条clean RMS比平方误差。teacher连续特征/方差/RMS只作loss，不使用TX/logits/CE/六类原型或新身份头，V/OE/novel/目标不训练、早停或选模，OE仅冻结诊断。F-guided而非F-free；无旧G/coarse/DURA/神经IQ教师祖先。

每记录训练只调用1次estimate，部署从received开始4次，更新为z_next=z+(predictedclean-z)/remaining，remaining4/3/2/1，末步直接预测clean。每个clean预测单位复功率，中间z不另作单位投影。确定、逐记录执行，不消耗额外真实观测或跨query统计。形式借鉴[InDI式5](https://arxiv.org/html/2303.11435v1)，不是原论文完整复现，不继承图像性能结论。

final30固定后自动完成原Main168000×七景，源质量不取消测试；全部预测固定后独立CPU truth-last。raw/DURA精确复用，新G全量4步推理，默认DURA保持。原Main已曝光、单seed，不构成新独立盲测或未知设备保真证明。

![完整端点训练曲线](E:/type10-7/automation_reports/CV-SincNet/20261101-phase1-Restore-DirectBridge-E30-s2026092701-r01/full_training_curves.png)

![按训练时间分开](E:/type10-7/automation_reports/CV-SincNet/20261101-phase1-Restore-DirectBridge-E30-s2026092701-r01/source_time_curves.png)

## 完整训练与成本

| 项目 | 对照 | 新G |
|---|---:|---:|
| 唯一参数 | 4670178.000000 | 4677538.000000 |
| 训练秒数 | 1487.327767 | 1858.307182 |
| worker秒数 | 1493.229222 | 1864.851426 |
| 轮时中位数（秒） | 49.471202 | 61.761007 |
| 峰值显存（MiB） | 2381.208008 | 2435.731934 |

source_time_metrics.csv记录每轮四个位置的计数和全部分项。端点t=1与较容易的中间位置分开，不用整体更低的损失推定恢复能力更强。曲线中PowerAnchor使用原received全体；桥模型t=1为当轮随机约半记录，不能把不同输入/曝光均值当作严格冻结面板的配对效果。StateBridge对照另外核实两版各轮time直方图与末轮CUDA RNG。

| 最后一轮新G时间 | 数量 | IQ NMSE | 输出F标准化误差 | latent标准化误差 | raw/clean RMS |
|---|---:|---:|---:|---:|---:|
| 0.25 | 10440 | 0.00008644 | 0.00056938 | 0.04695210 | 0.99397114 |
| 0.50 | 10638 | 0.00024038 | 0.00137402 | 0.04825766 | 1.00081024 |
| 0.75 | 10752 | 0.00051229 | 0.00286482 | 0.04960016 | 0.99412054 |
| 1.00 | 31170 | 0.02824908 | 0.07592674 | 0.10494073 | 1.00160955 |

静态部署1043189760实数Conv/Linear/attention MAC/包=4×260797440，0FFT/CG；参数共享一次存储，没有将4个调用算成4个独立模型。排除F0/norm/activation/softmax/FiLM逐点/插值/投影/bridge更新/内存，不是实测时延。共享GPU负载变化，墙钟差不证明隔离硬件因果提速。性能主线，效率支线。

各臂原Ls曝光189000、Us曝光1701000，F0源clean目标63000、在线输出特征1890000；中间训练位置额外使用同条clean IQ，成本与权限明确。技术快照计入训练墙钟，不能将30轮称为系统历史全部成本。

独立复算84比较单元、2604组、1736raw配对和1302额外配对，最大指标误差1.25e-14，raw/DURA判决差0。完整60轮/stdout、源顺序/draw、从零E0、固定四步配置、每轮总loss与各t重组/权重、30技术快照和E30-final一致性通过；复算0新增神经前向/更新，覆盖检查先于truth，使用原共享锁/CUDA空可见。

## 识别增强与恢复比例

| 模型 | 场景 | 修复clean原错误 | 损伤clean原正确 | 净变化 | Fclean判决一致率 |
|---|---|---:|---:|---:|---:|
| MSAT_PowerAnchor | clean | 522 | 344 | +178 | 99.421429% |
| MSAT_PowerAnchor | 高档郊区 | 4885 | 17566 | -12681 | 86.064881% |
| MSAT_PowerAnchor | 中档郊区 | 5024 | 19848 | -14824 | 84.524405% |
| MSAT_PowerAnchor | 低档郊区 | 5098 | 22523 | -17425 | 82.675595% |
| MSAT_PowerAnchor | 高档城区 | 5048 | 19730 | -14682 | 84.527976% |
| MSAT_PowerAnchor | 中档城区 | 5567 | 30775 | -25208 | 76.955952% |
| MSAT_PowerAnchor | 低档城区 | 5872 | 41328 | -35456 | 69.697619% |
| MSAT_DirectBridge | clean | 985 | 876 | +109 | 98.731548% |
| MSAT_DirectBridge | 高档郊区 | 4608 | 19518 | -14910 | 85.087500% |
| MSAT_DirectBridge | 中档郊区 | 4785 | 21833 | -17048 | 83.508333% |
| MSAT_DirectBridge | 低档郊区 | 4938 | 24485 | -19547 | 81.671429% |
| MSAT_DirectBridge | 高档城区 | 4786 | 21719 | -16933 | 83.551190% |
| MSAT_DirectBridge | 中档城区 | 5557 | 32860 | -27303 | 75.780357% |
| MSAT_DirectBridge | 低档城区 | 6001 | 43519 | -37518 | 68.424405% |

低档城区准确率为64.206548%，相当于raw clean准确率的74.194037%；该准确率保留比例不是波形或发射机信息恢复率，不是理论极限。

修复clean原错误是固定系统任务收益，损伤数需同时报告。IQ NMSE、F(clean)判决保持和真实TX准确率回答不同问题；四步先验可能改变可用线索的表达，不能由“像clean”推定真实发射机特征被增强，更不能据闭集分数证明未知设备保真。没有条件熵/MMSE或可辨识性界时，不给任意网络“最多几成”的伪数值。[理论边界](../20261101-phase1-Restore-FeatureCond-E30-s2026092701-r01/theoretical_limits.md)。

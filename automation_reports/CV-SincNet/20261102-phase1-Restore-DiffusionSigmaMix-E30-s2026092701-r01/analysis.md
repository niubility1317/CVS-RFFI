# 部署噪声尺度均衡训练：原 Main 七景结果

**六景达标 0/6，合并 clean 保持，两个 clean 总体均保持。** 同一原地面完整 Tri-SIG-TIME SSL/defaulthead/Main split；raw clean 86.538690%（145385/168000），每景至少83.538690%（140345条正确）。

本版未达到主性能门槛，完整目标保持 ACTIVE。 默认 DURA 保持。

![主性能](E:/type10-7/automation_reports/CV-SincNet/20261102-phase1-Restore-DiffusionSigmaMix-E30-s2026092701-r01/primary_performance.png)

| 场景 | DURA | 原残差扩散 | SigmaMix | 距 raw clean | 相对 原残差扩散 | 相对 DURA |
|---|---:|---:|---:|---:|---:|---:|
| clean | 86.615476% | 85.913690% | 86.718452% | -0.179762 pp | +0.804762 pp | +0.102976 pp |
| 高档郊区 | 84.343452% | 73.621429% | 73.260119% | 13.278571 pp | -0.361310 pp | -11.083333 pp |
| 中档郊区 | 83.016071% | 71.008333% | 71.301190% | 15.237500 pp | +0.292857 pp | -11.714881 pp |
| 低档郊区 | 81.322024% | 68.463095% | 69.273214% | 17.265476 pp | +0.810119 pp | -12.048810 pp |
| 高档城区 | 83.049405% | 70.593452% | 71.669048% | 14.869643 pp | +1.075595 pp | -11.380357 pp |
| 中档城区 | 75.705952% | 58.003571% | 63.918452% | 22.620238 pp | +5.914881 pp | -11.787500 pp |
| 低档城区 | 68.801190% | 48.688095% | 56.502976% | 30.035714 pp | +7.814881 pp | -12.298214 pp |

## 两个总体分别验收

| 总体 | 达标景数 | raw clean 正确数 | 新 G clean 正确数 | clean 净变化 |
|---|---:|---:|---:|---:|
| cross_rx | 0/6 | 111019/126000 | 111312/126000 | +293 |
| cross_both | 0/6 | 34366/42000 | 34375/42000 | +9 |
| combined | 0/6 | 145385/168000 | 145687/168000 | +302 |

全部TX/RX/day/RX×day、macro-F1及混淆计数见all_score_groups.csv，按互斥总体计数合并见combined_group_metrics.csv；未平均不同总体百分比。没有按场景切换模型、按目标重新选样或选择性补跑。

## 逐条纠错与损伤

| 对照 | 场景 | 修复 | 损伤 | 净变化 |
|---|---|---:|---:|---:|
| DURA_v1 | clean | 766 | 593 | +173 |
| DURA_v1 | 高档郊区 | 2755 | 21375 | -18620 |
| DURA_v1 | 中档郊区 | 4067 | 23748 | -19681 |
| DURA_v1 | 低档郊区 | 5590 | 25832 | -20242 |
| DURA_v1 | 高档城区 | 4045 | 23164 | -19119 |
| DURA_v1 | 中档城区 | 9948 | 29751 | -19803 |
| DURA_v1 | 低档城区 | 14432 | 35093 | -20661 |
| TF_ResidualDiffusion | clean | 2150 | 798 | +1352 |
| TF_ResidualDiffusion | 高档郊区 | 3867 | 4474 | -607 |
| TF_ResidualDiffusion | 中档郊区 | 5313 | 4821 | +492 |
| TF_ResidualDiffusion | 低档郊区 | 6644 | 5283 | +1361 |
| TF_ResidualDiffusion | 高档城区 | 6620 | 4813 | +1807 |
| TF_ResidualDiffusion | 中档城区 | 16762 | 6825 | +9937 |
| TF_ResidualDiffusion | 低档城区 | 22391 | 9262 | +13129 |

## 相对同条 raw clean 的真实 TX 与判决保持

| 模型 | 场景 | 修复clean错误 | 损伤clean正确 | 净变化 | Fclean判决一致率 |
|---|---|---:|---:|---:|---:|
| TF_ResidualDiffusion | clean | 590 | 1640 | -1050 | 98.552976% |
| TF_ResidualDiffusion | 高档郊区 | 3230 | 24931 | -21701 | 81.949405% |
| TF_ResidualDiffusion | 中档郊区 | 3279 | 29370 | -26091 | 78.909524% |
| TF_ResidualDiffusion | 低档郊区 | 3254 | 33621 | -30367 | 75.957143% |
| TF_ResidualDiffusion | 高档城区 | 3272 | 30060 | -26788 | 78.445833% |
| TF_ResidualDiffusion | 中档城区 | 3386 | 51325 | -47939 | 63.811905% |
| TF_ResidualDiffusion | 低档城区 | 3426 | 67015 | -63589 | 53.112500% |
| TF_DiffusionSigmaMix | clean | 716 | 414 | +302 | 99.188690% |
| TF_DiffusionSigmaMix | 高档郊区 | 2803 | 25111 | -22308 | 81.402976% |
| TF_DiffusionSigmaMix | 中档郊区 | 2980 | 28579 | -25599 | 78.990476% |
| TF_DiffusionSigmaMix | 低档郊区 | 3129 | 32135 | -29006 | 76.400595% |
| TF_DiffusionSigmaMix | 高档城区 | 3024 | 28005 | -24981 | 79.314286% |
| TF_DiffusionSigmaMix | 中档城区 | 4179 | 42181 | -38002 | 69.153571% |
| TF_DiffusionSigmaMix | 低档城区 | 4812 | 55272 | -50460 | 60.133333% |

## 完整训练与实际成本

| 项目 | 原残差扩散 | SigmaMix |
|---|---:|---:|
| 唯一参数 | 1614608.000000 | 1614608.000000 |
| 训练秒数 | 593.290977 | 819.593756 |
| worker秒数 | 597.716530 | 824.286164 |
| 轮时中位数（秒） | 19.699457 | 28.019856 |
| 峰值显存（MiB） | 2509.326172 | 2509.328613 |

![完整训练曲线](E:/type10-7/automation_reports/CV-SincNet/20261102-phase1-Restore-DiffusionSigmaMix-E30-s2026092701-r01/full_training_curves.png)

上述新模型曲线是带clean residual独立加噪的单步去噪训练输出，不是received起点完整15调用sampler的冻结源评测；低噪声时训练输入含更多clean信息，两组sigma分布不同，不能将其平均IQ/F0训练误差直接视作同分布的质量比较。完整source_training_metrics.csv保留两种measurement，未据源分数选模或跳过Main。

新G从零单阶段30/7410/原Ls6300Us56700/R07/同sourceorder-draw/LR，F0连续160特征及源尺度仅loss指导、原F0/defaulthead冻结。没有TX/logits/CE/原型/新身份头、V/OE/novel/目标更新、早停或选模，OE仅冻结诊断。配对clean仅用于训练加噪残差及loss；部署只读同条received与state内固定latent，无clean/F0/真信道/SNR/额外观测/batch统计/采样选择。架构/实际E0/源顺序draw/LR/损失系数/完整15调用部署及E01-E30全局CPU-CUDA Gaussian RNG保持；仅sigma抽样改为半原lognormal、半均匀8部署点。不同sigma分布改变期望训练风险，没有做importance correction，不称同一积分目标的无偏重采样。

1614608参数；训练每记录1denoiser，1890000core/1890000在线F0特征/63000clean特征，原F0训练及源方差知识成本保留。部署固定8步Heun/末步Euler、15denoiser、2654576640静态实数MAC，另45个complex256点FFT/IFFT；不是实测时延，不把更少参数/更短训练等同同质量加速。方法、种子、源依据、完整继承和主文献边界见[预登记](report.md)。

独立检查84单元/2604组/1736raw配对/1302额外配对及868相对clean配对PASS，最大指标差1.24e-14、raw/DURA判决差0。完整60训练记录/stdout/sourceorder-draw/LR/加权loss/每轮5sigma与9mixture统计、实际同E0/同Gaussian RNG E01与E30/fixedsampler/骨干更新、30技术快照与E30-final一致性核实。全部prediction覆盖先于truth，复算0新神经前向/更新。

低档城区准确率56.502976%，为原raw clean准确率的65.292155%；距3pp线仍差45420条正确预测。这是准确率保留比例，不是IQ、发射机信息恢复比例或理论极限。

扩散先验可能生成源端常见细节，真实TX纠错、clean损伤、Fclean一致率与波形误差分别解释，闭集收益不能证明未知设备身份被保留。单模型seed与既有基准曝光是证据边界；该结果不回流本次参数/选模/重跑。[理论边界](../20261101-phase1-Restore-FeatureCond-E30-s2026092701-r01/theoretical_limits.md)。


## 实际噪声尺度曝光

| 训练版本 | sigma>10曝光 | 总曝光 | 比例 |
|---|---:|---:|---:|
| TF_ResidualDiffusion | 3322 | 1890000 | 0.175767% |
| TF_DiffusionSigmaMix | 356297 | 1890000 | 18.851693% |

新组9项mixture累计计数（0为原lognormal，1至8为固定部署sigma）为[943964, 118409, 118091, 118131, 118260, 118887, 118550, 117860, 117848]；对应8个sigma=[80.0, 34.9921875, 13.698575019836426, 4.637074947357178, 1.2866142988204956, 0.2674753665924072, 0.03518665209412575, 0.0020000000949949026]。原组全部为lognormal组件。完整每轮5区间/9组件计数见sigma_exposure.csv；不以平均训练loss降低推定部署有效。

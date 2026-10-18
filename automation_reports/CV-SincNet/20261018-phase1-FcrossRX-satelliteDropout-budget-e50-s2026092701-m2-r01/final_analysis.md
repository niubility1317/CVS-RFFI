# P62：完整 practical satelliteDropout 配方的 E50 匹配预算验证

状态：**ANALYZED／VERIFIED，missing=[]**。run：`20261018-phase1-FcrossRX-satelliteDropout-budget-e50-s2026092701-m2-r01`。训练、配置和分析代码 commit：`e5c25a54461439e6c33c76ce92ec50def7242524`。唯一模型 seed：2026092701；split：392005。报告于 2026-10-18T08:18:27.436Z 完成。

延长到 E50 确实改善了 DET 自身：正式 mid urban +1.357143、low urban +1.420833、urban 三场景均值 +1.086706 个百分点，clean +0.236905 个百分点。但 DET50 仍落后同预算 CE50 和原 As-RDrop50，不能将自身预算收益归因于关闭 satellite Dropout。相对匹配 NoAs50，DET50 clean +0.063095 个百分点，满足本种子“下降不超过 0.1 个绝对百分点”的阶段约束；该微小正值不足以证明稳定双提升。

**预登记源域判据 FAIL，保留原判定，关闭本配方扩展。** 两条新增行仍按计划完成全部正式评价。没有追加种子、E70、参数扫描、选择性重跑。low urban=52.027381%，尚未达到 60%；总目标未完成。

## 1. 唯一假设、匹配关系与执行基线

本轮只验证原 source PASS 的 satelliteDropout 配方能否在自身 E30 后追加 20 轮、累计 E50，保留相对同预算 CE／原 As 的增量。原源域 PASS 选择发生在新正式成绩之前：DET30 相对 NoAs30 fullV clean −0.007407 个百分点，相对 RD30／CE30 的 pooled mid/low 分别 +1.013889／+0.763889 个百分点；四个旧源判据通过。旧证据与不利分层见 [E30 源域背景](<E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-satelliteDropout-budget-e50-s2026092701-m2-r01/E30_source_context.md>)。

新训练只有 AS_CE、AS_DET 各一行，均严格加载自身原 SAT18 FINAL30 student 和 own EMA，继承各自 scratch20（13L＋7U）→自身 10 轮的合规来源，再追加 20 轮。不是从 E200 初始化，也不是把 E50 后再追加 50。NoAs／原 As 的 E30、E50 与 CE／DET 的 E30 共六个参考逻辑行，复用既有不可变预测和评分，无重训练、重预测或重评分。

- NoAs：匹配 F＋跨 RX 同 TX 对比学习，无星地增强。
- RD：原完整 practical 拼接配方，satellite Dropout 和 RDrop KL 保持。
- CE：只移除实际 satellite64 行的 RDrop KL，保留 Dropout。
- DET：在 CE 上进一步旁路实际 satellite64 行的现有 Dropout 输出，仍抽取原完整随机掩码；ordinary／raw／U 的 Dropout 和 RDrop 均保持。

网络仍为 F／SIG_TIME_X3／统一 Sinc／joint160／默认 cosine scale30 头，原内部门控、参考响应和联合更新保持。无新结构、新头、新教师或额外一致性损失。每步 ordinary128×2、mixed128×2、raw128×2，共 768 L 前向；mixed 含 64 As＋64 ordinary。总监督尺度 1.25，继承 label_smoothing=.01、原 CROSS_RX 权重 .1／ordinary128、旧 U 流程。完整 practical 作用于 raw IQ，再进入原网络，全部合法有标签 As 均参与 CE，不接受学生置信度过滤。

训练三场景为 practical_low_suburban、practical_mid_urban、practical_low_urban，各 1/3；信道流随累计 epoch／step 刷新。保留几何、遮挡、质量与补偿依赖，15 个 RX/day session 不由 TX 标签定义，不反演未知 TX 硬件。

数据 L6300／U56700（TX 真值隐藏）／V27000，source RX1/3/4/6/8、day1/2/3、六 TX，完整源域，无留一 RX／Ldev。严格匹配既有 source_contract role_ids 和 own20→30 来源，复用 VALIDATED_ONCE。正式 IQ、六场景、SNR 和困难样本保持。

实际导入的 leo_practical 是 PROJECT/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code/leo_practical/{channel,residual}.py，而非凭仓库同名文件判断。fs=25 MHz、IQ256、原逐包 complex RMS（center=false）在 normalize_input；训练 fc=2.462 GHz，正式固定 2.45 GHz。实际 residual／post_sync／equalization_enabled=false（G=1）／reference_snr_db=30／接收质量适配、相位跟踪、IQ 补偿开启／reflect。两行 actual_channel 与各自 SAT18 完全一致；保存六配置用于审计不代表训练采六场景。

追加段重置 AdamW／cos20／temporal cache，lr=.0002→.000001，WD=.0001；own EMA 严格加载、不 startup average。局部 E1–20／4440 更新对应累计 E31–50／6661–11100；ordinary 冻结用 scratch20 的 E13 参数，CROSS_RX epoch offset30。没有中途按目标成绩选模。

## 2. 源域先判定，正式评价后解释

源判据在新正式分数读取前固定并写入 source_matrix_decision.json，原记录未重算：

| 固定条件 | 实际值（绝对百分点） | 判定 |
| --- | ---: | --- |
| DET50 fullV clean−NoAs50 ≥−0.1 | −0.033333 | PASS |
| pooled mid/low DET50−CE50 ≥+0.5 | +0.138889 | **FAIL** |
| low DET50≥CE50 | +0.166667 | PASS |

总体 source_qualified=false。源 writer 的 formal_scores_read=false／target_scores_read=false 是当时写入的历史事实，不能误解为最终分析未读正式结果。科学 FAIL 不等于技术失败，也不取消既定正式测试。

| row | full V27000 clean % | 面板 low suburban % | 面板 mid urban % | 面板 low urban % |
| --- | --- | --- | --- | --- |
| NOAS30 | 98.359259 | 49.861111 | 43.555556 | 35.611111 |
| RD30 | 98.255556 | 88.666667 | 72.805556 | 59.083333 |
| CE30 | 98.329630 | 89.111111 | 73.055556 | 59.333333 |
| DET30 | 98.351852 | 89.805556 | 74.333333 | 59.583333 |
| NOAS50 | 98.448148 | 49.361111 | 43.666667 | 36.166667 |
| RD50 | 98.374074 | 90.111111 | 74.777778 | 60.722222 |
| CE50 | 98.425926 | 90.416667 | 75.472222 | 61.000000 |
| DET50 | 98.414815 | 91.138889 | 75.583333 | 61.166667 |

fullV27000 clean 与面板1800 clean 分开。各 satellite 面板为同一 1800 物理包×两个命名信道 draw，共3600次观测；不是两个模型 seed。源仅三训练场景，无源 high urban／all7。fullV没有独立 F1／错误复算，不能拿面板 F1 代替。源详细指标、F1、最差 RX、质量分层见 [source_summary.json](<E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_dropout_budget_E50_20261018_r01/retrieved_r01/analysis_r01/source_summary.json>) 和 [source_strata.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_dropout_budget_E50_20261018_r01/retrieved_r01/analysis_r01/source_strata.csv>)。

## 3. 全部正式逐行／逐种子结果

以下均为冻结 checkpoint 内默认头，accuracy 单位 %。每个固定视图168000包，八行×七视图全部完整。六参考行复用原评分；本次 CPU 分析在全部固定预测完成后独立连接 truth，1008 份混淆矩阵与168项分区复算一致。

| row（seed 2026092701） | clean | practical_high | practical_mid | practical_low_suburban | practical_high_urban | practical_mid_urban | practical_low_urban | urban 三场景均值 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NOAS30 | 85.247619 | 42.685119 | 40.995238 | 39.472619 | 41.439881 | 35.132738 | 30.139881 | 35.570833 |
| RD30 | 85.341071 | 81.405357 | 78.661310 | 75.861905 | 77.767262 | 62.806548 | 51.244048 | 63.939286 |
| CE30 | 85.214286 | 81.019643 | 78.345238 | 75.639286 | 77.452976 | 62.654762 | 51.170238 | 63.759325 |
| DET30 | 85.121429 | 79.198810 | 76.707738 | 74.018452 | 75.819048 | 61.638095 | 50.606548 | 62.687897 |
| NOAS50 | 85.295238 | 39.952381 | 39.140476 | 38.218452 | 38.994048 | 34.036905 | 29.878571 | 34.303175 |
| RD50 | 85.660714 | 82.239286 | 79.556548 | 76.776190 | 78.743452 | 63.901786 | 52.387500 | 65.010913 |
| CE50 | 85.347619 | 81.240476 | 78.811905 | 76.163095 | 77.869048 | 63.677381 | 52.486905 | 64.677778 |
| DET50 | 85.358333 | 79.568452 | 77.244643 | 74.714881 | 76.301190 | 62.995238 | 52.027381 | 63.774603 |

| DET50 相对 | clean Δpp | mid urban Δpp | low urban Δpp | urban 均值 Δpp |
| --- | --- | --- | --- | --- |
| DET30 | +0.236905 | +1.357143 | +1.420833 | +1.086706 |
| CE50 | +0.010714 | -0.682143 | -0.459524 | -0.903175 |
| RD50 | -0.302381 | -0.906548 | -0.360119 | -1.236310 |
| NOAS50 | +0.063095 | +28.958333 | +22.148810 | +29.471429 |

同预算 DET50−CE50 的 urban 三均值 −0.903175 个百分点，DET50−原 RD50 为 −1.236310；这与“更多预算改善 DET 自身”同时成立。各自预算效应不能冒充方法新增收益。

预算交互 (DET50−CE50)−(DET30−CE30)：clean +0.103571、mid urban +0.334524、low urban +0.104167、urban 均值 +0.168254 个百分点；相对差距缩小，但 E50 的方法水平仍负。原表见 [formal_interaction.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_dropout_budget_E50_20261018_r01/retrieved_r01/analysis_r01/formal_interaction.csv>)。

下表单元格为 accuracy／Macro-F1／最差 RX accuracy，统一单位 %。DET50 全部七视图最差 RX 均为14-7。

| row | clean | practical_high | practical_mid | practical_low_suburban | practical_high_urban | practical_mid_urban | practical_low_urban |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NOAS30 | 85.248／84.980／78.450 | 42.685／39.311／34.304 | 40.995／37.955／33.521 | 39.473／36.759／32.054 | 41.440／38.481／34.000 | 35.133／33.447／29.496 | 30.140／29.185／25.750 |
| RD30 | 85.341／85.044／78.362 | 81.405／81.084／71.908 | 78.661／78.360／70.213 | 75.862／75.573／67.742 | 77.767／77.423／69.108 | 62.807／62.450／56.458 | 51.244／50.887／46.379 |
| CE30 | 85.214／84.943／78.642 | 81.020／80.732／73.021 | 78.345／78.079／71.188 | 75.639／75.379／68.883 | 77.453／77.164／70.129 | 62.655／62.333／57.246 | 51.170／50.762／47.025 |
| DET30 | 85.121／84.837／77.737 | 79.199／78.692／71.713 | 76.708／76.243／69.771 | 74.018／73.595／67.542 | 75.819／75.322／68.888 | 61.638／61.229／56.458 | 50.607／50.276／46.675 |
| NOAS50 | 85.295／85.009／77.479 | 39.952／34.168／33.212 | 39.140／34.088／32.204 | 38.218／33.751／31.242 | 38.994／33.810／32.767 | 34.037／30.803／28.667 | 29.879／27.843／25.846 |
| RD50 | 85.661／85.381／79.025 | 82.239／82.031／72.279 | 79.557／79.342／70.288 | 76.776／76.524／67.625 | 78.743／78.521／69.708 | 63.902／63.536／57.246 | 52.387／51.849／47.233 |
| CE50 | 85.348／85.079／77.304 | 81.240／81.011／71.242 | 78.812／78.576／69.625 | 76.163／75.885／67.367 | 77.869／77.644／68.804 | 63.677／63.285／57.108 | 52.487／51.815／47.237 |
| DET50 | 85.358／85.089／76.621 | 79.568／79.170／70.521 | 77.245／76.867／68.637 | 74.715／74.327／66.104 | 76.301／75.897／67.567 | 62.995／62.531／56.263 | 52.027／51.513／46.808 |

总体 clean 微升仍有负分层：DET50−NoAs50 在 RX8-8 为 −1.025000、2021_03_01 为 −1.457143 个百分点；DET50−CE50 在 RX8-8 为 −.945833、2021_03_08 为 −.583333。后者 mid／low urban 在日期2021_03_08 为 −1.369048／−.730952；RX20-1 的 mid 为 −1.479167，RX7-7 的 low 为 −.754167。全部 RX／day／TX 见 [formal_metrics.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_dropout_budget_E50_20261018_r01/retrieved_r01/analysis_r01/formal_metrics.csv>)，没有挑分层掩盖负结果。

DET50 的三个 suburban 相对 clean 仍低 5.789881／8.113690／10.643452 个百分点；相对 CE50 又低 1.672024／1.567262／1.448214，未实现 suburban 与 clean 一并保持。

## 4. 错误修复、默认头间隔及机制边界

| DET50 对照 | clean 修复／新错 | mid urban 修复／新错 | low urban 修复／新错 | urban 三视图修复／新错 |
| --- | ---: | ---: | ---: | ---: |
| DET30 | 2183／1785 | 7521／5241 | 8856／6469 | 21182／15705 |
| CE50 | 1763／1745 | 6405／7551 | 8124／8896 | 17956／22508 |
| RD50 | 2292／2800 | 7919／9442 | 10228／10833 | 22120／28351 |
| NoAs50 | 3269／3163 | 54887／6237 | 45756／8546 | 166756／18220 |

urban 三视图为504000次观测，不是504000个独立物理样本。同 packet 配对记录见 [formal_pair_summary.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_dropout_budget_E50_20261018_r01/retrieved_r01/analysis_r01/formal_pair_summary.csv>) 和 [formal_paired_effects.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_dropout_budget_E50_20261018_r01/retrieved_r01/analysis_r01/formal_paired_effects.csv>)。

固定 NoAs30 clean 正确集合为143216包，所有 row 使用同一集合。间隔统一为默认头真类对最强竞争类 margin／30，不把 cosine 尺度变化当改进。

| view | DET30 margin/30 | DET50 margin/30 | CE50 margin/30 | RD50 margin/30 | DET30 固定集合保持率 % | DET50 固定集合保持率 % |
| --- | --- | --- | --- | --- | --- | --- |
| clean | 0.14829363 | 0.14390247 | 0.14021811 | 0.14053288 | 97.735588 | 97.509356 |
| practical_high | 0.08644633 | 0.09387641 | 0.11333988 | 0.11085795 | 87.992962 | 88.408418 |
| practical_mid | 0.07806575 | 0.08526088 | 0.10328314 | 0.10042156 | 84.961876 | 85.565160 |
| practical_low_suburban | 0.06971578 | 0.07666548 | 0.09318335 | 0.09021663 | 81.818372 | 82.598313 |
| practical_high_urban | 0.07885900 | 0.08588224 | 0.10373545 | 0.10150775 | 84.103731 | 84.676293 |
| practical_mid_urban | 0.04663284 | 0.05238903 | 0.06383610 | 0.06223049 | 67.771757 | 69.280667 |
| practical_low_urban | 0.02369349 | 0.02866556 | 0.03601580 | 0.03499403 | 55.118143 | 56.786951 |

DET50 相对 CE50 的 mid／low 间隔/30 反而 −.01144706／−.00735025，固定集合保持率 −.316305／−.254860 个百分点。没有重训诊断头。现有完整 gate／分支梯度逐样本变化未记录，不能据此宣称内部门控改变是原因；全局主干梯度健康见下节。正式状态标签不可用，不能把源状态关联直接当正式物理因果。

源域自身预算的主要正层：mid BLOCKED/degraded n656，+3.506098 个百分点（48修复／25新错）；low 同层 n414，+4.106280（30／13）。但同预算 DET50−CE50 在 low BLOCKED/degraded 为 −.724638（20／23），mid BLOCKED/unlocked n678 为 −.589971（50／54）；相对 RD50 的 low BLOCKED/unlocked n1565 为 −.447284（126／133）。因此能观察到更多训练改善部分遮挡且接收退化组，不能声称 DET 独立解决多径、CFO 或噪声。

信号／新增噪声比不是总 SNR。low 3–8 dB n794 的自身预算增益 +3.778338（60／30），但 low<3 dB n1127 相对 CE50／RD50 仍 −.621118／−2.040816。LOS/degraded、LOS/unlocked、BLOCKED/locked 空组保留；PARTIAL/unlocked n1／2不支持稳定结论。未开展移除物理损伤后重算接收态与固定接收态的双重因果消融，因此对白功率代理／FIR边界／噪声着色的独立因果仍未完成；本次不新增诊断前向。

## 5. 完整训练日志、覆盖与实际成本

读取新增40个epoch及8880条satellite step，未以日志抽样代替全量分析。epoch960个数值字段、step115440个数值字段无NaN／Inf；没有跳过更新、丢弃As或额外梯度门控。CE／DET 主干梯度 epoch 均值范围1.132391–1.643318／1.221330–1.764234。native loss .549700→.494249／.542047→.486573；源V仍有波动，预算耗尽不等于收敛。

每行4440步、284160个生成As、523个非空 TX×RX×scene×state×lock 单元，6TX×5RX×3scene 的90组合全部出现，15个RX/day session。相对原E50 RD的物理L/U序列每行8880记录差异0，ordinary MSE流差异0。旧参考缺失部分TX覆盖字段，保留 unavailable，不补造。源V不是训练数据。

两模式每步26次原Dropout调用；CE保留satelliteDropout，DET每步旁路3391488个元素，累计15058206720。DET的satellite双前向logit差与该半批KL均为0；CE对应均值5.304984／.076706。ordinary半批KL仍参与。loss分解最大误差约1.27e−7。extra_CE是mixed128 CE，未单独保存As-only CE，不画虚构单独As曲线。覆盖明细见 [training_coverage_summary.json](<E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_dropout_budget_E50_20261018_r01/retrieved_r01/analysis_r01/training_coverage_summary.json>)、[training_channel_strata.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_dropout_budget_E50_20261018_r01/retrieved_r01/analysis_r01/training_channel_strata.csv>)，逐epoch和step见 [training_curves.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_dropout_budget_E50_20261018_r01/retrieved_r01/analysis_r01/training_curves.csv>)／[satellite_steps.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_dropout_budget_E50_20261018_r01/retrieved_r01/analysis_r01/satellite_steps.csv>)。

| 方法 | 原E1–20训练墙钟秒 | 原E21–30秒 | E31–50秒 | 本次新增 |
| --- | ---: | ---: | ---: | --- |
| NoAs | 1329.394069 | 816.796109 | 1602.126106 | 否，复用 |
| RD | 1810.249311 | 1050.659807 | 2069.076337 | 否，复用 |
| CE | 1827.538153 | 1068.983741 | 2134.828152 | 是，20轮 |
| DET | 1823.967163 | 1083.303986 | 2141.861192 | 是，20轮 |

每新行原父30轮／6660更新，新增20轮／4440更新，累计50轮／11100更新。每新行native NN6271922（student train4543920、student eval594002、teacher eval1134000、teacher train0），新增L前向样本3409920；父30轮native NN7960684，累计50轮训练native NN14232606。NN计数单位为前向样本，不是batch或独立物理包。

本次两行新增40模型轮／8880更新、L前向6819840、生成As568320；source读出25200 NN、正式预测2352004 NN、真实父smoke4 NN，共**新增远端14921052 NN**。两个模式本地合成检查另计1536 NN，来自实际登记；不可变costs.json.local_verification_NN=null，报告补充而不回写原分析。无额外模型参数或分析训练更新。

新增训练墙钟合计4276.689344秒；正式CE／DET49.217404／48.570763秒，共97.788166秒；CPU全量分析65.657922秒，NN／IQ／checkpoint／更新均0。两GPU并行进程墙钟之和不等于总历时。源读出墙钟未单独汇总，不补造。

去重共同父后，历史160模型轮／35520更新，加本次得到200模型轮／44400更新；已知远端累计66439244 NN。历史共享smoke未分摊，不能宣称这是所有历史操作的完整总额。六参考的重预测／重评分均0。逐段父成本和去重清单见 [costs.json](<E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_dropout_budget_E50_20261018_r01/retrieved_r01/analysis_r01/costs.json>)。

## 6. 实现、验证、命令和实际产物

代码 commit e5c25a54461439e6c33c76ce92ec50def7242524 包含 experiments/cvs_satellite_budget62/（design／reference_config／source／model／channel／runner／dispatch／qualification／evaluate／observer等）、analysis/cvs_satellite_budget62_analysis_20261018.py 及定向检查。没有改动其他健康实验。工作区未提交的 P61 margin／source_grad 草稿及无关索引修改保留。

执行过的验证：33项协议负例、4项容量toy、4项流程检查；两个实际F合成模式共1536 NN；ordinary logits和RNG匹配，默认头／Sinc梯度finite非零，ownEMA／scratch20 E13／reset AdamW检查；本地18文件及observer单文件compile；分析21项toy（1.508秒）与source schema／coverage定向修复，NN0。唯一独立P0/P1审查PASS，修复source schema、cap2、early/terminal source异常闭合后只定向核实，没有重复全审。

一次普通账号N607预检，单release archive SHA一次本远端比较，一次远端20py compile PASS；两个真实自身父checkpoint无query smoke共4NN／2CP／0IQ／0更新后直接训练。独立核实dispatcher1276889、CE1276916/GPU2、DET1276994/GPU3的PID/CWD/argv/CUDA和日志增长；终态自然退出，不是终止。release归档比较VERIFIED，训练／七预测／独立评分／CPU分析VERIFIED。详细历史时间和路径见 verification.json。无管理员操作。

精确执行入口（PROJECT为既有项目根，保护本机私有运行细节）：

```text
CWD=PROJECT/releases/cvs_FcrossRX_satellite_dropout_budget_E50_20261018_r01
CVS-RFFI/python -u experiments/cvs_satellite_budget62/runner.py
自动分析：analysis/cvs_satellite_budget62_analysis_20261018.py --project PROJECT --out BASE/analysis_r01
输出：PROJECT/runs/20261018-phase1-FcrossRX-satelliteDropout-budget-e50-s2026092701-m2-r01
环境：Python3.10.19 / Torch2.1.0+cu121 / NumPy2.2.5，原pinned runtime，FP32
```

全部实际图已生成并逐张查看，PNG和同名PDF均保存：

- [全部七正式环境曲线](<E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_dropout_budget_E50_20261018_r01/retrieved_r01/analysis_r01/formal_all7_accuracy.png>)。
- [E30→E50配对预算效应与交互](<E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_dropout_budget_E50_20261018_r01/retrieved_r01/analysis_r01/paired_budget_effects.png>)。
- [E31–E50完整训练曲线](<E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_dropout_budget_E50_20261018_r01/retrieved_r01/analysis_r01/E31_E50_training_curves.png>)。
- [mid urban源状态分层](<E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_dropout_budget_E50_20261018_r01/retrieved_r01/analysis_r01/practical_mid_urban_source_states.png>)。
- [low urban源状态分层](<E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_dropout_budget_E50_20261018_r01/retrieved_r01/analysis_r01/practical_low_urban_source_states.png>)。

机器可核查汇总：[complete.json](<E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_dropout_budget_E50_20261018_r01/retrieved_r01/analysis_r01/complete.json>)（ANALYZED、missing=[]）；[formal_summary.json](<E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_dropout_budget_E50_20261018_r01/retrieved_r01/analysis_r01/formal_summary.json>)；[new_training.json](<E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_dropout_budget_E50_20261018_r01/retrieved_r01/analysis_r01/new_training.json>)。所有图、聚合CSV、原预测与权重保持产物目录；不提交IQ／权重／日志／预测。

## 7. 跨种子与未完成项

P62只有一个模型seed，两draw不构成跨seed复现，无法估计P62配对波动。原完整As E50另有已完成三seed结果：mean clean84.563889%、mid62.989484%、low51.926587%、urban64.062831%；相对匹配NoAs50 clean平均−.150198±.955701个百分点，逐seed为+.365476／+.436905／−1.252976，阶段约束未满足。见 [原配方E50三seed报告](<E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-continuous-recovery-e30-s2026092701-m1-r01/E50_three_seed_readout.md>)。不能把原seed01或本次+.063095微升挑出来宣称可重复双提升。

本次已完成固定两行和所有预登记输出；仍未获得low urban>60%、可重复clean正收益或满足0.1pp约束的跨seed证据，物理损伤因果定位与完整gate逐样本机制记录亦不完整。预先源判据失败后不以正式结果救活方案。后续新机制必须先核对另一聊天“审计并设计 CVS 联合增强”的最新实际路线，避免重复；本轮未发送跨聊天消息、未启动新候选。

最终判断：更多预算缓解了源域部分BLOCKED/degraded观测上的错误，正式默认头low提高1.420833个百分点（相对自身E30）；相对同预算无星地增强low提高22.148810、urban均值提高29.471429，clean没有总体下降而微升.063095个百分点。但同预算原As／CE更好，本次只支持预算收益，不支持DET机制优越性，也没有可重复双提升证据。

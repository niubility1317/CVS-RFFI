# 纯 F E20 同轮首次 As 重访：完整结果

**结论：固定配方 SOURCE_FAIL，关闭，不追加 E50。** 完整训练、冻结源读出、7 视图预测和独立 truth-last 评分均完成；这是科学负结果，未发现技术失败。low urban 为 52.555952%，clean 为 81.397024%；相对普通 As 分别 −0.079762 和 +0.185714 个百分点，clean 相对原 NoAs 仍低 2.767262 个百分点。目标未达成。

## 身份与比较口径

run：`20261019-phase1-F-epochReplay-e20-s2026092701-m1-r01`；model seed 2026092701；split/data seed 392005；代码/config `39db239b9f7ac479a84080ff1b0f5ae363feb76c`。M 从零 E20＝13L＋7U、4440 次更新；B0/B1 复用 `20261019-phase1-F-nontrue-CSIL-e20-s2026092701-m4-r01` 的对应完整固定 FINAL20。三组初始 206 个 state tensor 完全相等，完整 L/U 实际顺序相等；不假定全训练 RNG 或墙钟相等。

固定纯 F：SIG_TIME_X3＋CENTER_25K＋CENTER_UNITS，NoExplicitCFO，原内部融合与 6×160 cosine 默认头联合训练。无 CROSS_RX、RDrop、外部 CSIL 或额外分类头。L6300/U56700 隐藏 TX/V27000，RX1/3/4/6/8、day1/2/3；正式每视图同一 168000 样本、6 类、未见 RX 与原 received IQ。测试 benchmark 已暴露，不称为新盲测。

唯一变化为每 epoch 每 physicalID 的第一次 full As 被缓存并跨更新重访；下一 epoch 清空。每步仍生成原 fresh128，命中时仅作 shadow。loss 0.5×ordinary CE＋0.5×used-As CE，均 mean128、T1、smoothing0.01；原 U/ownEMA 保留。改变同时包含时间相关性与有效观测多样性，不能只说多看几次且其他分布完全不变。

## 先固定的源决定

source decision timestamp 1792396503.4211354；源判定明确 formal_scores_read=false，且先于本 row 独立评分落盘。mid/low V1800×2draw 相对 B1 为 −0.416667 个百分点（门槛＋0.5），fullV27000 clean 相对 B0 为 −0.277778 个百分点（容差−0.2）；独立重算一致。两项均未通过。固定 epoch-first 配方关闭，不调整 refresh 间隔、seed 或日程救援。

|模型|full V27000 clean|源 low suburban 两 draw 均值|源 mid urban 两 draw 均值|源 low urban 两 draw 均值|
|---|---:|---:|---:|---:|
|B0|98.055556%|62.694444%|52.777778%|42.750000%|
|B1|97.762963%|89.666667%|75.638889%|62.138889%|
|EPOCH_REPLAY|97.777778%|89.777778%|75.361111%|61.583333%|

fullV27000 clean 与 V1800 面板 clean 是不同统计；两个信道 draw 不算两个模型种子。源失败仍按预登记完成正式评分，正式分数没有用于改判或延长。

## 七种视图的完整指标

下表各单元为 accuracy / macro F1 / 最差 RX accuracy，单位均为 %。

|视图|B0 NoAs|B1 普通 As|M 同轮重访|
|---|---:|---:|---:|
|clean|84.164286 / 83.847123 / 74.312500|81.211310 / 80.612354 / 67.750000|81.397024 / 80.799448 / 69.762500|
|practical_high|51.095238 / 49.431135 / 40.433333|80.755357 / 80.328296 / 74.033333|80.796429 / 80.375901 / 74.787500|
|practical_mid|49.149405 / 47.583552 / 39.641667|78.355357 / 77.994058 / 72.470833|78.460714 / 78.118800 / 73.266667|
|practical_low_suburban|47.230357 / 45.803993 / 38.862500|75.922619 / 75.574439 / 70.600000|75.882738 / 75.539468 / 70.958333|
|practical_high_urban|48.966071 / 47.210610 / 39.141667|77.277381 / 76.901938 / 70.766667|77.394643 / 77.038449 / 71.716667|
|practical_mid_urban|40.274405 / 38.674862 / 33.783333|63.472619 / 63.164196 / 58.645833|63.515476 / 63.261052 / 59.270833|
|practical_low_urban|33.649405 / 32.215066 / 29.345833|52.635714 / 52.145383 / 48.816667|52.555952 / 52.166470 / 49.320833|

## 配对修正、新增错误与种子统计

每项为同一 seed、相同 168000 物理样本的配对比较。模型 seed n=1，配对均值就是表中百分点差；样本 SD 无法估计。正差数为 0/1 或 1/1，不构成可重复收益。

### M 相对 B0

|视图|修正错误|新增错误|净变化（百分点）|正差 seed 数|
|---|---:|---:|---:|---:|
|clean|3439|8088|-2.767262|0/1|
|practical_high|55165|5267|29.701190|1/1|
|practical_mid|55101|5858|29.311310|1/1|
|practical_low_suburban|54748|6612|28.652381|1/1|
|practical_high_urban|53673|5913|28.428571|1/1|
|practical_mid_urban|47924|8879|23.241071|1/1|
|practical_low_urban|43087|11324|18.906548|1/1|

### M 相对 B1

|视图|修正错误|新增错误|净变化（百分点）|正差 seed 数|
|---|---:|---:|---:|---:|
|clean|1656|1344|0.185714|1/1|
|practical_high|2054|1985|0.041071|1/1|
|practical_mid|2454|2277|0.105357|1/1|
|practical_low_suburban|2723|2790|-0.039881|0/1|
|practical_high_urban|2415|2218|0.117262|1/1|
|practical_mid_urban|3708|3636|0.042857|1/1|
|practical_low_urban|4519|4653|-0.079762|0/1|

## Suburban 与自身 clean 的差距

|模型|high suburban|mid suburban|low suburban|
|---|---:|---:|---:|
|B0|33.069048|35.014881|36.933929|
|B1|0.455952|2.855952|5.288690|
|EPOCH_REPLAY|0.600595|2.936310|5.514286|

M 三场景绝对准确率为 80.796429%、78.460714%、75.882738%；自身 clean81.397024%。高 suburban 差距小不代表实现双提升；M clean 仍显著低于原 B0。

固定 B0 clean 正确集合 n=141396，所有模型/视图使用同一集合。完整逐视图保持率、RX/day/TX 分层和 confusion matrix 保存在本地 complete_analysis.json，未用这些目标分层选择后续方案。

## 完整执行与实际特征更新

完整读取 mechanism/replay/ordinary/unit/native 各4440条、两个20轮CSV及20轮伴随记录。568320包缓存来源逐项与训练采样 physicalID、origin step/slot、namespace、scene 对齐，首个矛盾为 null；ordinary 第一前向与 used-As 第二前向实际输入 witness 全4440步通过。所有128个有标签 As 均参与真实 CE，没有学生置信门控。原U真实1554步、U学生396900包、自身EMA396900包，全部4440次EMA更新计数正确。Sinc的物理单位、eps、显式decay与实际全局clip核对通过。

实际 joint160 每次前向有限，记录12767次、2127542包，默认权重形状6×160。Sinc坐标、三个signature局部投影/输出投影、时频/PA卷积、reference投影和默认头均有实际参数变化；不是仅更新融合或分类器。但非零更新不等于识别改善。没有新增部署参数，缓存不进入checkpoint，eval不使用缓存。

每轮 unique6300、hit22116；20轮 unique epoch观测126000、hit442320，均为跨step命中，within-step hit0。77.829392%的used-As曝光是缓存命中；全部命中均与同次fresh IQ不同，平均fresh-used MSE0.576152。E20在线As all20914/28416=73.599381%，first4637/6300=73.603175%，hit16277/22116=73.598300%；末轮没有可见的hit正确率优势。在线训练数不等于冻结模型对固定L观测的拟合。

## 冻结拟合诊断与限制

独立诊断 `20261019-diagnostic-F-epochReplay-sourceFit-e20-s2026092701-m2-r01` 已完成，0更新。固定复用DR180 L物理ID与step/slot；180个M缓存origin与旧seen坐标完全相同，直接复用两侧对应预测，B1不再加载。实际新NN1441＝M1440＋smoke1，最大授权1801；新V前向0，权重/缓冲不变、梯度None，source smoke PASS。历史As IQ没有归档，只能按原Config/session/seed重建，不声称历史字节验证。

M的low urban固定重建观测45/69，与B1同为45/69；同69个L的新信道两draw为M 85/138。M对全部180 L×2draw的low urban为222/360，与B1同为222/360。该固定面板未显示重复首次观测带来low拟合收益，也不能据此推断全L拟合或物理上限。完整CE、配对修正/新错与V新观测见配套诊断 final_analysis.md。

## 成本、证据与交付

|新增M项目|实际数量|
|---|---:|
|L_student_NN|1136640|
|U_student_NN|396900|
|ownEMA_NN|396900|
|external_teacher_NN|0|
|source_training_eval_NN_actual|594002|
|ordinary_generated_and_used|568320|
|ordinary_shadow|0|
|fresh_As_generated|568320|
|used_As_NN|568320|
|unique_epoch_As_realizations|126000|
|fresh_As_discarded_as_hit_shadow|442320|
|fresh_As_changed_on_hit|442320|
|source_qualification_NN|12600|
|formal_prediction_NN|1176002|
|training_wall_seconds|2148.785090446472|
|extra_analysis_NN|0|

B0训练L NN568320；B1/M训练L NN1136640。旧B1另有568320外部CSIL eval shadow仅作为历史成本披露，不重算；其构建load用fork_rng保护、eval不消费随机流。M没有该外部教师。训练墙钟2148.785秒；冻结诊断墙钟13.452秒，教师/旧对照训练成本没有冒充免费。分析0新模型前向、0新prediction、0新scoring。

独立post-state证据：原worker2187003/GPU4已完成；诊断PID2210590/GPU6的CWD/argv/CUDA6/CPU2及NVML已在运行时核实，日志0→115字节，最终complete.json与smoke PASS独立读回。正常结束不重启。新代码/config提交39db239b9（训练）、3145fad02（源诊断）；后续交付提交45e091679已push且远端OID核对一致。分析连接入口曾因相对导入失败，已按包名导入修复，随后真实完整汇总PASS；未重训或重测。

本地权威分析：`local_artifacts/F_epochReplay_E20_goal_20261019_complete/complete_analysis.json`；源诊断 `local_artifacts/F_epochReplay_sourceFit_E20_20261019/analysis.json`。远端原始完整日志/CP/predictions/scoring保留于 PROJECT/runs/20261019-phase1-F-epochReplay-e20-s2026092701-m1-r01 及诊断run，未提交原始日志、checkpoint、IQ或predictions。

图件：[七视图与训练曲线](E:/type10-7/exports/cvs_pureF_epochReplay_20261019/seven_views_and_online_As.png)、[配对误差与固定集合保持](E:/type10-7/exports/cvs_pureF_epochReplay_20261019/paired_errors_and_B0_clean_retention.png)、[冻结源拟合](E:/type10-7/exports/cvs_pureF_epochReplay_20261019/source_fit_accuracy.png)、[冻结源CE](E:/type10-7/exports/cvs_pureF_epochReplay_20261019/source_fit_CE.png)、[同IQ配对差](E:/type10-7/exports/cvs_pureF_epochReplay_20261019/source_fit_common_IQ_pairs.png)，相应PDF同目录。图件已真实生成并逐PNG视检。

## 后续范围

用户新方案明确转向双教师＋浅中层时序残差适配器：先源端确认Ts互补能力，再以既定合法clean F E20初始化，首轮冻结原F和默认类别权重，只训新增特征模块。学生追加10轮、累计E30已由用户确认。新方案不是本次失败缓存配方的间隔/seed/系数救援；其设计和教师选择继续由合法源证据决定。当前整体目标仍未完成，不把单seed微小变化或损失下降当作突破。

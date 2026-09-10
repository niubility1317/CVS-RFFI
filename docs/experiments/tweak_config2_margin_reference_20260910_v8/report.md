# Tweak V8：参考文献引导的margin采样解释验证

状态：ANALYZED / VERIFIED；100epoch、42个产物及20行独立评分完整。校准后性能明显改善，但严格复现与无校准跨配置迁移均未达成。用户于2026-09-10要求继续定位、优化并发布实验验证。

## 结论与单变量假设

V7已经排除欧氏距离MM相消导致的评分错误，但其严格hard目标仍持续奖励全局缩尺度。对固定且非空的严格hard集合，令`delta=dAP²-dAN²>0`，则`L(s*z)=s²*mean(delta)+0.1`，故任何正尺度下径向梯度都倾向缩小尺度；margin既不影响严格筛选，也不影响该集合内的梯度。这个论证适用于严格hard目标，不能把问题专归于“全量”采样，也不证明缩尺度是所有低分的唯一原因。

重新读取Tweak原PDF第6/9页确认其文本确实描述`dAN<dAP`；但它在hard mining处引用的[35]PVSNet第4页§2.3实际描述选取违反margin的负样本。[PVSNet原文](https://arxiv.org/pdf/1812.06271)还使用了归一化、预训练和自适应margin，但这些不属于Tweak明确设置，本轮全部不引入。

V8只检验一个解释差异：选集从`dAP²-dAN²>0`改为`dAP²-dAN²+0.1>0`，即在严格hard之外加入仍违反margin的半难triplet。loss仍是Eq.(1)平方L2 hinge、margin=.1，仍平均batch内选中的有效有向triplet，不改变网络或IQ。明确标记`REFERENCE_GUIDED_MINING_DIAGNOSTIC_NOT_STRICT_REPRODUCTION`，不能当成已验证作者代码或严格复现；默认`strict_hard`路径及V1—V7全部保留。FaceNet原论文同样讨论过极难负样本的退化风险，但本轮不是FaceNet改造，不引入其归一化/半难独占采样。[FaceNet原文](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf)

## 本地source-only因果诊断

只读Config2既定训练池、不读取测试帧。对照均从seed=20260908随机初始化，实际batch序列seed=20260909、8类×8样本、SGD momentum=.9、LR=.001、3,000batch；两者使用同一诊断脚本和相同固定source训练池监控。没有继承历史checkpoint。Torch2.10.0+cu128，RTX5070 Ti；未启用全确定性算法，短诊断不是统计显著性证据。

| 采样 | step1000 source监控 | step3000 source监控 | step3000严格hard占比 | 平均类中心距 | 平均类半径 |
|---|---:|---:|---:|---:|---:|
| strict | 52.8% | 56.4% | 44.380% | 0.0435433 | 0.0574704 |
| margin | 63.6% | 64.4% | 28.695% | 0.3068631 | 0.2548977 |

step3000的loss分别为.1021685和.1256986，二者选集不同，不能以loss高低比较优劣。类中心距/半径比约.758→1.204，支持继续检验更好的source可分性，但不证明target可移植性或长期不会退化。

新生产函数在LR=.01的额外3,000batch真实数据验证全程loss/16组梯度有限，step3000实际选中15,428个triplet，其中7,174个严格hard、8,254个半难；source监控仅29.2%，说明不能凭直觉选大学习率。正式仍沿用五个fresh probe和source监控选LR，不从target分数选择。

4项新回归先失败后通过；连同既有数值、输入边界、来源重置、预测不可覆盖和校准测试，相关36项通过。新测试独立手算6个违反margin的triplet均值`.365/6`，验证非零更新与扩尺度梯度；无违反margin样本时零loss且跳过动量更新；默认严格路径不变；新metadata禁止误报paper parity。

一次独立P0/P1发布审查已完成，无阻断项；核验CLI选项贯通全部probe和正式训练，scratch/source-only路径、20行预测与diagnostic标签一致。此审查不构成性能或严格复现认可。修改模块编译通过，最终相关36项测试无warning。

## 预登记矩阵、输入权限与科学边界

- 模型、数据、seed、split、SGD、学习率grid、source监控、100epoch和20行矩阵均延续V7；唯一科学变量为上述筛选条件。没有L2归一化、CE、额外数据增强、teacher或旧模型初始化。
- source=Config2、TX1…10；每record156,250帧，75%训练、校准为训练帧10%、25%测试，物理仿射split与V7一致。M=10，单配置校准4×4，联合四配置校准4行。此为官方LoRa外部对比诊断，不是CVS LEO/Stage2结果。
- Checkpoint输入：只允许本run从零生成。五个1epoch probe每次恢复相同随机初始state且重建SGD；选LR后再次恢复随机初始state，训练100epoch。禁止初始化/resume/teacher/EMA来自V1—V7或其他run；argv无任何加载入口。训练元数据直接记录`initialization=scratch`和实际miner。
- LR与checkpoint只看固定Config2训练池监控：参考逻辑帧0…255，监控256…505，最后6帧不计入M=10。不是独立validation；不得把训练池64.4%当作正式性能。
- 预测先写入每行独立`.pt`，再写truth；完成后独立scorer逐ID复算所有20行，不回流选择本次checkpoint。重复使用的公开benchmark属于研发探索，不宣称新数据独立确认或多seed显著性。
- 判定：完成100epoch不等于数值复现。比较V7同row、论文图13b/14近似值，报告平均、逐配置与弱类；若只有source改善而target无改善，结论为“训练侧假设部分支持，迁移未改善”，不自动发起下一轮。

## 发布与技术停止

代码commit=`d69640e12acc3a9792b01a9c0f5a07d1953a20e3`，push后独立读取远端branch OID与本地HEAD一致。N607根=`/home/szu2070436088/2510044040/CV-SincNet`；release=`releases/tweak_config2_margin_reference_20260910_v8/source`，run=`runs/tweak_config2_margin_reference_20260910_v8/official_config2_full`，log=`logs/tweak_config2_margin_reference_20260910_v8/train.log`。数据=`datasets/tweak_official_lora_configurations_20260908/Diff_Configurations_Setup`。

计划使用物理GPU2，preflight21:31 CST显示空闲（1MiB）；启动前再次核对该卡进程，最多2个训练实验，保护其他GPU上的健康作业。唯一launch owner为当前主Agent。

命令：`CUDA_VISIBLE_DEVICES=2 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m paper_reproduction.gaskin_tweak_2023.official_lora_experiment --data-root /home/szu2070436088/2510044040/CV-SincNet/datasets/tweak_official_lora_configurations_20260908/Diff_Configurations_Setup --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/tweak_config2_margin_reference_20260910_v8/official_config2_full --device cuda:0 --triplet-mining margin_violating`。

仅本run确定性系统故障、非有限训练、输入越界、输出碰撞或无法形成合法预测可触发技术停止；低性能不停止，不影响其他进程。新release和output不可覆盖。预期产物42个：checkpoint、results、20prediction和20truth，另有完整训练日志。训练日志每个probe/epoch记录首batch的hard、半难及实际选中数，作为机制实际启用证据。

## 实际落地与启动读回

- 源归档81,920字节，本地/远端SHA256均为`e7c1405cdcbfb831c0bb26b89e1c9ae3913b7947341b6a941a2071388bee7775`。40条数据及metadata存在，远端10个模块与诊断工具编译成功。
- N607真实CUDA source前反传：`[64,2,128]→[64,12]`，loss=.4642926455，16组参数梯度有限；25,088个有效triplet中，12,570个严格hard、3,758个半难，实际margin选中16,328个。这是source smoke，不是目标准确率。
- 启动前GPU2已有另一个PID1139396，未干预；确认该卡不足2个compute进程后添加本run。正式PID1140324，PPID1，物理GPU2；41秒独立读回CWD、cmdline及GPU UUID一致，CPU115%、448MiB显存。同卡两个训练进程，未超上限。
- 实际argv包含`--triplet-mining margin_violating`，没有checkpoint/resume入口，没有smoke batch或epoch限制。首次读回log为0字节：runner整epoch输出，首probe尚未完成；因此目前仅验证进程启动，不将空日志视为训练完成或完整健康证据。
- 已创建30分钟监控`tweak-v8`：健康/无可行动变化时静默，完成后下载全部42个产物和完整日志、独立复算20行并更新本报告；完成交付后关闭监控。不会因低分自动启动V9。

## 完整训练与独立评分（2026-09-11）

01:27 CST只读检查确认原PID1140324退出、日志以complete结束、42个文件齐全。全部文件下载到本报告根目录`readback/official_config2_full/`，完整57,104字节日志保存为`train.log`。没有重启、修改远端结果或干预其他GPU任务。

按training-log-analysis完整解析106条JSON：5个fresh LR probe、连续E1—E100和1条complete，没有非JSON异常或traceback。所有probe/epoch均18,310个active batch，合计1,922,550次；全部日志训练记录与results逐字段一致，数值有限。每条首batch记录均满足`selected=margin_violating=hard+semihard`，105条记录的半难数范围3,349—11,785，全部大于0；这证明实际参与，不只是CLI配置。遥测只记录首batch计数，不推断所有batch的半难比例。

五个1epoch probe的LR与source训练池监控分别为`.01:76.4%`、`.001:76.0%`、`.0001:77.6%`、`.00001:70.4%`、`.000001:60.4%`，均满足窗口loss下降。选中LR=.0001；正式重新随机初始化训练，source范围80.0%—88.4%，首次最高值在E83，checkpoint正确选择E83。checkpoint的`training`与results完全一致，包含`initialization=scratch`、`triplet_mining=margin_violating`和诊断标签；所有权重有限。checkpoint没有单独的`method_metadata`成员，诊断状态保存在`training.parity_status`，与results方法元数据及complete事件一致。

| 时点 | mean loss | source训练池监控 | 平均中心距 | 平均半径 |
|---|---:|---:|---:|---:|
| E1 | 0.1053241921 | 80.0% | 0.20800882 | 0.18693651 |
| E83（选中） | 0.0999959164 | 88.4% | 0.26879305 | 0.19458568 |
| E100 | 0.1003407237 | 84.4% | 0.26589266 | 0.19010659 |

V7选中E79的source监控64.4%、中心距.00120948、半径.00163267；V8不再呈现同样的持续缩尺度轨迹，支持训练侧可分性假设。但正式V7/V8的自动LR选择结果分别为.001/.0001，checkpoint轮次也不同：这是相同选模规则下miner改变的整体结果，不能当作固定LR、固定epoch的纯梯度消融。source监控不是独立validation；单seed与重复公开benchmark探索不构成显著性或新数据独立确认。

独立工具[score_tweak_saved_predictions.py](../../../tools/score_tweak_saved_predictions.py)不导入生产预测器，以NumPy float64显式差分复算全部20行Algorithm2，逐行先验证保存预测，再按唯一opaque ID连接truth。每行39,060个M=10判定，累计781,200次，预测差异全部为0，正确数20/20一致，results浮点accuracy与整数比差小于5e-8。校准行共享测试样本，累计判定数不是独立样本量。完整独立结果含逐类准确率、预测计数和球内比例，见[independent_scores.json](independent_scores.json)；原始结果与完整日志的Git副本见[results.json](results.json)、[train.jsonl](train.jsonl)。二进制checkpoint、预测和truth保留在本地读回目录与N607，不上传Git。

## 全部20行同row对比

模型仅在Config2训练；左侧是校准配置，右侧是测试配置。单位为%，差值为百分点；每行分母39,060。

| 校准→测试 | V7 | V8 | V8−V7 |
|---|---:|---:|---:|
| Config1→Config1 | 45.773 | 79.598 | +33.825 |
| Config1→Config2 | 4.373 | 11.142 | +6.769 |
| Config1→Config3 | 13.013 | 14.913 | +1.900 |
| Config1→Config4 | 0.246 | 0.013 | −0.233 |
| Config2→Config1 | 7.696 | 13.804 | +6.109 |
| Config2→Config2 | 62.906 | 85.963 | +23.057 |
| Config2→Config3 | 15.177 | 10.561 | −4.616 |
| Config2→Config4 | 12.002 | 10.000 | −2.002 |
| Config3→Config1 | 16.039 | 17.156 | +1.116 |
| Config3→Config2 | 2.138 | 18.666 | +16.528 |
| Config3→Config3 | 31.011 | 72.972 | +41.961 |
| Config3→Config4 | 12.058 | 10.481 | −1.577 |
| Config4→Config1 | 5.420 | 9.982 | +4.562 |
| Config4→Config2 | 7.606 | 10.000 | +2.394 |
| Config4→Config3 | 12.424 | 13.067 | +0.643 |
| Config4→Config4 | 28.090 | 65.100 | +37.010 |
| 联合→Config1 | 29.329 | 54.091 | +24.762 |
| 联合→Config2 | 46.272 | 79.862 | +33.589 |
| 联合→Config3 | 18.367 | 49.055 | +30.689 |
| 联合→Config4 | 17.906 | 49.805 | +31.900 |

同配置四行平均41.945%→75.908%；联合四行平均27.969%→58.203%；跨配置12行平均仅9.016%→11.649%。因此不是“只有source改善”：未用于网络训练的Config1/3/4在合法同配置校准后也明显改善；但这依赖对应配置的校准数据，不是无校准域不变性。

论文原PDF图13b同配置柱近似75%/90%/74%/68%，V8差约+4.598/−4.037/−1.028/−2.900个百分点；图14联合近似56%/82%/56%/53%，V8差约−1.909/−2.138/−6.945/−3.195个百分点。沿用V7已记录的原PDF读图近似值，不是作者精确表格或预设容差；不能以接近部分柱或均值接近宣称数值复现成功。尤其本轮主动改变了论文文本的严格hard筛选，必须保留`REFERENCE_GUIDED_MINING_DIAGNOSTIC_NOT_STRICT_REPRODUCTION`。

## 剩余问题与结论边界

1. **校准坐标仍严重不跨域。**Config2→Config4全部39,060次预测为TX10，球内比例0；Config1→Config4仅5次正确，预测只落在TX3/TX10。独立显式差分验证相同，不能归咎于已修复的距离MM相消。联合校准球内比例超过99.9%也不等于正确分离，跨域球重叠/竞争仍可能影响分类；当前结果未隔离唯一原因。
2. **弱类未解决。**同配置Config1/2/3/4最低类分别为TX4:19.944%、TX4:61.982%、TX3:30.876%、TX2:27.829%；联合对应最低类为TX4:15.156%、TX7:58.372%、TX6:11.086%、TX8:13.978%。这些是完整逐类评分，不用于类特定调参或选择性重跑。
3. **资源与稳定性范围。**100轮完整、全日志有限、没有已观测技术失败；此前定时GPU读回642MiB只是快照，不是峰值。未记录逐epoch计时/完整GPU遥测，不能给出精确训练时长、峰值或吞吐优势；单次稳定完成不等于长期稳定性验证。
4. **科学判定。**margin解释分支在本次同流程下改善source与校准后目标识别，支持继续理解严格hard的训练退化机制；并未证明作者实际采用此miner，也未解决无校准迁移或达到严格复现。此为官方LoRa外部诊断，不是CVS LEO/Stage2晋级证据。

本轮分析只消费固定产物，不把测试结果回流到LR、checkpoint或自动重跑。V1—V8全部保留，不启动V9；报告交付后删除`tweak-v8`监控。若另行开展研究，需先明确是严格论文复现还是独立方法优化，并在新的授权/预登记中界定，不能在本次监控中静默扩展。

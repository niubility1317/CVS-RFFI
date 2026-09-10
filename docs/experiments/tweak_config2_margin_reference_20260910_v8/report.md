# Tweak V8：参考文献引导的margin采样解释验证

状态：RUNNING / PROCESS_START_VERIFIED；已发布并读回进程身份，完整训练和评分未完成。用户于2026-09-10要求继续定位、优化并发布实验验证。

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

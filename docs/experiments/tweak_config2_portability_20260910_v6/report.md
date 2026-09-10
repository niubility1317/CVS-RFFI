# Tweak配置可移植性复现实验V6：mini-batch全量严格hard mining

- run_id：`tweak_config2_portability_20260910_v6`
- 当前状态：`ARTIFACTS_COMPLETE / ANALYZED / NUMERICAL_REPRODUCTION_FAILED`
- 唯一launch owner：Codex主Agent
- Git代码提交：`a234e6e6136e558fda0b47b910037662b36d3bd4`，已push且远端OID独立一致。

## 依据与单变量假设

- V5已纠正论文Eq.(1)平方L2并使对角准确率由约10—28%提高到27—47%，但其embedding仍塌缩，说明平方公式不是唯一失配。
- 论文IV-A明确要求在mini-batch经过网络后选出“triplets”中`A`比`P`更接近`N`的hard triplets。V5的每anchor一个随机正/负候选是未公开默认，且只覆盖batch内25,088个有效有向三元组中的64个；它不能代表论文所述mini-batch hard-mining集合。
- V6唯一修改：对全部有效有向三元组使用平方L2距离，选择`dAN<dAP`者并平均其triplet loss；无严格hard triplet时跳过optimizer step。batch=64下最多25,088项，使用pairwise距离与boolean mask，不改变样本、标签或梯度权重的定义。
- 不变项：Config2源训练、Config1—4/设备1—10、全记录seeded 75/25、模型拓扑、LeakyReLU/BN、margin=.1、SGD momentum=.9、五个fresh 1epoch LR probe、选中LR fresh100epoch、N=11,718、M=10、图13b/14矩阵与seed。V1—V5均保留且不复用输出根。

## TDD与受控诊断

- RED：新增4样本批测试，枚举得到6个strict-hard有向三元组及平方损失均值9.6；实现前以`ImportError`失败。
- GREEN：在`triplet.py`增加pairwise平方距离和3D布尔mask的全量miner；official runner和通用训练入口均改用它。30项相关Tweak/config测试、三个模块编译通过。
- 本地真实Config2受控诊断（不写正式输出）：以V5相同seed、模型、SGD、LR=.001、物理75/25和`N/M`，只训练3,000个source batch后，10设备×102个M=10 held-out决策取得568/1,020=`55.686%`，loss从0.5870到0.1019。它不能替代完整100epoch/论文矩阵，但首次提供了不再接近随机的、方向正确的V6依据。

## 已预登记的N607落地

- 数据根80个输入文件已读回；新的release=`releases/tweak_config2_portability_20260910_v6/source`、run=`runs/tweak_config2_portability_20260910_v6/official_config2_full`、log=`logs/tweak_config2_portability_20260910_v6/train.log`均不存在。
- 启动前全部GPU已有其他用户计算任务；按用户已明确允许每卡加挂实验，选择显存占用最低的物理GPU4（约5.6GiB/24GiB），不触碰其他PID。完整命令使用`CUDA_VISIBLE_DEVICES=4`与runner内`--device cuda:0`、默认1+100和100epoch，不传smoke限制。

## 已验证发布与启动

- 源归档由提交`1fe76128a7f4ec33c0277bf6281c7074ad7cdc32`导出，本地/远端SHA-256均为`73c00d7968a796339395cd3d31440780c96e9e932e43c9ffc8b9523e763c0f14`；解包及三个改动模块编译通过。
- N607真实Config2 CUDA前反传为`[64,2,128]→[64,12]`，全量hard loss=`0.4420853555`有限、16组梯度存在，启动前输出根为空。
- 2026-09-10 04:26 CST在物理GPU4启动PID=`696325`。15秒独立probe核验PPID=1、CWD/cmdline/run-root均为V6、GPU进程占448MiB；空log符合每epoch打印且无异常/最终产物。仅证明初始健康。

## 完成读回与同row结果

- PID=`696325`已自然退出。独立读回run目录只含`best_checkpoint.pt`（8,891,428 bytes）和`results.json`（40,970 bytes）；完整日志为106行：5个source-only LR probe、100个正式epoch和1个`complete`事件。所有正式epoch均为18,310/18,310 active batch；无`Traceback`、`error`、`exception`、`NaN`、`Inf`、`OOM`或`Killed`标记。
- 选中LR=.01、best epoch=95、best mean strict-hard loss=.1000000052。图13b对应Config2源模型的对角准确率：Config1=33.177%、Config2=14.293%、Config3=20.312%、Config4=23.280%；原论文图的相应柱约为75%、90%、74%、68%。图14四配置联合校准：27.875%、12.463%、16.301%、13.377%；原图相应柱约为56%、82%、56%、53%。V6未完成论文数值复现。
- 读回checkpoint的独立几何检查（每类1,024个官方Config2校准帧，eval模式）：类内平均半径=`6.088e-5`，类中心平均距离=`2.893e-5`，最小中心距离=`6.311e-6`，平均embedding范数=`.125959`且标准差=`1.877e-5`。中心距离仍小于类内半径，证实表示塌缩；train模式同样是这一量级，排除仅由BatchNorm推理状态造成的表观塌缩。

## 结论与下一步边界

- V6把mini-batch内全部`d(A,N)<d(A,P)`的有向三元组一起平均。它是可复现的解释，但论文只要求“从mini-batch中选取”这类triplets，并未规定全量平均、每anchor一个、batch-hard或其他采样器。全量平均在每batch引入大量互相矛盾的关系，训练损失迅速稳定在margin=.1并伴随表示塌缩，不能据此宣称是唯一的原作者算法。
- 同条件局部反证：仅改为每anchor最远正/最近负时，LR=.01在第1,000batch出现NaN，LR=.001训练3,000batch为44.412%（453/1,020），也不支持直接发射V7。
- 作者未公开代码及10个发射机选择、物理75/25切分、batch构成/采样器、离散LR选择和best-epoch判据。V1—V6均保留；经过多轮独立单变量修复，下一轮属于架构/实验设计选择，需用户确认后才会新建V7，不会静默重启或覆盖任何根。

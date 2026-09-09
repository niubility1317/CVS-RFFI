# Tweak配置可移植性复现实验V3：数值失配定位与候选修复预登记

- run_id：`tweak_config2_portability_20260909_v3`
- 当前状态：`TERMINATED_METHOD_MISMATCH_PRESERVED`
- 唯一launch owner：Codex主Agent
- 前序运行：V1是已保留的NumPy/PyTorch数组接口技术失败；V2完整闭合但陷入embedding塌缩，Config2同域闭集准确率仅11.677%，不能视为论文数值复现。
- 数据和矩阵边界：继续只使用已经核验的Config1—4、设备ID1—10、共80个官方`cf32`文件；仅复现Config2训练、图13b的4×4单域校准矩阵和图14的四配置联合校准。V3不复用V1/V2输出根，也不运行vanilla、消融或硬件可移植性实验。

## 设计—实现可追溯性

| ID | 来源章节 | 要求 | 目标文件 | 状态 | 验证 | 备注 |
|---|---|---|---|---|---|---|
| T1 | 论文Fig.8、VI-A | 2×128原始IQ、4层1D卷积、两次pool/BN、256/12FC和LeakyReLU的拓扑 | `model.py` | verified | 论文页图与模型层次逐项视觉比对；本地`[64,2,128]→[64,12]`前反传已通过 | 图未给出padding、bias、初始化；这些不据此改动。 |
| T2 | 论文Eq.(1)、VI-A | 使用triplet loss、margin=0.1、SGD momentum=.9、batch=64、100epoch | `triplet.py`、`official_lora_experiment.py` | verified | V2完整500epoch日志和配置已读回 | 学习率具体离散值及checkpoint判据仍是论文未公开默认。 |
| T3 | 论文IV-A、Eq.(1) | hard mining只选择`d(A,N)<d(A,P)`的高损失triplet；margin=0.1仅用于被选triplet的loss计算 | `triplet.py`、`official_lora_experiment.py`、`tests/test_gaskin_tweak_2023.py` | rejected | 论文原文复核明确写为“selecting triplets that have A closer to N than to P”；V3实现使用`d(A,P)-d(A,N)+0.1>0`，会纳入不满足该条件的候选 | 此处不能由PVSNet的adaptive-margin细节覆盖Tweak自己的明示定义；V3因此不具备论文方法数值复现资格。 |
| T4 | 论文VI-C | 每个单次传输保留75%训练、25%测试，N=训练集10%，M=10 | `official_lora.py`、`official_lora_experiment.py`、`tests/test_gaskin_tweak_official_lora.py` | verified | 新增固定seed物理索引、全覆盖、无交集测试先红后绿；实际Config2文件读入的train/test物理索引无交集 | 论文未说明75/25的物理帧选择顺序。V2连续前75/后25导致显著时段失配；候选采用run seed派生的全记录双射，严格保留比例和N/M，标为`PAPER_UNDERSPECIFIED_SPLIT_CHOICE`而非宣称作者原始实现。 |
| T5 | 论文VI-C、Fig.13b、Fig.14 | 同一Config2模型形成4×4单域校准和四配置联合校准；输出不覆盖 | `official_lora_experiment.py`、V3报告 | blocked | V3在第14/500个epoch后由其唯一owner停止；无checkpoint、无`results.json`、无评估产物 | 停止原因是T3确定的论文方法失配，而非低性能；V1/V2/V3目录均保留且绝不复用。 |
| T6 | AGENTS/N607技术失败规则 | 保存V1/V2产物，不覆盖；任何V3使用新release和唯一output root | 本报告、N607发布路径 | verified | V1/V2产物保留；V3使用独立release/run/log/output目录，源归档SHA-256已读回 | 低性能不作为技术停止条件。 |
| T7 | 论文VI-A与运行效率 | 在不改变每anchor均匀同类正/异类负候选分布的前提下，批量实现采样；避免将64个anchor逐个触发GPU随机/同步 | `triplet.py`、单元测试、基准记录 | pending | N607同一环境200个batch采样：当前逐anchor实现10.8548秒；原型向量化0.0511秒，212.3×加速，候选类别约束均通过 | 只允许优化采样实现，不以性能名义改为不同hardness定义或新增未公开训练策略。 |

## 已完成定位证据

- V2的5个学习率、500个完整epoch、总18,310批/epoch均无NaN/异常，但最终损失均接近0.1；checkpoint抽样显示类中心间距低于类内半径，属于表征塌缩，不是校准组合导致。
- 在相同模型、数据、优化器和seed的局部受控诊断中，原batch-hard在全记录分散75/25下仍仅12.353%（126/1020）并停在0.1；因此单独改切分不能修复极值挖掘问题。
- 将挖掘改为随机候选、只更新违反0.1margin的triplet后，连续75/25的一完整18,310批epoch仍只有16.667%（170/1020）；说明连续物理时间段也是独立失配。
- 对由seed=20260908确定的全记录双射（stride=104789、offset=104658）使用随机margin违反挖掘，3,000批达到53.529%（546/1020），一完整epoch为48.333%（493/1020）。该结果现在仅是受控诊断，不得再作为论文hard-mining复现证据，因为其候选过滤不满足论文后续明示的`d(A,N)<d(A,P)`。
- 修复后的真实官方`Config2`数据前反传读回：10个设备记录、`[64,2,128]→[64,12]`、loss=0.317624且有限、16组模型参数均取得梯度；seeded physical train/test索引无交集。该检查只读取数据，不写入实验输出。

## V3停止与效率根因

- V3日志已完整读回至其终态：14个完整epoch、每个18,310批，均为学习率0.01；loss范围0.100122—0.106935，无`Traceback`、`Exception`、`Error`或`NaN`。PID223093于第14个epoch后收到其唯一owner的`TERM`并独立核验退出；输出根没有文件。该产物保留为`TERMINATED_METHOD_MISMATCH_PRESERVED`，不是完成结果。
- 本次停止不是因loss低或耗时长本身。论文在IV-A直接定义hard mining为选择`A`比`P`更靠近`N`的triplet；V3把“margin内”错误当作“hard”，故训练目标不符合原文。
- 同一N607环境的只读采样基准还确认了耗时根因：当前实现对64个anchor逐一执行正/负候选检索、`torch.randint`与`.item()`，200次batch采样耗时10.8548秒；保持每anchor均匀正/负候选分布的向量化原型为0.0511秒，212.3×快，且正同类、非自身、负异类约束均通过。V3约20分钟/epoch的绝大部分新增开销由此造成。

## N607发布与启动证据

- Git代码提交`76bcb3bbe426f57f1519de7739eb739e214939e7`已push，远端分支OID独立读回一致；由此提交导出的V3源归档SHA-256为`22ecc93225834d2058b0ec47358b74da5ba7481672a1232746f0b20dd010f7f6`，上传至N607后再次读回一致，并在`/home/szu2070436088/2510044040/CV-SincNet/releases/tweak_config2_portability_20260909_v3/source`解包、编译检查通过。
- N607数据根`datasets/tweak_official_lora_configurations_20260908/Diff_Configurations_Setup`独立核验为80个所需`.dat`/`.sigmf-meta`文件；远端CUDA前反传同样得到`[64,2,128]→[64,12]`、有限loss和16组参数梯度。
- 唯一输出为`/home/szu2070436088/2510044040/CV-SincNet/runs/tweak_config2_portability_20260909_v3/official_config2_full`，日志为`/home/szu2070436088/2510044040/CV-SincNet/logs/tweak_config2_portability_20260909_v3/train.log`。以`CUDA_VISIBLE_DEVICES=0`和完整默认100epoch×五学习率启动，PID=`223093`。启动后约15秒的独立probe显示该PID存活、CPU139%、GPU0 88%/3855MiB，尚无错误日志或最终产物；这仅证明初始运行健康，不代表完成。

## V3拟定不变项和停止规则

- 保持原始IQ、12维embedding、margin=0.1、SGD momentum=.9、batch=64、五学习率网格、100epoch、每类N=11,718、M=10、设备ID1—10和图13b/图14矩阵。
- 仅替换T3和T4；任何未公开的采样/最佳epoch选择都会写入`method_metadata`和本报告。
- N607启动前必须有本地失败测试、聚焦测试、实际官方数据前反传、Git提交/push/远端OID读回，以及独立V3 output root。运行中仅确定性技术故障停止；低性能保留为科学结果。

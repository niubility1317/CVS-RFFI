# Tweak配置可移植性复现实验V3：数值失配定位与候选修复预登记

- run_id：`tweak_config2_portability_20260909_v3`
- 当前状态：`LOCAL_REPAIR_VERIFIED_RELEASE_PENDING`
- 唯一launch owner：Codex主Agent
- 前序运行：V1是已保留的NumPy/PyTorch数组接口技术失败；V2完整闭合但陷入embedding塌缩，Config2同域闭集准确率仅11.677%，不能视为论文数值复现。
- 数据和矩阵边界：继续只使用已经核验的Config1—4、设备ID1—10、共80个官方`cf32`文件；仅复现Config2训练、图13b的4×4单域校准矩阵和图14的四配置联合校准。V3不复用V1/V2输出根，也不运行vanilla、消融或硬件可移植性实验。

## 设计—实现可追溯性

| ID | 来源章节 | 要求 | 目标文件 | 状态 | 验证 | 备注 |
|---|---|---|---|---|---|---|
| T1 | 论文Fig.8、VI-A | 2×128原始IQ、4层1D卷积、两次pool/BN、256/12FC和LeakyReLU的拓扑 | `model.py` | verified | 论文页图与模型层次逐项视觉比对；本地`[64,2,128]→[64,12]`前反传已通过 | 图未给出padding、bias、初始化；这些不据此改动。 |
| T2 | 论文Eq.(1)、VI-A | 使用triplet loss、margin=0.1、SGD momentum=.9、batch=64、100epoch | `triplet.py`、`official_lora_experiment.py` | verified | V2完整500epoch日志和配置已读回 | 学习率具体离散值及checkpoint判据仍是论文未公开默认。 |
| T3 | 论文IV-A“hard-negative mining”；引用[35]PVSNet | 在线从随机负候选中保留违反margin的triplet；不得以farthest-positive/nearest-negative极值替代该引用策略 | `triplet.py`、`official_lora_experiment.py`、`tests/test_gaskin_tweak_2023.py` | verified | 新增的margin过滤和同类正/异类负随机候选测试先红后绿；26项Tweak聚焦测试通过 | PVSNet的adaptive margin不移植：Tweak本身明确固定margin=0.1。随机候选数量未公开，固定为每anchor一个候选并在结果中披露。 |
| T4 | 论文VI-C | 每个单次传输保留75%训练、25%测试，N=训练集10%，M=10 | `official_lora.py`、`official_lora_experiment.py`、`tests/test_gaskin_tweak_official_lora.py` | verified | 新增固定seed物理索引、全覆盖、无交集测试先红后绿；实际Config2文件读入的train/test物理索引无交集 | 论文未说明75/25的物理帧选择顺序。V2连续前75/后25导致显著时段失配；候选采用run seed派生的全记录双射，严格保留比例和N/M，标为`PAPER_UNDERSPECIFIED_SPLIT_CHOICE`而非宣称作者原始实现。 |
| T5 | 论文VI-C、Fig.13b、Fig.14 | 同一Config2模型形成4×4单域校准和四配置联合校准；输出不覆盖 | `official_lora_experiment.py`、V3报告 | local_verified_release_pending | 26项聚焦测试和`Config2`真实数据`[64,2,128]→[64,12]`前反传均通过；N607新release/run/output尚未创建 | 仅在Git发布读回后启动。 |
| T6 | AGENTS/N607技术失败规则 | 保存V1/V2产物，不覆盖；任何V3使用新release和唯一output root | 本报告、N607发布路径 | verified | V1/V2报告及远端已完成产物保留 | 低性能不作为技术停止条件。 |

## 已完成定位证据

- V2的5个学习率、500个完整epoch、总18,310批/epoch均无NaN/异常，但最终损失均接近0.1；checkpoint抽样显示类中心间距低于类内半径，属于表征塌缩，不是校准组合导致。
- 在相同模型、数据、优化器和seed的局部受控诊断中，原batch-hard在全记录分散75/25下仍仅12.353%（126/1020）并停在0.1；因此单独改切分不能修复极值挖掘问题。
- 将挖掘改为随机候选、只更新违反0.1margin的triplet后，连续75/25的一完整18,310批epoch仍只有16.667%（170/1020）；说明连续物理时间段也是独立失配。
- 对由seed=20260908确定的全记录双射（stride=104789、offset=104658）使用相同随机违反margin挖掘，3,000批达到53.529%（546/1020），一完整epoch为48.333%（493/1020）。这证明两项修复均有因果贡献，但尚未达到论文同域条形图约68—90%的量级，不能提前声称数值复现。
- 修复后的真实官方`Config2`数据前反传读回：10个设备记录、`[64,2,128]→[64,12]`、loss=0.317624且有限、16组模型参数均取得梯度；seeded physical train/test索引无交集。该检查只读取数据，不写入实验输出。

## V3拟定不变项和停止规则

- 保持原始IQ、12维embedding、margin=0.1、SGD momentum=.9、batch=64、五学习率网格、100epoch、每类N=11,718、M=10、设备ID1—10和图13b/图14矩阵。
- 仅替换T3和T4；任何未公开的采样/最佳epoch选择都会写入`method_metadata`和本报告。
- N607启动前必须有本地失败测试、聚焦测试、实际官方数据前反传、Git提交/push/远端OID读回，以及独立V3 output root。运行中仅确定性技术故障停止；低性能保留为科学结果。

# D92独立数据库存元数据补充

日期：2026-09-29。范围：本地既有库存、配置与builder源码，只读核查。未打开IQ、特征、checkpoint、query真值、预测或评分；未运行builder、数据重验证、训练或远端操作。本补充不改变正在运行的LocalRidge矩阵，也不新增实验门槛。

## 结论

**找到原7个目标RX的既有逐TX/day联合库存，但现有本地元数据仍不能给出其精确未用余量。**因此，[原可用性报告](D92_INDEPENDENT_DATA_AVAILABILITY_20260929.md)中“没有现成逐类记录”应限定为当前ManyTx数据版本的专属库存；未用余量为UNKNOWN的结论仍成立。现在不能据此宣布必须重新采集全部数据，也不能宣布已经存在可立即使用的独立验收集。

## 新发现的证据

本地[coverage.csv](E:/type10-7/local_artifacts/P2_CANONICAL_UNION_AUDIT_V1_20260828/inventory/coverage.csv)有`tx_id,rx_id,day_id,record_count,asset_count`五列。[summary.json](E:/type10-7/local_artifacts/P2_CANONICAL_UNION_AUDIT_V1_20260828/inventory/summary.json)记录`equalized="1"`、1,268,812条源记录、1,139,612条canonical记录、129,200条合并重复和0个冲突。这是2026-08-28库存的既有报告值，本次没有重新验证其原始输入。

对冻结26个TX与原7个RX（`1-1,14-7,2-1,20-1,7-14,7-7,8-8`）筛选后，共728个TX/RX/day行，覆盖4个日期。逐RX、逐TX跨日期求和后，7个RX的计数完全相同：

| TX集合 | TX数 | 每个RX、每个TX的联合库存数 |
|---|---:|---:|
| 旧类：14-10、14-7、20-15、20-19、6-15、8-20 | 6 | 4,000 |
| 新类：11-4、3-13、20-12、2-19、13-3、20-7 | 6 | 950 |
| 新类：17-11 | 1 | 800 |
| 新类：12-20、5-20、14-14、2-17、4-1、19-13、19-8、8-1、14-20、1-18、7-7、2-3、18-12 | 13 | 200 |

库存生成源码的[`_read_coverage_rows`](E:/type10-7/local_artifacts/P2_CANONICAL_UNION_SMOKE_V1_20260828_R2/archive_reopen/code/scripts/audit_wisig_canonical_union.py:58)按TX/RX/day计数`DISTINCT canonical.physical_sample_id`；`asset_count`只是来源文件种类数，CSV不保留具体文件成员或逐记录对应关系。本地该inventory目录只有coverage、summary与conflicts三个文件，没有canonical数据库。不能把联合库存数当作ManyTx专属记录数，也不能把多个asset直接相加当作不同物理样本。

**`equalized=1`不是当前输入不兼容的证据。**原7RX的[builder](E:/type10-7/code/snapshots/cvs_practical_baselines_20260927_wt/tools/build_practical_phase2_data.py:61)和[confirmation builder](../tools/build_d92_confirmation_data.py:71)均显式选取WiSig `equalized_list.index(1)`，输入状态为`WiSig_equalized1_center256_unit_rms`。`residual-noeq`表示后续LEO处理不再启用额外均衡。变换均衡状态、裁剪、归一化或信道seed都不能制造新物理记录。

## 仍缺少什么

1. **库存与当前数据版本的绑定。**联合库存对应的asset名称、实际文件/版本及每条记录的asset成员关系；确认它是否覆盖当前ManyTx所用同一记录集合。现有summary没有这些字段。
2. **逐记录身份映射。**至少包含TX、RX、采集日期/session、equalized状态、signal/packet编号及原asset记录索引。旧canonical源码使用五字段坐标的完整SHA256；当前builder使用`['WiSig/ManyTx', TX, RX, day, 1, signal_index]`的截短摘要。两种字符串不能直接做集合差；应使用已有坐标/来源映射，不凭名字认定不同样本。
3. **已用记录排除。**当前三个capsule的56,160个ID及其坐标映射、其它历史support/query/选型使用集合，以及与ManySig源数据和final-evaluation等已用来源的同包关系。当前[D92元数据证据](D92_INDEPENDENT_DATA_METADATA_20260929.json)证明三个capsule内部和相互不重复，但未提供全历史暴露排除或跨命名空间映射。

这些是计算未用余量所缺的具体元数据，不要求重新打开IQ、重复验证已有capsule、增加签名链或重新审查方法。没有这些信息时，缺失项保留UNKNOWN。

## 对采集量的影响

已知完整矩阵需要每个RX/TX有180条未用物理记录（沿用30条support pool）或198条（严格沿用36条support pool），分别为4,680条或5,148条/RX。13个新TX在旧联合库存中各只有200条，是明确的潜在容量瓶颈。

若后续元数据证明该200条集合包含原capsule已用的198条且版本不变，则这些TX至多剩2条；仅满足库存量就分别至少还缺178条或196条/TX/RX，历史额外暴露可能增加缺口。这是有明确前提的容量推导，**不是已核实余量或采购总量**。旧类及其余7个新类即使总库存更大，其未用身份也仍须从同一排除映射得到。

当前不能准确计算必须新增采集多少条。下一步可先取得上述已有元数据或其位置，再计算逐RX/TX的可用集合及缺额。同RX的新物理记录只能支持“已知RX上的未用样本确认”；新RX泛化需要原source及已暴露RX集合之外的接收机。不得通过删TX、降低K、跨RX拼类或替换表示来补足独立数据。

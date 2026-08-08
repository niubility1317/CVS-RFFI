# P1-PAMR冻结设计与实现追溯卡（2026-08-09）

## 冻结边界

P1-PAMR（Paired Angular-Margin Restoration）从每折GeoSat-C C checkpoint续训：C仅保留基线，G仅加`lambda_pamr=0.05`。它不是无教师方法：clean分支提供停止梯度的同批标量边界；但不引入EMA、外部teacher、新head、阈值或显式`z`对齐。proxy、held与LEO评估行不参与训练、校准或选择。

| ID | 冻结需求 | 落点 | 状态 | 验证 |
|---|---|---|---|---|
| PAMR-01 | `z_id=feat_joint`且与`id_backbone.cls_head.head.weight`逐行同类序/同维 | `phase1_pamr.py`绑定检查 | 已实现 | 单元负测与训练路径断言通过 |
| PAMR-02 | clean raw-cosine正确门；clean margin与head权重停止梯度 | `pamr_loss` | 已实现 | 梯度及raw-cosine测试通过 |
| PAMR-03 | 仅LEO分支从等TX hinge回传；`lambda=.05` | `pamr_loss`与训练接入 | 已实现 | 数值/类别置换测试通过 |
| PAMR-04 | 无RX/domain、GRL/MMD/CORAL、EMA/teacher/head/threshold、proxy/held路径 | 参数冻结验证 | 已实现 | 非法组合负测通过 |
| PAMR-05 | 每TX有效anchor/hinge覆盖；仅技术audit记录unscaled特征梯度与共享encoder梯度关系 | receipt/terminal | 已实现 | audit/formal分离终态负测通过 |
| PAMR-06 | 1epoch G-only technical audit跳过全部source-val、LEO、tail、leakage与heldout评估；40epoch C/G final-only | train与两launcher | 已实现 | technical-only静态测试与dry-run通过 |
| PAMR-07 | 六折12行固定C/G、八卡映射 | full launcher | 已实现 | launcher结构测试与12行dry-run通过 |

## 可证伪出口

任何绑定、None、非有限PAMR梯度或shared-gradient异常均以best-effort原子failure receipt记录后立即失败，且receipt写盘异常不得遮蔽原异常。1epoch技术audit在首个有效batch记录raw梯度关系；有限零梯度仅记录，但audit终态必须至少一次非零梯度。40epoch正式路径不执行额外`autograd.grad`，终态只要求每TX有效clean-correct anchor与active hinge覆盖。共享encoder梯度余弦与范数比只作audit健康收据，不按性能或符号选择。postfreeze只比较同物理ID的C/G clean与三LEO场景分类地板及角度margin；paired cosine距离仅诊断，不能成为补偿或调参入口。

## 资源矩阵

预飞：F1G…F6G各1epoch，GPU0…5各一行，技术健康且`NO_PERFORMANCE_RESULT`。全量：F1C+F5G/GPU0，F1G+F5C/GPU1，F2C+F6G/GPU2，F2G+F6C/GPU3，F3C/GPU4，F3G/GPU5，F4C/GPU6，F4G/GPU7；均40epoch、同checkpoint/seed/sampler、final-only。

本地验证：`python -m py_compile`通过；`pytest -q code/tests/test_phase1_ccpc_leo.py code/tests/test_phase1_pamr.py`为41通过；两份launcher均经`bash -n`，dry-run分别为6和12行。以上仅证明本地机制与入口闭合，不构成性能或晋级结论。

## 评审修复追踪（2026-08-09）

|ID|来源|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|PAMR-R1|独立复核P1|1epoch技术审计必须跳过全部source-val、LEO、tail与leakage性能评估，并固定声明`SKIPPED_TECHNICAL_AUDIT`/`NO_PERFORMANCE_RESULT`|`train_ssdg.py`、测试|verified|py_compile、41 focused tests、dry-run6/12|不改变训练数据、loss或矩阵|
|PAMR-R2|独立复核P1|None、nonfinite与shared-gradient异常以best-effort原子failure receipt持久化，写盘失败不得遮蔽原异常|`phase1_pamr.py`、`train_ssdg.py`、测试|verified|原子写入与writer失败不遮蔽原异常测试|不含raw样本或原始异常文本|
|PAMR-R3|独立复核P1|raw feature/shared-encoder梯度关系只在audit的首个有效batch审计；40epoch正式路径只计coverage并正常反传|`phase1_pamr.py`、`train_ssdg.py`、测试|verified|audit/formal终态分离与训练路径静态测试|正式terminal不要求梯度关系receipt|
|PAMR-R4|audit v1系统性技术失败|`data_ctx`必须显式携带局部训练类数；局部TX类序、checkpoint训练TX类序与live classifier head行数必须逐项绑定，禁止把全局数据集6类误当作局部训练4类或反之|`train_ssdg.py`、`phase1_pamr.py`、测试|verified|py_compile、CCPC+PAMR focused43项、bash-n、dry-run6/12|不改变loss、`lambda=.05`、C/G矩阵或audit边界；v1无性能结果|

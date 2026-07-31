# CVS-RFFI Phase2详细复现报告qKNN同场景结果补充说明

## 正文基准

- 唯一正文基准：用户提供的`CVS-RFFI_Phase2详细复现报告1.docx`。
- 原文档保持不变；本次输出为独立补充版本。
- 本次不重写已有定义、方法公式、7.16周报表和7.24周报表，只增加qKNN结果及必要的证据边界说明。

## qKNN证据来源

- 正式结果只采用`d92_registration_balanced_125_20260720`的retry2。
- retry2完成125/125个任务、375/375个LEO场景，失败数为0。
- 正式状态为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`；retry1和Role-Oracle结果不进入正文。
- 结构化结果核对面为`row_metrics.csv`、`scenario_metrics.csv`、`receiver_metrics.csv`、`per_tx_metrics.csv`、`summary.json`和`gates.json`。

## 插入位置

### Stage2-B

在原`3.6.3不同target receiver下的结果`之后增加：

- `3.6.4qKNN Stage2-B域适应实验结果`；
- 按K=1、5、10以及5个target receiver展开clear、low-elev、rain三种LEO弱信道场景结果，每个单元格为5个seed的均值；
- 明确原始`B-old`字段表示完成旧类support适配后的`S2B-old`，不是Phase1直接推理结果；
- `3.6.5qKNN与域适应方法的共同LEO场景对照`，按K=1、5、10并列直接ADV3B02、MRIOR-SDA、DADDA-SDA、ProtoNet CDA与qKNN；
- 原`3.6.4结果边界`顺延为`3.6.6结果边界`。

域适应矩阵与qKNN使用相同三类LEO场景，但seed集合、数据矩阵和artifact哈希并非严格配对，因此跨方法部分只作描述性比较。qKNN自身的接收机与场景分层表直接来自正式retry2的`scenario_metrics.csv`。

### Stage2-C

在原`4.5.2主要现象`之后增加：

- `4.5.3qKNN与类增量方法的共同LEO切片对照`；
- 在K1/new20、K5/new20、K10/new5、K10/new10和K10/new20五个共同切片中，逐行列出qKNN、CSIL官方流程和MoPC-HR官方流程；
- 每行同时保留适应前旧类、适应后旧类、新类、旧新调和均值和遗忘指标。

CSIL和MoPC-HR使用seed 713101–713105，qKNN使用713102–713106；三者base训练、状态构造和数据权限不同，因此不能把结果解释为严格paired算法排名。

## 结论边界

- qKNN在K10/new20得到注册后旧类71.333%、新类68.150%、旧新调和均值69.555%，遗忘14.778个百分点。
- qKNN相对matched control在K10/new20改善旧类和遗忘，但新类下降，全部绝对性能门仍失败。
- 报告只能声明qKNN为大规模注册下旧类遗忘的诊断性正信号，不能声明方法晋级或普遍优于论文对比方法。

## 文档验证

- 最终文档共16页、17张表。
- 原14张表均保持，新增qKNN Stage2-B分层结果、Stage2-B跨方法对照和Stage2-C共同切片对照3张表。
- 新增表均设置固定列宽、重复表头和禁止单行跨页拆分。
- qKNN类增量表的旧类、新类、调和均值和遗忘表头均保留为Word公式对象。
- 中文可见文本使用宋体，英文、数字和变量使用Times New Roman。
- 使用Microsoft Word排版引擎导出PDF并逐页检查，未发现裁切、重叠、缺字、孤立表头或表格越界。

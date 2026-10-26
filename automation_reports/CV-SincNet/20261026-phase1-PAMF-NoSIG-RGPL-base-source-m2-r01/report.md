# NoSIG 完整监督与普通 PL：匹配源对照

**ANALYZED：两个固定E50/11100对照及全部源V八视图完成。** 仅学生seed2026092701、源V27000；没有本轮目标测试，不是三学生seed结果，原联合性能目标未达成。

执行代码`f5eb718af9bf461b6b0b9fa6b422f8794bc19103`。PL复用合法scratch NoSIG E20＋10＋20，监督从零20＋10＋20；完整L机制、As KL0、原6×160默认头联合训练保留，整个SIG_TIME路径描述支路无参数。PL累计13L＋37LU，监督U loader完全关闭。G为同契约原固定G-LU，评估期间F/G所有参数和注册buffer保持。

## 完整源结果

下表单位%，H/G三景为low suburban、mid urban、low urban；源八视图包括raw、G(raw)及三组H/G。它不是目标六practical均值。

| 方法 | raw | G(raw) | H三景均值 | G三景均值 |
| --- | ---: | ---: | ---: | ---: |
| NoSIG-Sup | 97.759259 | 97.707407 | 73.670370 | 89.637037 |
| NoSIG-PL | 97.562963 | 97.537037 | 73.733333 | 89.628395 |

PL−Sup：raw -0.196296pp、G三景 -0.008642pp。源内仍无明显联合正收益，不能据源指标推断目标差。RX1分组仅用于后续附加机制留出比较，两个控制和其父均已见过RX1，不称真正未见接收机验证。

NoSIG-PL的RX1 raw99.685185%、G三景92.567901%；其余RX raw97.032407%、G三景88.893519%。未来学生判断继续相对本PL控制固定为RX1 G三景≥+.3pp、overall raw≥−.1pp，源条件只作证伪。原目标F0所有目标RX日期168000/seed、三个学生seed及raw／同G六practical标准均不降低。

## 全部执行核验

完整解析新增17760条训练记录、继承PL前缀4440条记录及3215行新stdout，共100个完整epoch。全部逐步核对L768、As KL0、原跨RX权重日程、实际U曝光／接受边界与零非有限跳步；六个实际segment checkpoint全部无sequence参数、默认头形状6×160、来源／epoch／更新／原生U与EMA关闭／无外部teacher均匹配。各自默认头在两个续训段均真实改变。

PL三段U曝光396900／567000／1134000，接受348183／510984／1057793；累计U曝光2097900、接受1916960，都是重复训练曝光，不能称唯一样本数或U正确率。监督三个阶段U曝光与接受均0。

独立从全部8×27000×6源logits重算每TX/RX/day及overall/留出分组，共272个单元，与原报告一致；全覆盖、finite、原物理顺序核实。controller和所有worker均已消失，完整结束。核算新增IQ读取／模型前向／U真值／target均0。

[源端点](source_endpoints.csv)、[全部分层](source_cells.csv)、[100轮完整曲线](source_epoch_curves.csv)、[完成证据](complete_evidence.json)、[预登记](preregistration.md)。本行没有训练图候选；RGPL源质量已独立失败，后续标签专家见[新路线](../../../docs/goals/PAMF_RESTORATION_LABELER_PLAN_20261026.md)。

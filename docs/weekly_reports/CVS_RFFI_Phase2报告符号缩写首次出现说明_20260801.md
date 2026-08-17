# Phase2报告符号与缩写首次出现说明

## 修订对象

- 源文档：`CVS-RFFI_Phase2详细复现报告1_qKNN域适应与类增量结果补充版_截至20260731.docx`
- 输出文档：`CVS-RFFI_Phase2详细复现报告1_qKNN符号缩写首次出现说明版_截至20260801.docx`

## 修订内容

- 在相关概念、公式或方法第一次出现的位置加入25处“符号说明”“缩写说明”或“术语说明”。
- 对集合、上下标、索引、概率、损失、权重、评价指标和训练变量补充数学含义及CVS-RFFI中的物理含义。
- 补充RFFI、FSL、DA、CIL、FSCIL、UDA、IQ、LEO、TX、ProtoNet、CDA、MRIOR、SDA、CE、DV-KL、DADDA、MMD、LMMD、RBF、SGD、CSIL、EWC、KD、MoPC-HR、HR、IoT、qKNN、KNN、IEEE和DOI等缩写的英文全称与中文含义。
- 明确ADV3B02是项目实验版本标识，CVS是项目/流程名称，均不虚构为通用算法缩写。
- 保持原有17张实验表、实验数据、结果边界和5篇对比方法参考文献不变；报告中继续统一使用qKNN，不出现D92旧称。

## 质量检查

- 结构检查：17张表、25处首次出现说明、308个Word数学对象；页眉和页脚为空。
- 字体检查：可见文本中文字体为宋体，英文和数字字体为Times New Roman，未发现违规run。
- 表格检查：17张表均保留重复标题行，153个表格行均禁止跨页拆分。
- 文本检查：未发现可见LaTeX反斜杠、D92旧称或公式分隔符残留。
- 版式检查：使用Microsoft Word导出PDF，共18页；逐页检查标题、公式、表格、分页和参考文献，未发现裁切、遮挡或溢出。

## 2026-08-17括号定义修订

- 新输出文档：`CVS-RFFI_Phase2详细复现报告1_qKNN首次出现括号定义版_截至20260817.docx`。
- 将原有25处独立“符号说明”“缩写说明”和“术语说明”全部移入相关术语或符号首次正文出现处的中文全角括号内；标题保持简洁，不在标题中堆叠长定义。
- 缩写在首次正文使用处说明英文全称、中文含义和CVS-RFFI中的具体作用；数学符号继续使用Word原生公式，并说明上下标、运算意义及物理对应关系。
- 独立说明标签数量由25降为0；文档段落数量与2026-07-31源文档一致，未新增集中符号表或文末术语表。
- 结构校验通过：25组括号定义、17张表、153个表格行、309个Word数学对象；页眉页脚为空，未发现可见LaTeX反斜杠、D92旧称或字体违规。
- 实验数据、17张表、损失函数、训练权限、结果边界和5篇对比方法参考文献保持不变。

## 2026-08-17符号统一修订

- 新输出文档：`CVS-RFFI_Phase2详细复现报告1_qKNN符号统一版_截至20260817.docx`。
- 发射机类别集合统一为`\mathcal Y`；类别数量保留普通大写字母，类别索引统一为`c`。
- source/target域下标统一为`\mathrm{src}/\mathrm{tgt}`；`P_{\mathrm{src}}`、`P_{\mathrm{tgt}}`等均保留为Word原生公式上下标，不使用普通文本下划线模拟。
- 分类概率统一为`\pi_\theta(c\mid x)`，prototype统一为`\mu_c`；few-shot任务、softmax温度、数值稳定项与随机扰动分别使用互不冲突的符号。
- 增量session保留`t`，优化步改为`u`，MoPC-HR参数组索引/总数统一为`j/J`。
- 域适应与注册状态使用`DA0_REG0`、`DA1_REG0`、`DA0_REG1`和`DA1_REG1`，不再混用`0/1`与`pre/post`表示同一状态。
- qKNN首次出现处明确其名称中的`K`不等于报告中的`K-shot`样本数。
- 自动核验通过：17张表、153个表格行、5条参考文献和309个Word数学对象保持不变；实验百分比、百分点及时延token序列与输入版完全一致。

## 2026-08-17实验结果按配置拆分

- 新输出文档：`CVS-RFFI_Phase2详细复现报告1_qKNN结果按配置拆分版_截至20260817.docx`。
- 将正式LEO弱信道结果、qKNN共同LEO切片对照和matched无LEO新类诊断三张大表，分别按`K-shot`与新类数的唯一组合拆分为18张、5张和15张配置表。
- 每张配置表只保留该配置下实际存在的方法和指标；qKNN共同切片的每张表均同时列出qKNN、CSIL官方流程和MoPC-HR官方流程。
- 正式LEO、共同切片和无LEO诊断仍位于三个独立小节，不跨实验数据、初始状态或训练权限合并结果。
- 自动核验通过：原三张大表的57条数据行均恰好保留一次，52张总表、188个表格行、5条参考文献和491个Word数学对象通过结构审计；百分比、百分点及时延token多重集合与输入版完全一致。

## 2026-08-17正式项目命名与重点突出

- 新输出文档：`CVS-RFFI_Phase2详细复现报告1_正式命名与重点突出版_截至20260817.docx`。
- 正式项目名称统一为“星地跨域少样本持续射频指纹识别（CVS-RFFI）”；Phase1方法命名为“地面跨接收机域泛化射频指纹表征方法”，Phase2方法命名为“星载少样本域适应与新类增量注册方法”。
- 从正文和表格中删除内部实验版本代号`ADV3B02`；后文使用“Phase1域泛化基座”等任务可读名称，不把版本号包装成未经验证的新算法贡献。
- 第1章正文开头的项目名称、两个阶段方法名称和核心任务，以及Stage2-B/Stage2-C核心目标和qKNN正式结论，使用红色加粗突出；其他正文维持原有层级，避免过度标红。
- 自动核验通过：内部版本代号出现次数为0；52张表、188个表格行、5条参考文献、491个Word数学对象及全部实验数字保持不变。

## 2026-08-17中英文名称与D92 E0方法纠正

- 新输出文档：`CVS-RFFI_Phase2详细复现报告1_正式中英文命名与ERTB-IDR版_截至20260817.docx`。
- 项目英文名统一为“Ground-to-Satellite Cross-Domain Few-Shot Continual Radio-Frequency Fingerprint Identification”；Phase1英文名为“Ground-Based Cross-Receiver Domain-Generalized Radio-Frequency Fingerprint Representation Learning”；Phase2英文名为“Spaceborne Few-Shot Domain Adaptation and New-Class Incremental Registration for RFFI”。
- 报告中的项目对比方法不是原始D92，也不再以`qKNN`命名；正式名称为“高效稳健任务均衡增量判别注册方法（Efficient Robust Task-Balanced Incremental Discriminant Registration, ERTB-IDR）”，对应`D92 E0_FULL_ONLY`。
- 首次定义明确：ERTB-IDR保留288维联合特征、ground-spectrum Cauchy稳健中心、旧/新任务均衡收缩协方差和统一LDA仿射头；关闭原D92的Fisher/Pareto安全门和K折full/block双几何融合，注册态只执行一次full主几何闭式拟合。
- 名称修改不得改变52张表、纯数值实验单元格、491个Word数学对象和5条参考文献。

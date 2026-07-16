# Stage2周报规划与数据协议追踪表

目标文件：`周报_三种域适应对比方法_20260716.md`

依据：

- `E:\type10-7\项目.md`，版本2026-07-16；
- `paper_reproduction/configs/cvs_stage2c_extreme_light_tx_sets_20260714.json`；
- `paper_reproduction/configs/cvs_stage2c_effective8_class_split_20260715.json`；
- 用户要求：在三种域适应方法周报中加入详细Stage2实验规划、数据划分和数据协议。

|ID|来源章节|要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|S2W-01|项目.md§7、§9|说明Phase2总体目标和Stage2-A/B/C关系|周报Stage2规划章节|verified|章节7.1文本检查通过|Phase3已与Phase2主线分离|
|S2W-02|项目.md§8.1|说明WiSig/ManySig仅是地面代理数据|周报数据角色章节|verified|章节7.2文本检查通过|已明确不是真实卫星数据或在轨验证|
|S2W-03|项目.md§8.2|明确`R_s`与`R_t`不相交|周报receiver划分章节|verified|集合交集检查为空|已列出7个source receiver和5个target receiver|
|S2W-04|项目.md§8.3|明确`Y_old/Y_new/Y_unknown`互斥语义|周报TX划分章节|verified|集合与边界文本检查通过|Phase3 unknown不能充当seen-new|
|S2W-05|项目.md§8.4|列出ManySig旧类与ManyTx新类来源|周报数据来源章节|verified|章节7.2、7.4文本检查通过|ManyTx为Stage2-C主来源|
|S2W-06|嵌套新类配置|列出5/10/20个真实新类集合|周报TX划分章节|verified|JSON集合一致性和嵌套检查通过|集合大小分别为5、10、20|
|S2W-07|项目.md§9.1|说明Stage2-A的support/query与声明边界|周报Stage2-A章节|verified|章节7.6文本检查通过|已禁止报告seen-new|
|S2W-08|项目.md§9.2|说明Stage2-B的support/query与指标|周报Stage2-B章节|verified|章节7.7文本检查通过|已禁止使用target-new support|
|S2W-09|项目.md§9.3|说明Stage2-C的support/query与指标|周报Stage2-C章节|verified|章节7.8文本检查通过|同row输出old/new/H与遗忘|
|S2W-10|项目.md§7.1、§7.3|写明LEO_weak-only、clean不可达及Stage2禁止源域数据|周报Phase2输入协议章节|verified|clean与source强制协议字段检查通过|仅允许sealed Phase1 ADV3B02 checkpoint继承源域训练|
|S2W-11|项目.md§7.2|写明逐样本全注册类决策和无Oracle/配额|周报预测协议章节|verified|5个决策字段检查通过|已禁止dense query图和全局分配|
|S2W-12|项目.md§7.2|写明prediction与scorer隔离|周报评分协议章节|verified|章节7.9、7.10文本检查通过|prediction artifact先固化并绑定SHA256|
|S2W-13|项目.md§8.5|写明三个LEO场景和自适应1→3→5-view|周报View规划章节|verified|场景和View文本检查通过|已区分物理场景与TTA View|
|S2W-14|项目.md§10.3.1|写明K10开发、K5稳健性、K1压力和K20遗忘|周报K-shot规划章节|verified|章节7.5、7.12、7.13检查通过|K1/K5/K20 confirmation禁止继续调参|
|S2W-15|项目.md§10.3.1|写明旧类、新类、H和遗忘指标门槛|周报指标与门禁章节|verified|门槛数值检查通过|5/10/20新类门槛分别92/90/86%|
|S2W-16|项目.md§10.3.1及用户最新要求|写明统一星上正式资源门槛|周报资源规划章节|verified|章节7.14数值检查通过|统一50k/256KB；adapter≤20epoch，原层更新≤5epoch/50step|
|S2W-17|项目.md§10.3.1|给出分阶段矩阵规模与启动顺序|周报实验矩阵章节|verified|25/75/100/125/300行及预测量算术检查通过|先K10筛选，再全K与Stage2-C|
|S2W-18|项目.md§8.5、§10|写明artifact、分组统计和晋升规则|周报输出与判定章节|verified|章节7.15、7.16字段检查通过|包含逐receiver/逐类/资源证据|

## 反向审计结论

18项要求全部落实并完成静态一致性验证：`verified=18`、`deferred=0`、`rejected=0`、`blocked=0`。

本次交付是实验规划和协议文档的严格对照更新，不代表Stage2-C矩阵已经执行，也不代表性能目标已经达到。最高风险是现有MRIOR-SDA/DADDA-SDA runner仍具备source cache入口，必须先完成无source predictor package、validator和runtime access audit修复，才能启动当前协议下的新Stage2矩阵。

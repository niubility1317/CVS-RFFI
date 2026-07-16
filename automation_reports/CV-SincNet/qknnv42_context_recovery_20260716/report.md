# qKNNV42当前任务对话上下文恢复报告

## 1.恢复结论

当前主任务的原始session仍完整存在于：

`E:\codex\home\sessions\2026\07\14\rollout-2026-07-14T17-16-24-019f5fe9-b4ed-7c00-b935-91eb4657c1fc.jsonl`

用户在2026-07-14至2026-07-15给出的关键引导、2026-07-15 23:45左右生成的“关键层＋K-shot”turn总结，以及`JG_R8_LR020=88.8354%`的结果均可从该session与Git报告交叉恢复。当前问题属于聊天界面的历史可见性/折叠问题，不是底层主session证据丢失。

Codex当前没有可将历史message原位重新插回既有聊天气泡的安全接口。因此本次修复采用三层可见承载：

1. 本Git恢复报告，保存用户引导、缺失turn总结和证据边界；
2. `analysis/qknnv42_context_recovery_traceability_20260716.md`，逐项追踪恢复完整性；
3. 当前任务中的用户可见回复，重新给出88%版本和当前接续点。

这不是从已删除污染会话恢复内容。会话`019f6610-86af-7572-b857-2544e7b598ba`及其strict300、EvidenceNorm、JP-R4影响继续保持删除和隔离状态。

## 2.恢复的用户引导

以下内容来自当前主session中的直接user message，不包含重复的internal goal注入。

|ID|时间|恢复的用户引导|当前落实口径|
|---|---|---|---|
|G01|2026-07-14 21:51|打开目标模式，在提升旧类、新类性能的同时继续压缩计算资源，探索极轻型快速适应|当前最终门槛已在后续提升为K10旧类92%、最低类88%、5/10/20新类92%/90%/86%|
|G02|2026-07-15 08:18|允许用support更新ADV3B02 backbone，但必须轻量；也可用地面训练的小参数快速适应模块|只允许预注册关键层稀疏更新或小adapter，不做full-backbone微调|
|G03|2026-07-15 08:20|先分析适合星上训练部署的参数预算；压缩60epoch adapt和固定5-view|首选≤50k参数、≤20epoch、≤256KB；关键层候选实际为3,840或6,400参数、5epoch|
|G04|2026-07-15 08:59|禁止角色Oracle与类别配额|所有正式候选逐样本面对全部注册类，role/quota/global assignment全部为false|
|G05|2026-07-15 09:00|只更新ADV3B02关键层，原60epoch太重|已定位`joint_proj`和`id_gate＋joint_proj`，早期Sinc/卷积/归一化冻结|
|G06|2026-07-15 09:05|多View是历史高性能关键，必须作为压缩重点|保留多View能力，不采用固定5次forward|
|G07|2026-07-15 10:20|为接近性能目标，压缩预算可放宽50%至100%|只允许performance-relaxed资源档，不放宽Oracle、clean或query拟合禁令|
|G08|2026-07-15 10:44|默认1-view，低置信度时触发额外View，形成自适应多View|目标策略固定为逐样本1→3→5，报告平均/P95 forward及触发率|
|G09|2026-07-15 11:17|重点提升1-shot；qKNN在极少shot下必须带来正收益，也可加入其他轻型DA方法|K1必须相对identity适配不为负，并显著优于strict direct ADV3B02|
|G10|2026-07-15 11:29|把不同K值下的遗忘率加入优化目标|K∈{1,5,10,20}同row报告注册前后旧类、forgetting和gain|
|G11|2026-07-15 11:30|从理论和底层分析ADV3B02哪些层最有效，以及目标与损失函数设计|保留关键层消融、LOPO、边界/anchor/Gram/View损失设计证据|
|G12|2026-07-15 11:31|K=1适应后必须明显优于直接ADV3B02|确认矩阵目标为总体至少+2pp且paired 95% CI下界>0，逐receiver不低于0|
|G13|2026-07-15 11:32|不要遗漏任务引导，必要时可调用子agent|本报告将引导固化为可审计条目；子agent仅在独立问题确有必要时使用|
|G14|2026-07-15 12:49|`项目.md`已更新，先读相关文档|本次恢复前已重新完整读取`AGENTS.md`和695行`项目.md`|
|G15|2026-07-15 19:01|使用ADV3B02作为基底模型|当前主线checkpoint固定ADV3B02|
|G16|2026-07-15 19:37|满足项目要求即可，不把精力重点放在协议握手，重点提升qKNN性能|协议作为硬边界；算法时间优先用于adapt、注册、遗忘和多View|
|G17|2026-07-15 20:07|实验启动后回顾之前对话|与三轮复盘规则合并执行|
|G18|2026-07-15 20:26|先把重点放在adapt|先完成关键层与support-only适配设计，但不能因此省略注册后实测|
|G19|2026-07-15 20:40|适应后及加入新类后都必须显著高于MRIOR且更轻量；先分析高效更新层|MRIOR-SDA采用matched Stage2-B/Stage2-C同row性能和资源对照|
|G20|2026-07-15 23:47|实测没有新类注册后性能，不能忽略|没有`seen_new_acc/H_old_new/注册后old_acc`的路线一律不算完成|
|G21|2026-07-16 00:19|不要偏离目标|当前恢复与后续路线均以active goal和`项目.md`为唯一成功口径|
|G22|2026-07-16 00:19|不仅有域适应，还有新类注册|Stage2-B适配与Stage2-C注册同等重要，且必须分注册前、注册后报告|
|G23|2026-07-16 00:22|每三轮探索回看目标、`项目.md`、历史路线并吸取教训|第四轮前必须写入正式复盘，不得连续盲扫|

当前还必须执行的最高优先级约束为：support与query都已在Phase2边界外叠加`leo_clear_weak`、`leo_low_elev_weak`或`leo_rain_weak`，Phase2不能访问clean样本或clean派生信号；适配进程不能访问query；预测逐样本面对全部注册类；评分在不可变prediction之后独立连接truth。

## 3.恢复的缺失turn总结

原始session在2026-07-15 23:45保存了一段“关键层＋K-shot”turn总结。该总结的核心结论是：K10和K20可以从极小关键层更新获得正收益，但K1不能沿用K10更新规则；K1继续扩大层数或epoch会放大单shot噪声，必须缩小到最终联合投影并加入support-only保护。

恢复时把“已经实测”和“下一步设计”分开，避免原总结中计划项看起来像已验证事实。

### 3.1已经实测

|K|候选|P4 identity acc/floor|适应后acc/floor|相对identity acc/floor|资源|结论|
|---:|---|---:|---:|---:|---|---|
|1|`JG_R8_LR020`|88.0334%/73.5484%|87.9371%/73.3333%|-0.0962/-0.2151pp|6,400参数、5epoch/5step、0.650s|负迁移|
|5|`JG_R8_LR020`|88.2900%/72.6882%|88.5146%/72.2581%|+0.2246/-0.4301pp|6,400参数、5epoch/25step、1.078s|总体升但floor退化|
|10|`JG_R8_LR020`|88.3221%/73.7634%|**88.8354%/75.9140%**|**+0.5133/+2.1505pp**|6,400参数、5epoch/50step、1.293s|当前保留winner|
|20|`JG_R8_LR020`|88.4184%/73.7634%|88.8996%/73.9785%|+0.4812/+0.2151pp|6,400参数、5epoch/50step、1.330s|正收益|

K1随后进一步收缩为只更新`joint_proj`的`JP8_LR005`：3,840参数、5epoch/5step、0.626s、patch 9,306B、持久状态51,374B，结果为88.0013%/73.5484%。它相对strict direct ADV3B02仍有+0.6096pp accuracy和+2.5257pp floor，但相对P4 identity为-0.0321pp/0.0000pp。因此只能说“整体系统高于direct”，不能说“K1 target梯度适配成功”。

### 3.2尚未验证的下一步设计

- K1：冻结P4安全基线，只允许3,840参数`joint_proj`候选；通过support-only最差`View×类别`信赖域决定缩放或撤销delta。
- K2至K5：拟采用`joint_proj`、rank8、最多25步并增加最低类margin保护；该K依赖策略尚未完成正式target验证。
- K10及以上：采用`id_gate＋joint_proj`、rank8、6,400参数、最多50步；目前只有source receiver证据。
- K1改进方向：注册support的接收侧shift/CFO增强与逐样本自适应1→3→5 View；不能用query、角色、quota或批统计选delta。

## 4.88%版本的完整身份

|字段|值|
|---|---|
|候选|`P4＋BPJG-LOPO JG_R8_LR020`|
|基础模型|ADV3B02 checkpoint，SHA256 `2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|ground P4|`projection_feature rank16/e8`，SHA256 `95f9a8bac7880d42f705db7f16523c37cf4ce5ff8438ac2c500c7550a38de446`|
|更新层|`id_gate.0＋joint_proj.0`|
|LoRA|rank8，6,400可训练参数，lr=0.02|
|训练|5epoch、50 optimizer step、SGD无momentum|
|输入|source validation receiver `2-19`、6个source类、K10、3个已登记`LEO_weak`support View|
|适应后准确率|**88.8354%**|
|最低类准确率|**75.9140%**|
|相对P4 identity|+0.5133pp accuracy、+2.1505pp floor|
|相对strict direct ADV3B02|+1.4437pp accuracy、+4.8913pp floor|
|时延/显存/状态|1.293s、154.08MiB、57,084B|
|support forward等价|1,980|
|query训练访问|0行，false|
|role/quota访问|false/false|
|result SHA256|`fa4265d3c2a61d24aeca0efbe594a92339627fce8c3f284dd69aa2d9133b42e4`|
|adapter SHA256|`15d00c3ffac69fcacdccf826301d2419c7eee43959e8ca1f59edabc7736dad4d`|

三场景结果均为正向：

|场景|strict direct acc/floor|P4 identity acc/floor|适应后acc/floor|相对identity|
|---|---:|---:|---:|---:|
|`leo_clear_weak`|90.6641%/77.4194%|91.4341%/76.7742%|92.0115%/78.7097%|+0.5775/+1.9355pp|
|`leo_low_elev_weak`|86.5255%/69.3182%|87.6805%/72.9032%|88.1617%/75.4839%|+0.4812/+2.5806pp|
|`leo_rain_weak`|84.9856%/65.9091%|85.8518%/71.6129%|86.3330%/73.5484%|+0.4812/+1.9355pp|

### 声明边界

该88.8354%只证明source receiver、K10、6个旧类条件下，极小关键层BPJG-LOPO更新具有正收益和计算压缩潜力。它不是：

- target receiver Stage2-B确认结果；
- 5/10/20个target-new注册后的Stage2-C结果；
- 5receiver×5seed×3场景确认矩阵；
- 已显著超过matched MRIOR-SDA的证据；
- 已达到旧类floor≥88%的证据。

当前保留谱系没有合法的新类注册后`old_acc`、`seen_new_acc`或`H_old_new`实测。因此“适应旧类和新类注册的当前性能”必须写成：旧类source适配88.8354%/floor75.9140%；新类注册后性能尚未取得，不能填入推测值。

## 5.污染隔离与当前最强版本修正

已删除会话曾使当前主任务错误引用EvidenceNorm/strict300/JP-R4结果。清理后这些代码、报告和N607运行产物已撤销，不能从仍保留的聊天历史文字中重新引用。

|内容|当前状态|是否可用于最强版本|
|---|---|---|
|`JG_R8_LR020` v21/v22/v23 source LOPO/K-shot证据|保留，独立主线|可以，但必须标source-only|
|strict300运行时与矩阵|已删除|不可以|
|EvidenceNorm Round1|已删除|不可以|
|JP-R4 Round2|已删除|不可以|
|历史92.28% legacy diagnostic|历史不同切分、20新类、单seed且含60epoch/5-view/FFT/Oracle|只可作历史上界说明，不可作当前正式结果|

所以，当前可验证的“最强保留适配版本”是`JG_R8_LR020`；当前不存在可验证的“最强正式target Stage2-C版本”。

## 6.当前任务接续点

恢复后不再从已删除strict300/EvidenceNorm/JP-R4路线继续。下一算法接续点为：

1. 以ADV3B02＋P4为基底，把`JG_R8_LR020`抽成真正support-only enrollment kernel，使适配进程的文件权限中不存在query；
2. 在K10开发row中同时输出注册前旧类、注册后旧类、5/10/20新类、`H_old_new`、最低旧类和遗忘；
3. 对所有注册类使用同一prototype/head规则，保持无role Oracle、无class quota、无query拟合；
4. 保留默认1-view，依据单query margin/entropy/View分歧触发3/5-view；
5. K10锁定candidate后再运行K1/K5/K20，不用这些query重新调参；
6. 每三轮算法探索后执行正式复盘，再决定第四轮。

最高风险项不是source适配器本身，而是当前清理后的主线尚未重新建立合法、可运行的target Stage2-C support-only enrollment→truth-free predictor→isolated scorer闭环。该闭环完成前不能启动正式确认矩阵，也不能给出新类注册后性能声明。

## 7.证据来源

|来源|用途|
|---|---|
|当前主session JSONL第6383、11083、11132、11810、11826、11905、13340、13792、14366、14578、14617、14633、14652、16186、22565、23287、23783、24137、24419、27349、28014、28016、28084行|恢复用户直接引导|
|当前主session JSONL第27290至27291行|恢复缺失turn总结|
|`automation_reports/CV-SincNet/qknnv42_extreme_light_optimization_20260715/report.md`第17.8、18.2、18.4节|核对v21/v22/v23同row指标、资源与artifact SHA|
|`automation_reports/CV-SincNet/qknnv42_session_cleanup_20260716/report.md`|确认污染会话删除范围、保留谱系与隔离边界|
|`项目.md`第7.1、7.2、8.4、8.5、9、10.3.1节|确认当前数据权限、无Oracle、Stage2-C和目标门槛|

本恢复报告为严格证据恢复，不是对缺失内容的近似重写；但聊天界面气泡无法原位重插，因此“界面层恢复”只能由本报告和当前回复替代。

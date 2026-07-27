# CVS-RFFI英文初稿主张—证据矩阵

日期：2026-07-28

用途：约束摘要、结果、结论、投稿信和后续改稿中的正式表述。任何新结果只有在同协议、同row、完整artifact闭合后才能替换本表。

|编号|拟写主张|证据来源|证据等级|允许表述|禁止表述|下一步|
|---|---|---|---|---|---|---|
|C1|CVS-RFFI研究source-only跨接收机DG以及部署后的support-only旧类适应与新类注册|`项目.md`、`p2_min_v1`|协议定义|“protocol-governed two-stage framework”|“single end-to-end training protocol”|保持不变|
|C2|Phase2每条物理IQ只有一个固定LEO仿真观测；K-shot是K个独立物理样本|`项目.md`|协议定义|“one fixed received observation per physical sample”|“multi-view observations increase K”|在数据表补物理ID计数|
|C3|Phase2不读取query真值、真old/new角色、类别配额或query-query关系|`项目.md`、predictor/scorer流水线|协议与实现闭合|“independent all-registered-class prediction”|“role-aware deployment”或“batch-balanced prediction”|保留独立scorer receipt|
|C4|Phase1选中候选达到`overall=89.18%`、`strict UDU=84.89%`、`receiver_floor=75.55%`、`satellite strict floor=68.77%`|`phase1_adv3_mechanism32_queue_20260701/full_analysis_20260702.md`|完成的内部审计|按原指标逐项报告|“statistically significant SOTA”|补冻结独立复验和区间|
|C5|Phase1相对ADV2均值的差值为`+2.26/+4.80/+6.48/+0.94pp`|同C4|描述性比较|“descriptive improvement”|“paired causal gain”|补相同seed/相同split比较|
|C6|Phase1不能证明unknown拒识|`source_overflow=0.459`、`bridge_accept=1.0`|完成的负诊断|“closed-set DG evidence only”|“open-world rejection solved”|另行完成Phase3|
|C7|RTB-IDR是从固定IQ与support构造统一全类仿射头的完整Phase2候选|`docs/D92_METHOD_COMPLETE_REPORT_20260727.md`、代码和125运行|实现与执行证据|“complete support-only candidate pipeline”|“promoted final deployment method”|等待后续候选|
|C8|RTB-IDR的任务均衡协方差在`K=10,new20`上提升注册后旧类2.622pp、最低旧类4.600pp、H 0.964pp，并降低遗忘2.622pp|D92 retry2的D81/D92 matched 125矩阵|严格paired组件效应|仅作为任务均衡协方差的paired效应|把效应归因于整个RTB-IDR相对所有基线|保留同row上下文|
|C9|同一行的新类准确率下降0.653pp|同C8|严格paired负效应|与C8同时报告|只报旧类提升、隐去新类退化|设计合法跨角色校准|
|C10|K1上RTB-IDR区别性模块不激活，paired增益为0|D92 K1 fallback和结果|结构边界与实测|“no identifiable one-shot gain”|“one-shot adaptation solved”|设计K1可识别受限先验|
|C11|RTB-IDR绝对指标未达到项目门槛|D92报告：`K10/new20 old=71.333%,min-old=42.667%,new=68.150%`|完成的负结果|“diagnostic, not promotable”|“deployment success”|运行新的正式主候选|
|C12|CSIL/MoPC-HR在CVS接口的结果可作描述性对照|800cell正式复现报告|不同权限的外部比较|显式列权限和old-before差异|“strict paired leaderboard”|补同capsule支持型基线|
|C13|26类RTB-IDR核心数组约16.11KiB，仿射头7488MAC/query|D92资源审计|解析式与artifact证据|“compiled head storage/compute”|“end-to-end system is 7488MAC”|补主干、FFT/RF和硬件测试|
|C14|WiSig/ManySig和LEO仿真信道均为代理；当前IQ仅实现后同步残余基带链路|`项目.md`、`code/sat_channel.py`、`code/training_controls.py`|场景与实现边界|“terrestrial proxy and simulated LEO residual channel”|“real on-orbit validation”或“complete satellite link budget”|真实卫星或硬件在环|
|C15|论文当前可作为IEEE Transactions结构化初稿|`manuscript.tex`和编译PDF|文稿交付证据|“initial evidence-locked draft”|“submission-ready Q1 paper”|清除全部AUTHOR ACTION|

## 摘要允许使用的数字

- `89.18% overall`
- `84.89% strict UDU`
- `75.55% receiver floor`
- `+2.62pp old / +0.96pp H / -0.65pp new`，必须同时出现`K=10,new20`和“versus matched control”

摘要不得使用历史qKNN V76--V89高分、Role-Oracle上界、不同新类规模的跨矩阵最优值或外部方法的非paired差值作为主方法胜出证据。

## 必须保留的限定语

- terrestrial proxy
- physics-inspired simulated LEO residual channel
- support-only
- immutable Phase1 bundle
- independent all-registered-class decision
- matched component diagnostic
- not in-orbit or flight-software validation

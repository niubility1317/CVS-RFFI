# 现行Stage2自研方法对比追踪

完整中文报告：`E:\type10-7\automation_reports\CV-SincNet\stage2_method_comparison_20260725\report.md`

报告SHA256：`536486f4c828e51e54d79774358940f055d0c5a5f4a3f8af47194a6b7ce6c104`

状态：`EVIDENCE_BOUND_COMPARISON / NO_METHOD_PROMOTED`

|方法|证据|核心结论|
|---|---|---|
|D62|完整125|总体B/A/N/H/F=`81.51/64.39/59.11/61.09/17.11%`；相对D81所有关键paired CI跨0|
|D69|development 15row|B/A/N/H/F=`92.78/81.67/74.67/77.39/11.11%`；跨stage拼接负交换|
|D91|development 15row|B/A/N/H/F=`92.78/82.22/84.67/82.62/10.56%`；15/15 prediction与D62相同|
|D92|完整125|K10/new20相对D81：A-old+2.622pp、Min-old+4.600pp、N−0.653pp、H+0.964pp；K1无效|
|SVRN/r4.2|完整125|总体B/A/N/H/F=`73.10/43.03/23.46/29.25/30.07%`；125/125个matched row的N与H均低于D62|
|D102|source-held|K1/K5/K10均值BA略正，但TX max BA=50.3199%和9/42 LOCO退化导致拒绝|
|D103-R2|21条truth-free row|21/21活动；K10两个receiver合计3次INT8翻转，release撤回|
|D104-R1|8400条tap全池向量＋2条已知失败row|含2478条历史诊断query、0新held；7575改善、825相同、0退化；两个K10行修复到0翻转；尚无性能|

完整报告按完整125、development cell、source-held falsifier、truth-free技术证据和设计草案分层，未将不同证据等级混排为同等性能结论。D92 role-Oracle许可上界、旧clean/source诊断和partial技术失败score均排除在正式比较之外。

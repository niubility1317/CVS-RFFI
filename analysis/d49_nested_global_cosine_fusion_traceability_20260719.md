# D49严格nested全局余弦原型融合追踪

|需求|实现位置|验证|状态|
|---|---|---|---|
|D42全局单位球上的cosine prototype，不增加query view|D49 `_cosine_component_fit`|support重算＋reference/affine逐元素等价|VERIFIED|
|顶层held fold完整重拟合D45及其内部LOO权重|D49 `_nested_head_evidence`|每折state/audit、held/train交集0、实际targets绑定|VERIFIED|
|两head使用各自inner-train/full-support RMS|D49 fit＋verifier|从绑定support/state重算RMS与held logits|VERIFIED|
|global权重为稳定softmax(-C×macro CE)|D49 `_strict_weights`|CE重算、tie、endpoint、和为1测试|VERIFIED|
|K1逐位回退D45，K2不强制顶层1:1|D49 fit|K1 bitwise、mock K2及真实locked K2|VERIFIED|
|old/new同式本类support prototype|D49 fit＋top-level wrapper|before/new-support隔离、targets/inputs绑定|VERIFIED|
|一次FP32融合、canonical center、既有int8/FP16编译|D49 fit＋resource wrapper|matched FP32逐位绑定、独立重编译int8/FP16|VERIFIED|
|FP32/int8精确top-tie fail-close|runner score guard|support及outer score入口测试|VERIFIED|
|K8 before+final共292次LDA|D49资源wrapper/verifier|8组inventory逐项与MAC闭合|VERIFIED|
|无query/clean/source/role/quota/scan|继承D42 runner＋D49 verifier|105/105行artifact中query读取/fit/truth/role/quota/count/global assignment均0，clean/source均false|VERIFIED|
|每版完整性能与行为报告|D49 report第12–22节＋`full_performance_summary.json`|完整读取D49/D45各105行，保留7候选、3场景、11类、15折、20步、机制、量化、资源及matched差值|VERIFIED|

最终运行前独立复核：P0=0、P1=0。strict nested仅指冻结outer-B20后的head层，不得扩写成全链路nested或无泄漏泛化。D42–D49全链144项通过。

运行后：105/105行完成，D49 int8与matched FP32逐决策一致且15/15个outer预测SHA相对D45改变；但总体after-old/new/H分别为76.67%/72.67%/73.90%，相对D45下降5.56/11.33/8.25pp，forgetting增加4.44pp。low-elev/rain的cosine final权重均约0.8且性能恶化最重，确认support nested CE不能作为该head的稳健泛化代理。最终状态为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不formalize、不运行125。

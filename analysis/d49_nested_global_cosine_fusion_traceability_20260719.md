# D49严格nested全局余弦原型融合追踪

|需求|实现位置|验证|状态|
|---|---|---|---|
|D42全局单位球上的cosine prototype，不增加query view|待实现D49 script|reference/affine逐元素等价测试|PENDING|
|顶层held fold完整重拟合D45及其内部LOO权重|待实现D49 script|nested audit、held/train交集0|PENDING|
|两head使用各自inner-train/full-support RMS|待实现D49 script|RMS闭包与篡改测试|PENDING|
|global权重为稳定softmax(-C×macro CE)|待实现D49 script|CE tie、端点、和为1测试|PENDING|
|K1逐位回退D45，K2不强制顶层1:1|待实现D49 script|K1/K2测试|PENDING|
|old/new同式本类support prototype|继承runner生命周期＋D49 fit|before/new-support隔离验证|PENDING|
|一次FP32融合、canonical center、既有int8/FP16编译|待实现D49 script|state SHA/数组重编译验证|PENDING|
|FP32/int8精确top-tie fail-close|待实现D49 verifier|合成tie测试|PENDING|
|K8 before+final共292次LDA|待实现D49资源wrapper|fit inventory/MAC闭合测试|PENDING|
|无query/clean/source/role/quota/scan|继承D42 runner＋D49 verifier|105行artifact审计|PENDING|
|每版完整性能与行为报告|D49 report第9节及完成章节|全日志解析和同候选同场景同类表|PENDING|

当前设计审计：P0=0；原三块query归一化和非nested D45复用各有P1，均已在预注册中删除。代码尚未实现，outer结果尚未读取。

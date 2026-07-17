# D33球面同尺度注册与Fisher快速适应追溯表

|ID|要求|落地|状态|证据|
|---|---|---|---|---|
|D33-01|单一LEO_weak IQ|z160/FFT96/RF32均来自同一已接收IQ，不增加view/overlay/K|implemented|共享sealed support入口|
|D33-02|域适应与注册同等重要|Stage2-B对角适应+Stage2-C球面注册，同run before/after|implemented|D33报告与candidate设计|
|D33-03|消除新旧标尺失配|old/new统一transform、centroid、radius和`-d/r-log(r)`评分|verified|D33 core测试|
|D33-04|floor优化|A overall、B balance、C floor三种固定LOSO排序；类半径median shrink+ratio cap|verified|D33 10项测试|
|D33-05|快速适应|Fisher近闭式0步，6旧类2,016标量，MAC较Adam15估算降低84.10%|verified|Fisher 4项测试|
|D33-06|K1与多新类|K1统一半径纯cosine；2/5/10/20新类|verified|D33参数化测试|
|D33-07|轻量部署|20新类活动7,828参数、实际常驻8,848B、无FP32 centroid、无dense图|verified|int8 resource audit测试|
|D33-08|无query/Oracle|API无query/role/quota；逐样本all registered classes|verified|协议字段与签名测试|
|D33-09|自动化闭环|v11 lock、105行、fold/full/resource/selection/receipt/source closure|verified|54项相邻测试；2-new四候选fold/full smoke|
|D33-10|真实实验|逐类、场景、floor、trace、资源、artifact闭环|pending|N607待启动|

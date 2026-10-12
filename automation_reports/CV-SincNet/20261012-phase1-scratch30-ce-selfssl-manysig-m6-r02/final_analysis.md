# 从头30轮 CE／当前学生半监督：完整最终结果

VERIFIED：六个固定finalE30模型，三个seed；每模型clean和六practical各168000，全部view先prediction后独立truth-last评分。每row自动收尾，不等待全批训练结束。

|方案|视图|三seed准确率均值|SD|最差RX均值|
|---|---|---:|---:|---:|
|CE30|clean|79.4562%|1.8033%|70.4111%|
|CE30|practical_high|54.8708%|1.7848%|40.1333%|
|CE30|practical_mid|52.2256%|1.7521%|39.1611%|
|CE30|practical_low_suburban|49.8625%|1.7273%|38.1569%|
|CE30|practical_high_urban|52.1266%|1.7327%|38.5208%|
|CE30|practical_mid_urban|41.1073%|1.4220%|32.1000%|
|CE30|practical_low_urban|33.2760%|1.2588%|26.9667%|
|SELF_SSL30|clean|79.4310%|1.3557%|70.8583%|
|SELF_SSL30|practical_high|54.4907%|1.7910%|40.2931%|
|SELF_SSL30|practical_mid|51.8938%|1.8877%|39.6014%|
|SELF_SSL30|practical_low_suburban|49.5312%|1.8860%|38.2736%|
|SELF_SSL30|practical_high_urban|51.7833%|1.7867%|38.6472%|
|SELF_SSL30|practical_mid_urban|40.7792%|1.5023%|32.2000%|
|SELF_SSL30|practical_low_urban|33.0510%|1.3227%|27.0097%|

预登记主方案SELF_SSL30 clean三seed均值≥84%：False。

全部source从零30epoch/6660更新，无旧模型/固定老师、EMA、卫星训练或辅助网络。CE30实际不打开U；SELF_SSL30前5epoch纯CE，之后当前student clean eval软目标、95%置信度及相邻完整遍历两次稳定门控、.015固定噪声强视图、5epoch线性ramp至.25。U真值始终隐藏，V仅epoch10/20/30健康评估，无选模。

逐seed、RX/day/TX、混淆和配对差见analysis.json，完整6660步source日志保留。目标结果未回流训练或候选排序，所有负结果保留。数据为地面代理及固定模拟practical，不能称真实卫星部署或新的独立盲测；K与Phase2注册均N/A。

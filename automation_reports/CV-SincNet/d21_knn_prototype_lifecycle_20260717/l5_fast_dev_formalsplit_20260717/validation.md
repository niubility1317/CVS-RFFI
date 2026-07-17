# 正式切分对齐与执行验证

- 状态：PASS，36/36个cell完成；split policy=`somph_offline_split_v1`。
- 正式排序键：`SHA256(somph-offline-split-v1|receiver|seed|role|tx_label|sample_id)`，并以`sample_id`作并列键；receiver=`20-1`、seed=`713101`。
- 每类support=`ordered[:K]`，固定query=`ordered[20:40]`；new5/new10/new20严格取同一new20 master registry的前缀。
- K10/new5对齐：3个场景support均110行、query均220行；support IQ顺序、query IQ正式全局顺序以及逐行post-channel IQ SHA256均与现有formal capsule完全一致。
- K10/new5 pooled：after old=`0.7305555556`、seen-new=`0.7733333333`、H=`0.7513360424`；相对目标显示值`.7306/.7733/.7513`的差分别为`-0.0000444444`、`+0.0000333333`、`+0.0000360424`，均只是四位小数显示误差。逐场景H的算术均值为`0.7508432303`，不得与pooled H混淆。
- query预测隔离：`predictions_truth_free.json`不含truth、role或quota字段；query标签和角色只保存在`scorer_truth_sidecar.json`并由评分函数读取。
- 算法阶段耗时：3.460秒；`py_compile`通过。

## Formal capsule IQ根

|场景|support IQ root|query IQ root|
|---|---|---|
|leo_clear_weak|`5a5d46047262701df5ac8b324f0c6f41ddf9d334e12f45590c16e1f8a53aa3f1`|`99be2d0bbac3b985116b3253d4ffacc964270aaa205d7aba347a3eb047f037d9`|
|leo_low_elev_weak|`deca9133ac2dc7f9cc267138f496723a89f010c6fa9a837cfb320b5280dbdd02`|`6317592a3456b7222feeb0f742f4d4cbfa038f7bf40ad2675ed0ece7781fdfe7`|
|leo_rain_weak|`b29115079759e9dc4bd1585353777f106204429050240cb3946d96f4f014af8b`|`8f5303b2b0feba1d8b6f76b6db05b2d76a99d5aae586225b1f415a0c1ddb9762`|

## Formal capsule逐行SHA256集合根

集合根定义为对逐行`post_channel_iq_sha256`去顺序排序后，以canonical JSON编码并取SHA256。脚本对mother选择结果和formal capsule分别计算、逐元素比较；下表每格是两侧共同根，三场景support/query均为`EXACT_SET_MATCH`。

|场景|support SHA集合共同根|query SHA集合共同根|状态|
|---|---|---|---|
|leo_clear_weak|`6779389c5912a35514c00220c09106510b65102d486edaea10675d10ee50e180`|`498e09daf621658f81570e5619056a767e04120a3d9d3858fae2a68954ed5625`|EXACT_SET_MATCH|
|leo_low_elev_weak|`fc430a873272120519f1c044059d447a8badf35a576044e0acc13955c80a7982`|`797cfaf93b6f6dd4917692c3286a99da7362ad714dc9906478f96579fc879c8d`|EXACT_SET_MATCH|
|leo_rain_weak|`0b63751f3acb68363021a526948d875492b3b4d8a1e59df98423646168fbfd02`|`fa017ef3f63cda808c96702a7831e89189f2e62d161780cc76560b3a29a23140`|EXACT_SET_MATCH|

## 输出SHA256

- `results.json`：`44be2ecbb224dc95db55cc3b46bd8694767dbe59f5b51d3ba3f192afcbce4355`
- `results.md`：`6521ad0a0c61fc6bbef990a56c7150eb1c0f5d44ff8ff6e97b8778ae70e7d7b6`
- `predictions_truth_free.json`：`69fd0c6c29b8131c9afe5793b19f10c51254c7bd960407ea269357fb34b19889`
- `scorer_truth_sidecar.json`：`b1b839bbd3d3ea773d86c66a13aee13f3ff3b55491c6eca8c5a9049c53c40b7a`
- `command.txt`：`8ef4c645f16a9e3620e0c17960fe0e3f8926a56555334fcfa6fd413b0774056b`
- 执行脚本：`../l5_fast_dev_20260717/run_l5_fast_dev.py`，SHA256=`52542568beafc97316d3a80de5a3bb1b012cd2003a958bc5284ce9a13628c9fe`。

相邻旧目录`../l5_fast_dev_20260717/STATUS.md`已明确标记`SPLIT_MISMATCH_DIAGNOSTIC`，其性能禁止引用、比较或晋级。

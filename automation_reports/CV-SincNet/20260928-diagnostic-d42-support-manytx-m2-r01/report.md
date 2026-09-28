# D42 support-only技术诊断

只读取两失败row的固定support和新训练模型v2 ground。禁止query/truth或目标准确率参与。逐行配置见experiment.json。原失败输出保留。

状态：LOCAL_VERIFIED。诊断代码commit 281b92099b76233c1d4dc341372426894006bd84；本地compile通过；独立P0/P1由主Agent确认。

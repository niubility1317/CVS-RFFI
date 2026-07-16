# D3 K10开发结果

候选：`d3_scenario_oldlock_newfit`

范围：receiver=`20-1`、seed=`713101`、K=`10`、真实seen-new=`5/10/20`。所有support/query均来自已预叠加三种`LEO_weak`信道的密封包，predictor无clean/source/query truth/role/quota/global assignment访问。

|new|B old|B floor|C old|C floor|seen-new|H|forgetting|结论|
|---:|---:|---:|---:|---:|---:|---:|---:|---|
|5|90.28%|73.33%|86.39%|68.33%|81.67%|83.96%|3.89pp|FAIL|
|10|90.28%|73.33%|87.78%|70.00%|77.17%|82.13%|2.50pp|FAIL|
|20|90.28%|73.33%|83.33%|58.33%|77.00%|80.04%|6.94pp|FAIL|

每行before/after均包含3场景×20epoch=60条完整loss记录，无NaN、Inf、OOM或traceback。三规模、三场景、每个旧类support的`old_class_intrusion_count`均为0，但query floor显著下降。D3未通过预登记门，不进入125确认矩阵。

机制结论：

1.D1共享head为2,022参数、180条support；D3拆为3个各2,022参数、各60条support的head，参数/样本比扩大3倍。三场景均在epoch5–7达到support 100%，但query B显著下降，属于已收敛的scenario-head过拟合。
2.由B正确变为C错误的旧query在new5/new10/new20分别为14/9/25条，100%都是old→new；旧头冻结有效，但support margin没有覆盖query尾部。
3.新类错误的new→old/new→new计数为new5 33/22、new10 55/82、new20 94/182。多新类下new→new主导，单一old/new offset不足。
4.困难新类`20-12`、`14-11`、`4-10`、`1-18`仍未分离；new20最弱旧类`20-19`仅58.33%。
5.下一路线应继承D1共享Stage2-B旧头，只追加一个跨三场景共享的极轻新类head，并同时约束true-old逐类margin和new-new angular separation。

资源上限均满足：最大C trainable parameters=`17,280`，最大实测NPZ状态=`102,688B`，每场景20epoch，无dense query图。性能失败而非协议或资源失败。

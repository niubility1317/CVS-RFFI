# Phase2 T1可行性冻结
状态：`DESIGN_FROZEN -> IMPLEMENTING`
1.科学协议固定为`p2_min_v1`，仅消费同一P1 bundle封存知识和target support，fit接口不接受query。
2.Stage2-A/B/C统一factory；补齐`P2-S2A/S2B-PROTO/S2B-DIAGOFF/S2B-FULL`及19个Stage2-C逻辑arm。
3.`P2-FULL`采用D81稳健中心、D92任务均衡协方差、D46类级LOO、D62 Fisher安全门、D42统一等先验头和F3量化。
4.7个同权限基线共享bundle、capsule、physical IDs、support/query和逐样本全注册类argmax。
5.K≤2统一走`P2-FALLBACK-KLE2`；必须证明state、FP32/量化logit和prediction逐项闭合。
6.逻辑arm按effective config+input binding动态物理去重；alias不增加样本量、CI或显著性检验。
7.预测先不可覆盖封存，truth scorer后打开sidecar；调度器不得读取性能值决定停止、重跑或晋级。
8.screening固定75 row/arm，confirmation固定900 fresh row/arm；support/query/class draw与P1 bundle seed分别绑定。
9.16固定lane为8 GPU×2，外部训练PID计入上限；占用未知时失败关闭。
10.T1先实现主arm、Stage2-A/B状态、7基线、fallback、scorer和16-lane runner。
11.T2内部arm、T3连续注册/资源和T4非晋级诊断保留设计顺序，不用screen结果删除主表arm。
12.实现复用D42/D43/D45/D46/D61/D62/D81及D92最终树`f65f8934`，不移植旧结果或旧8-shard runner。
13.方法、执行器和scorer分文件所有权；主代理唯一整合`full_ablation_spec.py`和正式row pipeline。
14.发布门为真实P1-FULL bundle、capsule/split/physical-ID hash齐备，聚焦/回归/真实checkpoint测试通过且独立`P0=0,P1=0`。

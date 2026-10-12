# 独立 P0/P1 发布审查

结论：**PASS；剩余 P0/P1 为 0。** 日期：2026-10-12；审查者：`/root/stage2_release_review`。

范围为 `20261012-phase1-crossrx-relation-manysig-m15-r01` 的固定设计、实际目标与来源、源训练、冻结预测、评分、发布和自动报告。审查者未参与本包实现，仅写本文件；未使用 GPU、操作远端、启动实验或读取真实目标预测/评分产物。本结论确认发布路径正确性，不代表真实运行已完成或性能有效。

## 设计与实际路径

- 15 行登记 config 与当前 `design.config` 逐项相等：5 方案 × 3 seed，随机无显式 CFO 学生20轮／4440更新，共66600更新；固定 final20。slot27屏蔽和内部补偿与无CFO纯CE基线相同。原身份CE及固定其他两seed源CE教师ALL软KL保留；无EMA、卫星、Fishr或新增辅助头。
- 每次额外30L覆盖6TX×5RX、同day且循环3day，支持内PID唯一，排除主L查询的全部PID。999次调用对应448至4440的每4步，总29970额外L读出。每query按自己的RX排除该RX参考，中心来自其他4RX的24个参考，全部6TX竞争。
- 合并原型为各TX跨其他4RX的中心化均值再归一化；分RX原型分别归一化后做cosine／tau与logmeanexp，tau固定0.1。组合层等权平均。浅time_down、中t2.pool的均值与population std来自主clean查询实际前向；final_joint也保留查询梯度。支持由当前学生eval/no_grad读出，原型不反向，模式恢复并检查buffer不变。
- L30_CE使用同支持采样、间隔及ramp，以额外L普通train前向和CE反向，恢复主RNG。它匹配额外样本与前向数量；设计及报告没有声称与eval/no_grad关系参考具有完全相同计算成本。
- 教师proof与上批noCFO缓存生产者逐项一致：固定新CE run／提交／seed／config／final20、源角色和空祖先；缓存独立schema、U无TX真值、PID对应及概率合法性检查保留。仅读上批已固定U缓存，不生成新缓存或按目标成绩选教师。V仅健康评估。CE与固定SSL预测只在独立scorer中作同seed对照，不进入训练。
- 每行训练冻结后立即clean168000预测，然后独立truth-last评分；完整15行共2520000预测。共享容量包含现有CUDA所有者和启动前预约，每卡最多3个总进程；独占新run/release/output、唯一owner，UNKNOWN不自动重发，失败保留产物并不干预健康任务。实际归档列表65个路径全部存在，新包直接跨包导入均被覆盖；preflight和dispatch入口正确。

## 验证证据

已读回登记的 `crossrx_relation_cuda.json`、`crossrx_relation_local.json` 和 `crossrx_relation_analysis_checks.json`，均PASS：五方案原生CUDA目标和优化入口、strict own-checkpoint、PID/RX排除及TX重命名、模式/RNG恢复、真实source入口，以及五方案合成22200步完整源解析、每行999关系调用和非法target标记拒绝。未重复运行CUDA验证。

独立执行以下本地CPU合成检查：

1. 逐query／TX／RX手工计算两种原型规则，并比较CE查询梯度。合并／分RX logits最大绝对误差分别2.22e-15／1.78e-15，梯度误差分别8.33e-17／1.39e-16，均通过。
2. 原生随机网络三层单独关系损失均可反向到身份参数；time_down、t2.pool、id_feat_joint梯度范数分别约4.16、12.87、4.23。参考表示全部detach，参数未更新，参考前向的buffer、mode及RNG保持不变。
3. 实际调用 `final_report.finish`、collector及替换后的完整collection payload，验证15行、210条主评分记录、60条日期记录、84条去重CE/SSL参考记录、13项配对；完整报告、资源、experiment状态及events正常生成。只在临时previous_run中设置共享缓存，新run没有peer_cache，仍能完整收尾，确认CACHE替换及复用路径正确。

最后一项使用合成已评分数据，并mock源日志扫描的已闭合结果；它验证收尾字段和流程，不冒充真实66600步训练或实际测试成绩。源解析本身另有上述22200步合成证据。

本次未发现需要修复的P0/P1；无额外审批或性能门槛。

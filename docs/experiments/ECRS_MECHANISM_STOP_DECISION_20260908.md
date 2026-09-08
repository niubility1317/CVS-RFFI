# ECRS剩余组机制差异与条件停止

用户要求核对剩余4组是否存在核心变化，无核心变化则关闭训练。本次据实际决策路径分组执行，而不是把V2目标分数用于训练调参或候选重排。

|组|相对已完成组的差异|处理|
|---|---|---|
|B0|关闭ECRS，原身份主干基线|已完成|
|B3c/B4|V2 compact8估计器；real8/complex24响应读取；独立response CE；融合关闭|已完成|
|B3a|V2 legacy28旧参考，real8读取；物理估计冻结；response CE独立；融合关闭|按用户条件停止|
|B3b|B3a改为估计参考/规范化参考框架；仍是legacy28/real8；融合关闭|按用户条件停止|
|B2-V1|旧V1 R7完整目标与旧梯度接线，额外raw CE=0.30；旧响应分类共用身份分类路径|继续：具有训练目标/梯度差异|
|B2|V1R修复包，保留旧物理目标；独立response CE=0.15；额外raw CE=0；响应与物理识别梯度隔离|继续：不能视为B3c/B4重复|

B3a/B3b在响应表示层有实际差异，但没有新增最终raw身份决策通道。`ecrs_v2.py`的物理响应在encoder前detach，fusion=off时z_id_fused直接取z_raw；`model_dual_cvsincnet.py`将V2独立响应头与基线raw训练路径分开。它们主要补充消融与估计器比较，继续训练不应被寄予raw性能大幅跃升的直接机制预期。共享训练的随机数/数值路径可能造成raw差异，不能声称各组必然逐位相同。

B2-V1/B2与固定物理估计的V2不同，保留canonical/content/cycle/split-fit/pair等物理学习目标，且raw CE、分类头共享关系及梯度路由发生改变。这是真实训练机制差异，不保证会有大提升，但不满足“无核心变化”的停止条件。两组尚未训练到E91的后期响应阶段，不能提前宣告完整机制无效。cross-RX/U/fusion在这4组均未启用；也没有启动B5/B6。

停止范围仅B3a PID3537463、B3b PID3547432及其已核实数据加载子进程。队列pending为空，dispatcher仅记录退出且不会重启。保留best/latest、CSV、JSONL和stdout；停止属于用户资源决策，记USER_CANCELLED，不记技术失败或E200完成。B2-V1/B2及其他实验不受影响。

执行前证据见[ecrs_stop_20260908/before.json](ecrs_stop_20260908/before.json)，精确PID、启动时钟、父子关系、CWD、argv和checkpoint文件均已核实。停止脚本先本地验证并进入Git，再发布执行。最终停止状态以独立读回补充。

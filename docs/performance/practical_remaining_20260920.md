# Practical LEO剩余执行优化与服务器验收

本轮完成了物理常量缓存、相位递推C实现、不可变full配置复用、训练轻量元数据、residual恒等滤波短路、clean验证特征复用、固定评估有界预取，并补充IQ输入缓冲复制和增量epoch日志。跨训练batch预取未实施；原因与替代范围见下文，不能把固定评估的结果写成训练预取结果。

## 验证范围

- N607普通用户、独立benchmarks目录、物理GPU3；原活动实验未停止、未覆盖、未重新启动。本轮没有读取目标truth或历史checkpoint，没有开启正式训练。
- 合成B128×2×256 IQ，四种配置×六个环境；单项、组合、组合加双进程各预热后测5次，报告中位数。输入复制31次，日志累计200epoch重复3次。
- 基线是上一轮已经有FIR滤波核缓存的串行路径。局部倍数不能相乘，也不能当成端到端epoch倍数。
- N607训练环境Torch2.1/NumPy2.2边界保持不变。测试依赖离线放入独立目录，没有修改训练Conda环境。90项回归测试通过。
- 信道检查IQ逐元素完全相同、终态相同、完整元数据或轻量保留字段一致；真实CVS验证检查指标、模型参数/缓冲及RNG一致；预测器检查相同预测与独立计数；还检查预取生产者/消费者异常清理、负视图/非连续输入及动态日志schema。

## 逐项实测

|优化项|N607实测|处理|
|---|---|---|
|物理常量缓存|1.016～1.063倍|数值通过；各配置与六环境均已测试|
|相位跟踪C递推|1.017～1.064倍|数值通过；各配置与六环境均已测试|
|复用不可变full配置|1.024～1.036倍|数值通过；仅residual适用，full中的约1倍波动不是收益|
|训练轻量元数据|1.041～1.115倍|数值通过；各配置与六环境均已测试|
|residual无均衡恒等滤波短路|1.011～1.024倍|数值通过；仅residual适用，full中的约1倍波动不是收益|
|clean验证复用|1.685倍；26.42→15.67ms|真实双分支CVS模型；保留第二次DataLoader遍历|
|单线程CPU生成+预取|0.985倍，略慢|不推荐开启此组合|
|双进程CPU生成+预取|full1.100倍，residual1.538倍|八批次合成固定评估流水线，输出相同；仅双进程组合建议开启|
|IQ输入缓冲复制|18.07倍；3.979→0.220ms|去掉Python浮点列表，仍保留float64参考计算|
|增量CSV/JSONL|91.95倍；17.079→0.186s|200epoch模拟日志累计节省16.894s；不是每epoch节省此数|

## 组合结果

|信道配置|本轮组合/此前串行|组合加双进程/此前串行|mid场景：此前串行→组合加双进程|
|---|---|---|---|
|full无均衡|1.129～1.136倍|2.150～2.248倍|395.16→177.83ms|
|full正则化限增益ZF|1.092～1.108倍|2.058～2.174倍|456.87→222.03ms|
|full MMSE|1.090～1.113倍|2.151～2.194倍|455.86→211.94ms|
|residual无均衡|1.312～1.328倍|2.531～2.556倍|174.30→68.87ms|

六环境为high、mid、low_urban、low_suburban、mid_urban、high_urban。本轮没有通过性能评分改变训练采样比例，也没有把六环境全部设成训练视图。

## 接入及推荐

所有新增行为默认关闭，历史recipe保持原样。未来训练可在独立recipe显式开启：

```json
{
  "practical_execution_fast": true,
  "practical_buffer_input": true,
  "practical_buffer_output": true,
  "practical_channel_workers": 2,
  "source_validation_reuse": true,
  "incremental_epoch_telemetry": true,
  "practical_eval_prefetch": true
}
```

`practical_execution_fast`在每个训练epoch/角色/场景首次调用仍记录完整证据，之后仅生成训练实际使用的元数据；固定评估始终完整。随机信道仍按物理ID、epoch/角色namespace和增强种子重新生成，不缓存固定训练IQ。clean与星地视图的拼接、监督损失及普通增强先于星地增强的顺序未改变。

`source_validation_reuse`仅支持已审计的真实CVS双分支模型，逐批核对输入及模型版本，缓存上限128MiB；超限、输入改变或模型状态变化时回退。它去掉重复模型前向，保留原DataLoader迭代及随机数消耗。日志仅在字段集合增加时重写CSV，JSONL逐epoch追加，历史内容保持一致。

## 预取边界与最终复查

跨训练batch预取仍未实现。普通增强使用全局Torch RNG；直接提前产生下一batch会改变与模型随机运算交错的顺序，无法声称保持旧训练轨迹。现有实现只接入固定评估，通过单任务队列让CPU生成与GPU推理重叠。不得把这个实现计为已经完成训练跨batch重叠，也不得为了加速固定整轮训练增强。

最终优化后residual的cProfile显示：B128累计约163.4ms，其中初始化约76.9ms，命名RNG生成器构造约48.7ms，滤波约21.8ms。它们是嵌套累计时间，不能直接相加；profile本身也有开销。后续仍有三类真实空间：

1. 为独立记录构造命名RNG的开销仍明显；可研究延迟创建未用的equalizer RNG、拆出residual专用初始化，但必须保持各命名流的派生与抽样顺序，不能改成共用一个RNG。
2. full仍有逐采样状态/阴影递推和逐record滤波，保真批量化需要覆盖连续流历史、边界处理和状态跳变，不能用residual近似冒充full加速。
3. 训练循环仍有诊断性`.item()`同步；应按实际激活分支的profile合并诊断读取，不能删除非有限值检查、梯度裁剪或算法反馈所需读数。训练跨batch重叠需独立设计随机状态隔离后再验证。

双进程预取补测使用Practical mid、八批次、轻量元数据及随机初始化CVS模型，不包含磁盘缓存和完整预测器文件写入；它是流水线组件证据。实际固定评估仍保留完整元数据，因此不承诺完整评估也达到同一倍数。此前单线程CPU生成的负收益与双进程补测使用不同批次数，不能直接把两种加速倍数相除；各自只与同配置的串行消费对照比较。

固定评估仍同步执行CPU独立评分；本轮没有改成异步评分。仅依据代码存在不能断言这些后续项能带来多少整轮收益。本轮尚未测真实完整训练epoch，不能给出200epoch总训练时长承诺。

## 证据与交付

- 主基准代码提交：`03d62ccdf89164ac8666004b83fbd549e5fd1d45`。
- IQ/日志代码提交：`3ae14b601848813c07568dbcdc1058253a3ab660`。
- 主服务器目录：`/home/szu2070436088/2510044040/CV-SincNet/benchmarks/practical_remaining_03d62ccd_20260920_b128_r03`。
- IO服务器目录：`/home/szu2070436088/2510044040/CV-SincNet/benchmarks/practical_io_3ae14b60_20260920_r01`。
- 本地原始证据：[主逐项结果](evidence/practical_remaining_result_20260920.json)、[主状态](evidence/practical_remaining_status_20260920.json)、[IO结果](evidence/practical_io_result_20260920.json)、[IO状态](evidence/practical_io_status_20260920.json)、[90项服务器测试](evidence/practical_remaining_tests_20260920.log)。
- 首次GPU0预检拒绝启动；r01因缺pytest未开始基准；r02依赖下载失败未开始基准。旧目录保留，成功r03使用离线测试依赖。独立审查发现的模型类型、消费者异常清理、负视图三项问题已修复并有针对性验证。

- 双进程预取代码提交：`7499df180b8cb0ab9380aedeb8e56c580cccbe68`；[补测结果](evidence/practical_prefetch_result_20260920.json)、[补测状态](evidence/practical_prefetch_status_20260920.json)。
- 补测服务器目录：`/home/szu2070436088/2510044040/CV-SincNet/benchmarks/practical_prefetch_7499df18_20260920_r01`。
- VERIFIED：三个成功基准的结果与退出状态独立SCP读回；最后GPU3为1MiB/0%利用率。其他GPU活动任务保持运行，本轮没有对其发出停止或修改操作。

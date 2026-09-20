# Practical执行加速实施与验证

状态：本地聚焦验证及N607 batch32合成基准完成；profile新增的FIR核缓存将追加batch128验证。用户最新要求优先：不启动任何正式训练实验；本轮仅验证执行加速，场景选择作为未来配方建议。

## 实现范围

- 默认关闭的真实训练profiling窗口：`execution_profile_epoch/start/steps`，写Chrome trace及CPU聚合。从batch进入训练体开始，因此DataLoader等待需另读timeline／后续补充；不宣称当前计时包含取batch之前的读取。异常退出也关闭profiler并保留原训练异常。
- 独立`a1_gradient_snapshot`开关，只改变梯度观测与同步，不自动启用捆绑教师路由的`a1_runtime_fast`。
- `practical_buffer_output`用自有bytearray＋torch.frombuffer承接float32信道输出，避免NumPy ndarray ABI和Python数值列表输出；输入方向仍保留兼容值拷贝。
- `practical_eval_cpu_pipeline`在固定目标评估时先于GPU上传生成／读取Practical IQ，再传入GPU；`a1_eval_identity_only`仅用于预测的身份分支。
- `practical_channel_workers=2`可选spawn进程池，同batch分块、独立物理ID随机流、顺序拼回；并非跨batch预取。
- 缓存不可变Config对象及其配置哈希。没有删除完整Python诊断，也没有缓存有状态ChannelStream。
- 独立模块的新增三个环境接入此工作树；预测和scorer支持clean＋任意唯一Practical场景子集。保留原PRACTICAL三场景常量及全部旧配方。周期覆盖从无标签manifest计算。

所有新的加速开关默认关闭，历史运行不受影响。已经确认无用的源域星地前向修复沿用45bda90a。源域clean验证合并、增量日志、异步CPU评分、配置数组常量、跨batch预取及编译信道内核尚未实现；本轮先确认高价值路径的实测收益。

## 本地正确性

旧25项相关测试及新增31项测试通过。新测试覆盖四信道配置×六环境并行／buffer精确IQ与metadata、动态训练generator推进、输入不变、全局重复ID拒绝、真实CVS模型梯度观测／裁剪／更新一致、非有限梯度检测、实际predictor场景子集与scorer闭合、identity logits与z_id一致、profiling RNG及异常退出、manifest自动覆盖计数。只使用合成数据和随机初始化测试模型。

独立审查无P0/P1，提出的profiler异常关闭P2已修复并回归；建议补测的manifest推导也已完成。不是N607正式epoch保真或整轮吞吐测试。

## 测速定义

`tools/benchmark_execution_acceleration.py`不读取数据集／checkpoint，不启动训练；仅用合成IQ、随机初始化模型的前向和一次反向构造梯度做有界算子基准。batch32，每项预热一次、重复3次，默认单CPU线程，spawn worker继承OMP／BLAS线程限制。

比较对象是当前实现各执行开关的开／关，不是整个历史release对新release的端到端比较；两侧均已享受共享的Config缓存。测量GPU时在每个有界样本边界同步；不把这一模式用于普通训练。进程池启动及磁盘首次生成不计入热态倍数，首次缓存写入另记录。

首次本地benchmark存在跨scene未清空cache配置的问题，其输出`E:/type10-7/.codex_tmp/execution_acceleration_local_20260920.json`作废，禁止用于收益结论。已修正每个case重置配置，并增加bypass/hit断言；有效重跑为`execution_acceleration_local_corrected_20260920.json`。

## N607第一轮结果：VERIFIED

代码94158378，Torch2.1.0+cu121、NumPy2.2.5、RTX3090 GPU0，B32、长度256、float32输入。GPU0启动前无计算任务，结束后显存回到1MiB；未停止／改写其他任务。独立读回result.json与verification_status.json，24case均exact_iq=true，模型identity及梯度观测一致。

|信道|双进程／串行速度比范围|热缓存＋CPU评估／重新生成速度比范围|
|---|---:|---:|
|full不均衡|1.95–2.07|32.03–32.91|
|full ZF|1.98–2.06|38.02–42.89|
|full MMSE|2.02–2.05|38.17–42.81|
|residual|1.91–2.00|22.35–22.54|

identity-only前向17.792→6.456ms（2.756倍）；梯度观测30.205→18.804ms（1.606倍）。这是随机初始化模型eval／既定梯度观测，不是AMP训练step或整个RC4/DAOT epoch。

必须拆开增量贡献：相对已有热缓存路径，CPU前置本轮只额外约1.3%–3.2%速度比收益；上表22–43倍主要来自不再重新生成固定信道。输出buffer单独只有约0.1%–1.8%速度比收益，且本机CPU有波动，不能称其是主要瓶颈修复。

证据：[N607 B32](evidence/practical_execution_n607_20260920.json)、[命令与状态](evidence/practical_execution_n607_status_20260920.json)、[本地有效基准](evidence/practical_execution_local_20260920.json)。远端根目录：`/home/szu2070436088/2510044040/CV-SincNet/benchmarks/practical_execution_94158378_20260920_r01`，日志`benchmark.log`。

## 基于profile新增的FIR优化

首轮cProfile仅取residual/high_urban一个batch：Practical总调用约95.25ms，其中分数延迟核约30.64ms、Kaiser窗约27.87ms，两者为嵌套累计时间，不相加。配置固定时滤波核被反复生成，属于真正的不变量重复。

现将原实现保留为参考函数，以fraction与half_length为键建立最大256项LRU，公开入口返回独立copy，保留旧调用方可修改返回值的语义。无种子、state、equalizer或IQ变化。新增5项测试覆盖返回数组修改隔离及四配置×六场景精确IQ/metadata一致；新增测试总数36，旧相关测试25。

## 场景选择建议（不启动）

建议暂定两个星地训练环境：`practical_mid`与`practical_low_urban`，clean继续保留。依据已有四组探索性测试：high与clean接近，mid约77.38%–78.48%，low_urban约52.90%–53.86%，两者差距明显。各row的checkpoint epoch不同，不能据此比较方法强弱；此处仅描述每row内部场景难度。证据为原r03报告2026-09-20已保存评分，不是本轮最新现场评分。

新增mid_urban没有同一CVS checkpoint下的历史实测，不能宣称它比mid更合适；暂作独立评估候选，不强行作为第三训练环境。若未来仍按原采样机制每步只生成固定数量视图，缩小场景池不减少训练前向次数；六场景改两场景会减少评估枚举量和缓存占用。减少DAOT视图数是另一项科学改动，本轮不做。

原报告：`automation_reports/CV-SincNet/20260918-phase1-daot-rc4-practical4-manysig-s392005-r03/report.md`；禁止将已曝光的目标场景选择包装成独立确认性结论。

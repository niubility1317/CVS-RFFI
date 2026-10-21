# R65 prune_after10 的推理时延与综合优化分析

日期：2026-10-21。对象为用户指定的 `20261021-phase1-restoration-R46-delayed-prune-s2026092701-m2-r65/prune_after10`，原方法 commit 为 `4fd058f9bae79687f924e06d057e6a20083ff6dc`。

**分析结论：先优化执行路径，再考虑继续裁剪。** R65 已把原 R17 的参数减少 94.13%、Conv/Linear MAC 减少 91.30%，但同卡单样本 G+F 时延只减少 4.64%，批量 32 的 G+F 反而增加 24.25%。当前证据支持优先处理小算子调度、前向内部同步、重复物理算子和数据搬运。继续缩小通道不能代替实测提速。

本次完成原始记录复核、原权重结构统计、本机合成输入剖析及一个残差代数等式检查。未修改模型实现或权重，未训练、读取实际 IQ/新目标数据、加载 F，未启动 N607 任务。下文新优化均为待实施方案；没有新版本的提速或识别率声明。

## 1. 已核实的基线与时延

R65 先在 M35 的 350521 参数图上训练 10 轮，再按训练前固定的通道索引裁剪，训练剩余 10 轮，合计 E20/4940。最终是静态的 273765 参数恢复器，推理不需要密集父图、教师、优化器或训练缓存。M35 在此指 R46 `prune_refine`；原 R17 `unroll_unshared` 仍是固定压缩对照。

以下为原 run 的 N607 GPU2 同卡测量：64 条 source V、FP32、3 轮交替顺序、预热后同步计时；G 为完整恢复器，F 为原 F50 默认头及内部 FFT/PA/时域/统计分支。

|指标|原 R17|M35|R65 prune_after10|
|---|---:|---:|---:|
|恢复器参数|4665073|350521|273765|
|Conv/Linear MAC 每条 IQ|353427584|39081004|30761512|
|G，B1，中位 ms|51.387|48.540|47.799|
|G+F，B1，中位 ms|61.362|59.630|58.513|
|G，B32，整批中位 ms|57.834|87.982|76.196|
|G+F，B32，整批中位 ms|70.459|99.480|87.544|

相对 M35，R65 的 B1 G/G+F 时延分别减少 1.53%/1.87%，B32 分别减少 13.40%/12.00%。但相对原 R17，B32 的 G/G+F 分别慢 31.75%/24.25%。后一个对照必须保留，不能只选择 M35 来叙述加速。

R65 B1 的 G/G+F P95 分别是 65.829/75.960 ms；B32 的整批 P95 分别是 100.084/103.197 ms。G+F B32 的实测吞吐为 360.09 条/s。87.544/32=2.736 ms 只是整批摊销值，不能写成单请求响应时延；实时凑批还包含等待时间。

上述计时预加载 GPU 输入，不包含传输、外部预处理、加载和排队。原报告同时驻留三个 G 和一个 F，因此 suite 驻留值与增量峰值都不能冒充独立部署总显存。MAC 只统计卷积和线性层，不包含 FFT、CG、归一化、激活、搬运或 F。

原正式预测采用 B128，168000 条 × 7 视图总耗时 713.766 s，约 1647.60 条/s；每视图 100.299 至 105.040 s。这一墙钟包含原预测循环的数据搬运和产物写入，属于不同口径。B32 的 360.09 条/s 与 B128 的 1647.60 条/s 不能直接相除宣布批量收益，测量时段与过程也不同。

证据：[原正式报告](../automation_reports/CV-SincNet/20261021-phase1-restoration-R46-delayed-prune-s2026092701-m2-r65/report.md)、[原结果](../automation_reports/CV-SincNet/20261021-phase1-restoration-R46-delayed-prune-s2026092701-m2-r65/results.json)；原预测完成记录保存在本地 collections 的 `rows.prune_after10.target_prediction_complete`。

## 2. 小模型仍执行了大量串行工作

完整推理顺序为：频域 coarse U-Net → 时域 coarse U-Net → action estimator U-Net → 三个依赖前一阶段输出的 prior U-Net，每个 prior 后执行两次 CG 更新。输入和输出都是 float32 `[B,2,256]`，最终同一份恢复 IQ 统一进入原 F。

|组件|独立参数|参数占比|Conv/Linear MAC 每条 IQ|MAC 占比|
|---|---:|---:|---:|---:|
|coarse 频域与时域|106138|38.77%|8068608|26.23%|
|action estimator|79477|29.03%|5612584|18.25%|
|三个 prior 合计|88150|32.20%|17080320|55.53%|
|合计|273765|100%|30761512|100%|

三个 prior 共享长卷积权重，仍然调用三次网络；输入中的 current 和 innovation 每阶段变化，不能共享前向结果。参数共享减少存储，没有删除这些计算。

原权重一次实际前向的动态计数如下，CPU 计数与 CUDA trace 中对应算子次数相符：

- 216 次 Conv1d、1 次 Linear。
- 168 次 GroupNorm、168 次 SiLU。
- 32 次 FFT、31 次 IFFT，合计 63 次；其中 coarse 为长度 256，物理算子主要为长度 512。
- 3 个 prior、每个 2 次 CG，共 6 次 CG 更新。
- 12 次张量转 Python 标量判断；CUDA trace 同时出现 12 次标量回传及 stream 同步。

源码位置：[最终结构](../experiments/pamf_r46_delayed_prune/models.py)、[context 与推理入口](../experiments/pamf_iq_joint_compact/models.py)、[三阶段恢复](../experiments/pamf_r17_compact/models.py)、[Action 与伴随算子](../experiments/pamf_action_restore/operators.py)、[weighted_map](../experiments/pamf_uncertainty_restore/models.py)。

## 3. 本机剖析说明了什么

本次仅使用原 R65 权重和合成输入，无数据集、标签或 F。环境为 Windows、RTX 5070 Ti、PyTorch 2.10.0+cu128、CPU 2 线程。每个普通计时用 6 次预热和 20 次重复；它与原 N607 Linux/PyTorch 2.1 环境不同，不能混为一条时间序列。

|本机原模型，G only|B1 中位 ms|B32 整批中位 ms|
|---|---:|---:|
|GPU，no_grad|82.346|75.201|
|GPU，inference_mode|73.537|69.189|
|CPU，inference_mode|25.884|未测|

顺序执行、重复次数较少且系统存在桌面负载，本表用于发现执行问题，不能作为稳定加速比例或 N607 部署预测。它说明必须保留小批量 CPU 对照；“放到 GPU 就更快”在这条执行路径上没有成立。`inference_mode` 值得纳入后续同卡交替测量，不能单凭本表宣称正式提速。

单次 B1 CUDA trace 记录 2037 个 GPU kernel，1954 个低于 5 μs，占 95.93%；kernel duration 相加约 3.140 ms。168 次 GroupNorm 在该后端分成约 504 个 kernel，卷积 bias、激活和残差计算也分别产生大量 kernel。CPU 侧同时记录 1972 次 `cudaLaunchKernel` 调用；GPU kernel 数另包含 driver 路径，二者口径不同。

这是大量短任务的执行图，支持优先减少调度与 Python 路径。**3.140 ms 不是可实现的整体时延下界**：它不包含调度、同步、拷贝等成本，profiler 本身还会扰动时间。带标注的 scope 区间也不能当作独占算子耗时相加。当前没有 N607 算子级 trace，因此尚不能给各瓶颈分配可靠的 N607 时延百分比。

本地原始证据：`local_artifacts/pamf_restoration/r65_latency_analysis_20261021_01/audit.json`、`original_B1_trace.json`。权重与原始 trace 不进入 Git。

## 4. 优先级最高的执行优化

|顺序|优化内容|依据与预期作用|需要保留的行为|
|---|---|---|---|
|1|推理计算核与检查分离，使用 inference_mode|前向存在 12 次 CPU/GPU 往返，阻碍连续提交和图捕获|形状、输入合法性及异常输出检查仍在统一边界执行；中间异常可累计 device 状态，不能静默忽略|
|2|固定 shape 的 CUDA Graph；必要时先对六个 U-Net 分段编译|约 2000 个短 kernel，B1 对调度敏感|每次复制当前输入并重新生成当前 action；不缓存某条 IQ 的 action 或输出；保留尾批路径|
|3|复用当前残差，删除 CG 初始化中的重复 Action 调用|保持六次 CG，可把 FFT/IFFT 从 63 减到 39|只适用于已核实的窗内常量实方差；必须使用归一化前的原始残差|
|4|融合 GroupNorm、SiLU、残差缩放等逐元素工作|168 次 GN、168 次 SiLU，在本机产生大量短 kernel|保持原 GN4 分组、epsilon、统计维度与输出；不能按 BatchNorm 方式静态折叠|
|5|缓存固定索引和常量，复用单次输入内的共轭量/缓冲区|reflect 索引固定，action 的 response 与共轭在一次恢复内重复使用|跨输入仍更新 learned action；图重放不得读旧输入；并发请求不能共用可变 scratch|
|6|统一预测入口，减少逐批 Python 和 CPU list 搬运|正式预测直接调用 training_outputs，输出使用 cpu().tolist() 桥接|优化须同时覆盖正式预测实际入口，保留原 truth-last 产物和同一恢复 IQ 接原 F|

CUDA Graph 适合减少 CPU 调度，但禁止捕获内部 CPU/GPU 同步并要求稳定地址与形状；部分不安全区域可以保留 eager。不能在当前带数据相关 Python 判断的完整图上只加一个开关便假定成功。依据：[PyTorch 2.1 CUDA Graph 说明](https://docs.pytorch.org/docs/2.1/notes/cuda.html#cuda-graphs)。

GN 在训练和 eval 时均根据当前输入计算统计量，不能预先折成固定卷积；可以研究保持计算语义的融合 kernel。依据：[PyTorch 2.1 GroupNorm](https://docs.pytorch.org/docs/2.1/generated/torch.nn.GroupNorm.html)。

**实际入口是关键。** `pamf_uncertainty_restore/evaluation.py:91` 调用的是 `model.training_outputs(r)`；原 profile 则调用 `model(x)`。如果只重写 forward，profile 会显示提速，正式预测仍可能走旧实现。应建立两者共用的推理计算路径，并检查返回的 restored、variance 和辅助量契约。

F 的默认头、原冻结状态及 FFT/PA/时域/统计分支继续保持。F 内的 FFT 来自恢复后 IQ，与 coarse 的输入 FFT 不是同一信号，不能跨这两处直接复用。F 的 Sinc 已有冻结滤波器缓存，不能把再次“冻结 F”当作新增优化。

## 5. 重复 CG 初始化可以怎样消除

记当前阶段输入为 c，prior 输出为 p，观测为 y，当前 Action 为 A，prior 方差为 τ，噪声权重 W=1/v。R65 的 v 在每个 256 点窗口内为同一个实标量，因此 W 与该窗口的实线性 A 及其伴随可以交换。

阶段开始已经计算未归一化 innovation：

```text
g = A*(y - A c)
```

原 weighted_map 的 warm-start 初始化再次计算：

```text
δ0 = c - p
b  = A* W(y - A p)
r0 = b - A* W A δ0 - δ0/τ
```

利用实线性可改写为：

```text
r0 = W g + (p - c)/τ
```

这里 `A*` 表示真实的实内积伴随，包含 image 共轭分支。推导没有把它替换为普通复线性算子，也没有删除反射边界。

每阶段省去两个 A/A* 配对，即 8 次 FFT/IFFT；三个阶段合计省去 24 次，从 63 降为 39，FFT 调用数减少 38.10%。三 prior、全部原权重和六次 CG 更新均保留。这个比例只针对 FFT 调用次数，不是整体提速预测。最后一次 CG 后不再使用的 residual/direction 更新也可单独删除，进一步减少标量与逐元素工作。

本次在原权重产生的 context 上检查 16 个合成窗口，幅度为 0、0.01、0.1、1，共 12 组阶段检查。单阶段复数输出最大绝对差约 1.284×10⁻⁶，均有限；零/低幅输入的相对差可达约 1.32×10⁻⁵。检查每阶段沿用原 current，尚未验证三个替换阶段串接后的累计差，也没有连接 F。代数等价不代表 FP32 逐位相同，更不代表识别率必然相同。

本地证据：`local_artifacts/pamf_restoration/r65_latency_analysis_20261021_01/algebra_check.json`。该检查没有保存新模型、测量新实现速度或访问真实 IQ。

## 6. 精度、显存、存储与完整链路

### 混合精度与量化

优先完成 FP32 执行优化，再对卷积子图尝试硬件支持的低精度；FFT、复数物理算子、噪声方差、CG 内积与除法先保留 FP32/complex64。原方差下限可到 1e-8，CG 使用 1e-20 防除零；直接全图 half 会改变这些数值行为。卷积低精度带来的转换、GN 和非卷积分支开销必须计入 G+F，不能只报告单层卷积。

动态量化若只覆盖 global_head 的 5289 个参数，只涉及完整恢复器的约 1.93%，无法解决 216 次卷积和大量 GN/FFT 调度。全卷积 INT8、TensorRT 或替换归一化需要新的适配与数值验证，当前证据不支持将它们作为首个动作。

通道 12/28 等形状是否导致后端选择低效 kernel，应在实际硬件上测量。原同卡数据中小模型 B32 比大模型慢、workspace 差异很大，提示后端和 shape 值得排查，但尚未证明具体 cuDNN 算法是原因。不能为对齐 8/16 随意补零通道后直接送入 GN，否则均值和方差会变化。

### 显存与存储

独立 FP32 参数占 1095060 字节，最终无 optimizer checkpoint 为 1388956 字节，约 1.325 MiB；剩余空间包含序列化结构、元数据和 buffer，不能全部算成可删除权重。删除 optimizer 已经完成，不是下一轮结构压缩收益。

原 R65 B32 的增量 allocated 峰值约 217.14 MiB，属于原多模型 suite 中一次前向的 allocator 指标。本机单 G 的峰值与它不同，说明环境、后端和口径必须明确。后续独立进程应同时报告参数、allocated/reserved、进程 VRAM、FFT workspace、图内存池、输入输出与 F；CUDA Graph 减时延可能增加驻留内存。

启动时延另计：现有 load_checkpoint 会验证元数据并构造密集/压缩 skeleton，依赖导入链也较长。若用于频繁冷启动，可研究独立的严格加载与部署封装，保留来源信息和权重别名检查；这不会降低常驻模型的单次前向时延。

### 数据搬运与产物

原 `array_tensor` 使用 contiguous → tobytes → bytearray → frombuffer → device，输出使用 tensor.detach().cpu().tolist() → NumPy。其原因是 N607 原 Torch 2.1/NumPy 2 桥接不兼容；不能直接换成 numpy() 并假定可用，也不能修改其他 owner 的共享环境。可在隔离环境验证兼容的二进制批量输出，再评估 pinned buffer、异步拷贝与写入重叠。

正式每条保存 IQ512、feature160、logit6 和方差1个 FP32 值；168000×7 共约 3.194 GB，另有 ID/头信息。正式证据保留要求继续遵守，服务模式可以另设仅返回识别结果的接口。两种输出模式必须分别计时，不能通过少写必需产物来虚报正式预测加速。

## 7. 识别性能与后续结构优化边界

原正式数据是同一 168000 个物理样本、冻结 F50 seed2026092701/defaulthead、clean 与六种 practical，全部预测先于独立 truth-last。以下只是复述已固定历史结果，不用于本次选择新宽度、CG 步数、损失或 checkpoint。

|场景|R65 准确率|较 M35，pp|较原 R17，pp|
|---|---:|---:|---:|
|clean|84.908929%|+0.452381|+0.322024|
|practical_high|82.825595%|+0.373214|+1.506548|
|practical_mid|81.620238%|+0.398214|+2.042262|
|practical_low_suburban|80.155952%|+0.535714|+2.159524|
|practical_high_urban|81.743452%|+0.356548|+1.389881|
|practical_mid_urban|75.063095%|+0.616071|+1.377976|
|practical_low_urban|68.024405%|+0.514286|+0.807738|

low urban 距 70% 仍差 3319 条正确预测；mid urban 距 raw clean 84.676190% 的差为 9.613095 pp，超过 5 pp，距该目标尚差 7750 条正确预测。相对原 R17 逐景最多下降 1 pp 的容忍约束已满足，但这不是全部目标完成。

低 urban 的 TX1 为 51.221429%，RX `20-1` 为 64.158333%，day `2021_03_15` 为 66.095238%，分组差异仍大。clean 相对 raw clean 修复 809 条、损伤 418 条，净增 391；low urban 相对同景 raw 修复 73447 条、损伤 10069 条，净增 63378。净修复须和损伤一起报告，也不能把修复数量写成相对 M35 的收益。

先保留现有函数和识别行为，解决执行浪费，再考虑减少 prior 次数、CG 步数、U-Net 层数或更换轻量卷积。这些结构变化会改变恢复映射，需要独立消融；权重共享、低秩分解或更小通道本身都不能证明时延下降。历史已暴露结果和本次用户指定 R65 的工程延续背景必须保留，不声明新的独立盲测。

后续训练继续使用合法 L_s/U_s IQ、模拟配对及合规 IQ 教师，TX 无标签、F-free；F 只用于冻结后的评测和 no-query smoke。不得按上述 target 分组指定损失或另训头。每个新版本仍需固定后完整七景正式测试，无源收益也不取消；测试结果不回流调参或选择性重跑。

训练成本也需保留：R65 的 dense/compact 阶段为 810.412/760.030 s，完整墙钟 1571.545 s，训练峰值 allocated 1423320064 字节，教师/student 各曝光 1260000 条。密集阶段与教师开销不会因部署文件变小而消失。未来固定教师的源 IQ 输出缓存可单独评估计算与存储权衡，不能隐去教师生成成本。

## 8. 下一阶段的最小实施顺序

1. 以原 R65 为用户指定优化对象，保留原 R17 与 M35 对照；先建立实际推理入口共用的 FP32 fast path，整理检查边界、固定索引与 inference_mode。参数、三 prior 和六次 CG 保持。
2. 分开检验图执行与残差复用，避免把多个修改混成无法归因的一行。图捕获先解决内部同步，残差复用先检查串接累计误差、低幅/零值/异常输入及真实伴随边界。
3. 仅用合成或合法源 IQ 测量 B1/8/32/128 的 G、F、G+F；独立进程、同卡交替次序，分列冷启动、设备预加载、含传输和完整产物的墙钟、median/P90/P95、吞吐与内存。实时批量另计凑批等待，不以摊销时间替代响应时延。
4. 冻结候选后按原最小流程登记并正式测试每个新版；原 168000×7、原 F/defaulthead、先预测后 truth。逐景报告相对 R65 和原 R17 的正确数差、clean 损伤、TX/RX/day、净修复以及原目标差距。

CUDA Graph 的设备地址/shape 检查、不同输入交替重放与尾批检查，以及代数复用的数值检查，只服务于这次真实执行路径，不增加审批、哈希链或报告审查。当前健康实验、既有 owner、历史产物和正式预测均保持原状。

当前可交付结论是：R65 的压缩已显著降低权重和卷积 MAC，下一步应优先去掉执行浪费；实际提速幅度、独立部署显存和新版本识别结果仍需按上述相同口径测得。

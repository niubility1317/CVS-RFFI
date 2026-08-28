# SF-TAPFT P1紧凑部署与性能矩阵设计

## 1.目标

本设计承接提交`910a1c8bdce7c5349be4ef2acd995f7b53617981`及用户提供的复盘报告。P0C已经闭合H6原位适配、FP32 prefix cache、delta v2和Q180 truth-last评分。本轮在不扩大Phase2数据权限的前提下完成两类工作：

1.修正资源测量并把H6部署路径收敛为独立紧凑suffix训练引擎；
2.在一个新的、未暴露query truth的合法capsule上比较D0–D4，寻找高于H6的性能工作点。

## 2.协议与不变量

- 使用`protocol_schema=p2_min_v1`和匹配的`VALIDATED_ONCE/capsule_id/split_id`。
- support为6个旧类、每类K=10，共60条互异物理样本；本轮不注册新类。
- 适配、精度复核、温度拟合和候选选择均不得读取query、query truth、query role或source/clean样本。
- 报告只使用`DA0_REG0`和`DA1_REG0`；新类指标为`N/A`。
- 所有候选使用原位训练、delta v2 only、无Adapter、无完整t3、无frequency/domain更新、无EMA、无HardPair。
- D0–D4必须共享同一checkpoint、support、query、receiver、scene、seed和scorer。
- 每条query独立面对全部6个注册类；禁止配额、全局重排或跨query状态更新。

## 3.工程底座

### 3.1资源测量

Linux当前RSS从`/proc/self/status`的`VmRSS`读取；`resource.getrusage(...).ru_maxrss`保留但重命名为`process_lifetime_maxrss_bytes`。资源结果同时输出：

- 当前RSS起点、采样峰值和适配附加峰值；
- 进程生命周期最大RSS；
- CUDA allocated/reserved峰值；
- `torch.cuda.mem_get_info()`的适配前后free/total及最小free；
- `cold_start`和`resident_process`两种模式。

冷启动模式每个测量重复使用独立子进程，包含解释器、模型加载和CUDA context。常驻模式复用已加载模型，只测cache构建、适配和delta导出的附加成本。

### 3.2cache存储与计算精度分离

将单一`prefix_cache_dtype`迁移为：

```text
cache_storage_dtype
suffix_compute_dtype
cache_device
```

历史`prefix_cache_dtype`继续兼容。训练开始前只执行一次storage→compute materialize，后续步骤复用计算cache，禁止每步重复FP16→FP32转换。首轮正式候选使用：

- D0/D1/D2/D3/D4：FP32 storage、FP32 compute、CUDA cache；
- 工程对照E1：FP16 storage、一次性FP32 compute；
- 工程对照E2：BF16 storage/compute，仅在硬件和等价测试通过时运行。

FP16/BF16路径必须保留一次FP32 full-path support安全复核；失败后回滚许可参数并以FP32重训，不读取query。

### 3.3独立CompactH6Suffix

`CompactH6Suffix`只复制H6后缀实际需要的冻结子模块，并持有可训练`t3.norm`与target head。它不得持有完整model引用。执行链为：

1.常驻推理模型执行一次`encode_h6_prefix`；
2.从完整模型构建`CompactH6Suffix`；
3.在紧凑对象上完成H6/Q2/R1训练；
4.导出许可模型参数和target head；
5.原子应用回常驻模型；
6.失败时恢复训练前许可参数锚点。

参考路径和Compact路径必须在固定seed下满足support argmax、逐类recall、margin符号一致，并记录最大logit差和Norm梯度最大差。

### 3.4delta原子性

delta v2先写入同目录临时文件，完成格式自检后使用原子替换写入最终路径。加载或应用失败时不得留下部分适配状态，必须恢复许可参数锚点和target head锚点。

### 3.5编译加速边界

CUDA Graph、`torch.compile`/AOT和预计算Norm标准化只作为独立工程候选，不修改D0–D4科学行。只有通过logit、梯度、support预测和margin等价测试，且实际墙钟改善后才能进入默认部署路径。冻结suffix eval同样必须作为单独候选，不能无条件替换历史H6训练语义。

## 4.P1性能候选

|行|方法|唯一变量|训练上限|
|---|---|---|---:|
|D0|P0C H6|新capsule基线|520步|
|D1|Q2A-Deploy|`t3 weight/bias+t2 weight`|不超过历史选中步数503|
|D2|Q2B-Deploy|D1再加`t1 weight+time_fuse weight`|不超过历史选中步数231|
|D3|R1-T|R1训练不变，增加support-only OOF温度|不超过1584个可训练元素|
|D4|H6+head-only class-CVaR|H6后追加30步head-only CVaR|550步|

D4在H6结束后冻结backbone和`t3.norm`，缓存160维embedding，仅更新960个target head参数。每类support损失为：

\[
L_c=\frac{1}{n_c}\sum_{i:y_i=c}\ell_i
\]

尾部损失和最终目标为：

\[
\mathcal L_{\mathrm{CVaR}}=\operatorname{Mean}(\operatorname{Top2}\{L_1,\ldots,L_C\})
\]

\[
\mathcal L_{\mathrm{head}}=\mathcal L_{\mathrm{CE}}+0.03\mathcal L_{\mathrm{CVaR}}+\lambda_a\mathcal L_{\mathrm{anchor}}
\]

首轮固定30步、`lambda_t=0.03`，不搜索类别ID专属权重。

D3温度只能由support OOF logits拟合。温度不得改变argmax、BA、floor或逐类准确率，只用于检验R1的NLL是否主要来自概率尺度。

## 5.实验与晋级规则

相对同capsule D0，候选必须同时满足：

\[
BA_{new}\ge BA_{D0},\quad floor_{new}\ge floor_{D0}
\]

\[
\min_c(Acc_{c,new}-Acc_{c,D0})\ge-5\text{pp}
\]

\[
NLL_{new}\le NLL_{D0}+0.02
\]

资源约束：可训练元素不超过1584、delta不超过10KB、适配时间不超过20秒、query推理不增加额外分支。

早期只运行这个单seed五行最小矩阵。全部prediction闭合后由独立scorer一次连接truth；低性能不触发技术停止，也不选择性重跑。

## 6.验证

- 配置负测覆盖非法dtype、非法CVaR参数、HardPair非零和超过资源上限。
- 单元测试覆盖当前RSS、lifetime max RSS、一次性materialize、Compact无完整model引用、参考/Compact等价、原子delta回滚、CVaR Top2和温度argmax保持。
- 一次真实CORE90 checkpoint、60条support、无query smoke。
- N607发布只使用一个release归档并比较一次本地/远端SHA，远端编译一次。
- 启动后检查PID/CWD/cmdline/run-root/GPU/log增长；prediction闭合后truth-last评分。

## 7.明确不做

- 不继续搜索HardPair权重。
- 不在本轮提前扩展多receiver、low-elevation、rain、多seed、K=5、K=2或新类注册。
- 不把CUDA Graph/AOT可行性当作D0–D4发布门。
- 不重新验证未变化的`VALIDATED_ONCE`数据。
- 不增加报告SHA、成员SHA、签名、receipt链或其他白名单外gate；相关要求标记为`REJECTED_EXTRA_GATE`。


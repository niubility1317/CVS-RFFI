# F＋跨 RX：冻结默认头的 ordinary／As 梯度关系

**VERIFIED／ANALYZED。预定共同支持条件未通过：RATIONALE_REJECTED。** 本轮没有训练新模型、调整参数或访问目标数据。固定基线仍为 F＋跨 RX 同 TX 对比学习、真实 joint160、默认 cosine30 分类头正常联合训练；新增训练预算仍为 E20。本轮只读取既有匹配 C0/C1 的 final20。

本轮问题是：完整 As 的梯度是否比同 IQ 重复 ordinary 控制更经常反向于原监督，并在两个 checkpoint、两条观测流中共同支配原监督。此处“clean 参考”实际是现有普通增强后的 IQ，不是未增强 raw。原 PA、循环移位和全部普通流程保持；没有新增一致性、分类器或部署分支。

| 冻结模型／观测流 | As／primary 范数比中位数 | As 负内积批比例 | 重复 ordinary 负内积批比例 | 原条件 |
|---|---:|---:|---:|---|
|C0/r0|4.944537|42.5%|0.0%|通过|
|C0/r1|4.827113|32.5%|0.0%|通过|
|C1/r0|1.426489|12.5%|0.0%|未通过|
|C1/r1|1.352292|17.5%|0.0%|未通过|

固定条件要求四组全部满足范数比中位数 >1，且 As 负内积比例比匹配控制至少多 20 个百分点。C1 两条流分别仅多 12.5／17.5 个百分点，因此共同条件失败。C0 有 10%／7.5% 批的 p+s 对 primary 欧氏方向反转，C1 均为 0；此为一阶几何，不是实际 AdamW 更新或损失变化。

这支持未做 As 训练的 C0 面对该源视图时梯度较强；没有支持“两个最终模型都持续受 As 梯度支配”的固定理由。C0、C1 是两次独立 E20 的最终模型，不是同一个模型的时间轨迹，不能由此推断早期冲突或何时开启 As。本轮只用 RX1-held 契约的 fit L，且该折既有 C1 held clean 原本由 83.444444% 升至 86.166667%；没有检验失败 RX6 的模型，不能把结果外推为 clean 退化的唯一原因。按原规则关闭该共同支配解释，不调层位、RX、观测流或阈值救援，不据此加入梯度投影或调权训练。

全模型平方范数和内积严格等于默认头、SIG_TIME、Sinc、时域路径、其余参数五组之和。Sinc q 梯度除 25000 换回原物理 Hz 坐标，其他参数保持原坐标；同组范数比和全模型几何均保留，不从某个阳性分组挑结果。实际模型为 900722 参数、GroupNorm／LayerNorm，0 BatchNorm、0 运行均值和方差；BN 状态污染这一实现解释已排除。

全部 5040 个合法 fit L 包来自 RX3/4/6/8 × day1/2/3 × 六 TX，共 72 单元、每单元 70 包。标签无关 permutation 分成 40 批，末批 48 包保留，两条流均覆盖全部样本。原 source cache 会反序列化未使用的源 V 和旧父特征，但不消费其数值；不使用 U 标签、V 前向或梯度、heldRX1 IQ、target。旧父模型特征没有进入本次计算。

两份模型的来源已核实为相同物理 L/U/V、scratch final20、无未知祖先或 target 选模；完整 strict state 加载。worker 在读取 L 数值前完成两真实 checkpoint 各 2 包 zero IQ 的 no-query smoke。模型参数和所有 buffer 前后逐 tensor 相等，.grad 始终为 None，无 optimizer／EMA／头拟合。

每批只生成一份原 ordinary、一次完整 As(raw)，两模型复用同一实际 IQ 和 physics。As 采用原 low_suburban、mid_urban、low_urban nominal 先验，几何、传播、噪声、接收质量和补偿依赖不变，无状态拒绝、置信度门控或样本删除。RX/day/eq 硬件 session 稳定且不由 TX 定义。末期 ordinary 按真实 E13 截断 configure，RDrop 两次前向复用同 IQ。原 SupCon 只作用于 primary 的浅／中层。

p=.5·dRD_ordinary+.1·dSupCon；s=.5·dRD_secondary。RD 使用原 alpha=1、label smoothing=.01。secondary ordinary 与 As 使用相同命名 dropout 流，两个模型使用相同命名观测；观测流与原训练／源验证／正式测试随机流分离。

| 模型／流 | primary ordinary 准确率 | repeated ordinary 准确率 | As 准确率 | As Macro-F1 |
|---|---:|---:|---:|---:|
|C0/r0|91.666667%|91.250000%|49.067460%|47.337842%|
|C0/r1|92.757937%|92.757937%|49.226190%|47.435075%|
|C1/r0|89.623016%|89.365079%|76.230159%|76.105474%|
|C1/r1|90.416667%|90.634921%|77.123016%|77.064373%|

上表只是已见 fit L 上开启原随机子网络时、同 IQ 两次 logits 均值的描述性默认头读出。As 列混合三个训练场景，不能写成 low urban、held clean 或正式测试成绩；也不是本轮新训练收益。两条观测流不是两个模型 seed，不能据此声明跨 seed 重现。全部 RX／TX／day／scene／state／lock 分层见 source_strata.csv；新增实验为零更新，没有新的训练曲线或性能提升。

| 成本 | 新训练模型轮 | 更新 | 网络包 | 墙钟 |
|---|---:|---:|---:|---:|
| 本轮冻结诊断 | 0 | 0 | 120964 | 59.252369 s（worker 主体） |
| 既有 C0 | 20 | 4440 | 3579018 | 训练及源 V 1123.572077 s，六源 practical 64.008927 s |
| 既有 C1 | 20 | 4440 | 3579018 | 训练及源 V 2046.192236 s，六源 practical 63.471685 s |

本轮网络包=两模型 × 两流 × 5040 × 六前向+4 包真实 CP smoke；60480 条默认头决策、1920 条梯度组记录、80 条实际观察批、完整 160 行进度均独立复算。新增 ordinary 与完整 As 各 10080 包。59.25 s 含权重／缓存读取、输入构造、模拟、梯度和写出，不含 dispatcher 导入、排队、SCP、独立复算和绘图。父模型成本分别保留，不把复用写成免费训练。

验证：48 合成网络包／0 更新；10 协议配置负例、6 真实来源元数据负例、4 标量异常及解析公式 oracle、跨平台路径与 tuple／JSON 等价检查通过。独立 reader 的 5 标量异常、14 元数据负例，以及全部实际数据的物理批配对、72 单元覆盖、全／分组内积守恒、完整预测和日志复算通过。一次限定独立 P0/P1 审查发现并定向确认修复两项：远端 Path 的 POSIX 序列化、Config tuple 与持久 JSON list 的等价比较；未增加重复审查。

发布使用普通账户、一次成功 preflight、一个不可覆盖 release／run root、单 release 归档本远端比较和 remote compile。初次本地 preflight 在 SSH 前因子进程缺失标准环境失败，修复仅限子进程环境；没有远端重复提交。唯一 dispatcher／worker 的 PID/CWD/argv/GPU 和 0→14426 字节日志增长独立核实，进程自然退出、0 failure／重启／热改，未干预其他 owner。

实际执行代码为 54e152dd8617ea1787f4ab0c95da86233f70ddce；离线读取和绘图为 0da384ef59d43c07666d3aaf6a1f734038b967dd。后续报告提交不冒充执行代码。精确命令、环境/CWD、来源、路径、GPU、stop 和 artifacts 见 experiment.json 与原 preregistration.md；运行接口为该 release 的 experiments/cvs_f_gradient_interference/runner.py。

完整 practical 的实际 __file__、全部 Config 保存在私有 observation_audit.json；调用来自已核实的 adv3b02_xuc 后端，25 MHz／256 IQ／residual／post_sync／EQoff 即 G=1／reference_snr_db=30／reflect，逐项与原 C1 Config 相等。新增噪声 SNR 不冒称总 SNR，没有调整正式测试 IQ、SNR、信道或困难样本。

当前没有通过源规则的独立新 Ag；真正 B00/B10/B01/B11 及交互项、low urban >60%、suburban 接近自身 clean、正式七视图 truth-last 和跨模型 seed 重现仍未完成。既有另一个 owner 的 additive As 三行矩阵已有两个源对照结果，As 行尚无完整比较；本任务不重复启动、不合并不同前向／heldL 契约的绝对分数，也不修改其健康运行。

实际产物（全部已生成并查看）：
- [gradient_interference.png](E:/type10-7/local_artifacts/cvs_FcrossRX_gradient_interference_20261016/final/figures/gradient_interference.png)
- [parameter_groups.png](E:/type10-7/local_artifacts/cvs_FcrossRX_gradient_interference_20261016/final/figures/parameter_groups.png)
- [source_fit_readout.png](E:/type10-7/local_artifacts/cvs_FcrossRX_gradient_interference_20261016/final/figures/source_fit_readout.png)
- [gradient_interference.pdf](E:/type10-7/local_artifacts/cvs_FcrossRX_gradient_interference_20261016/final/figures/gradient_interference.pdf)
- [parameter_groups.pdf](E:/type10-7/local_artifacts/cvs_FcrossRX_gradient_interference_20261016/final/figures/parameter_groups.pdf)
- [source_fit_readout.pdf](E:/type10-7/local_artifacts/cvs_FcrossRX_gradient_interference_20261016/final/figures/source_fit_readout.pdf)
- [analysis_summary.json](E:/type10-7/local_artifacts/cvs_FcrossRX_gradient_interference_20261016/final/analysis/analysis_summary.json)
- [gradient_pairs.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_gradient_interference_20261016/final/analysis/gradient_pairs.csv)
- [gradient_summary.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_gradient_interference_20261016/final/analysis/gradient_summary.csv)
- [source_decisions.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_gradient_interference_20261016/final/analysis/source_decisions.csv)
- [source_strata.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_gradient_interference_20261016/final/analysis/source_strata.csv)

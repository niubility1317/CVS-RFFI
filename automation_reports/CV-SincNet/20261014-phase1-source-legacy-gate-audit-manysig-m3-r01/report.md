# 合法源样本的旧门控覆盖与精度诊断

状态：ANALYZED，三个既有模型的源 L/V 前向和聚合诊断全部完成。完整固定矩阵见 [experiment.json](experiment.json)。

## 目的与边界

既有真实训练显示旧门控约接收 1.6% 的 U 包次。原生实现要求同一 RX/day/eq、sig 与 base_index 距离均 ≤2 的物理邻包同时出现在当前随机批次，并且预测一致；它不是跨批次或跨 epoch 的同一包稳定性检查。先区分置信度/强一致性与批次邻包可见性，而不是先调整算法或分类头。

只读取已冻结 OLD_GATE_COS50 的三个固定模型 seed（2026092701/02/03）及各自真实保存 EMA。原配方为 scratch50、33 label＋17 pseudo、11100 更新、旧门控＋EMA＋cos、既有 NoExplicitCFO。全部 checkpoint 用历史 source_proof 核验，来源为原 M12 中的 OLD_GATE_COS50。它们仅用于冻结源观察，不初始化任何新训练，不按目标分数挑选。

只使用合法 L_s=6300 和 V=27000 的既有 source_cache，source split392005、RX1/3/4/6/8、day1/2/3、旧六 TX 与物理 ID 保持。U IQ、U TX 真值与目标 IQ/标签/分数均不访问。缓存中旧 parent feature/margin 字段不消费。没有分类头拟合、原型、RX 分类器、梯度、优化器、校准或 EMA 更新。

## 固定计算与成本

每个源角色只采用一次固定随机排列。L/V shuffle seed 为 2026101410/20，strong-noise/dropout seed 为 2026101411/21，跨模型 seed 使用相同规则。批次 256，保留尾批。弱路径是实际自身 EMA eval 的 clean 默认 logits；强路径是冻结学生 identity 的 train-mode 既有 dropout 和原生 std0.015 高斯增强。身份路径无 BN，所有学生/教师参数和 buffers 前后逐项不变。跳过不参与门控的 nuisance 前向；合成检查证明 eval identity logits 与完整原生 U 路径一致，此诊断不复现整个优化步骤或原 U 的随机数轨迹。

门控计算直接调用原生函数：RX/day quantile0.86，tau 在 0.92 至 0.97 间截断；邻包 window2、minconf0.8；domain gate 关闭。分别观察 CONFIDENCE、CONFIDENCE_STRONG、FULL_OLD 和只考虑物理邻包是否在批次出现的 GEOMETRIC_NEIGHBOR。标签在所有 masks 固定后才用于正确率计数。按 overall/RX/day/源类报告 accepted、correct、coverage、precision、拒绝的正确包及 EMA weak accuracy。

使用原 WiSigIndex/opaque_id 和 source-base 索引顺序，仅为已合法标注的 L sig0..69、V sig700..999 恢复原 rx/day/eq/sig/base_index。既有数据直接复用；该步骤用于附上原门控所需元数据，不是重验 IQ 字节或新增 hash/签名/receipt gate。不枚举或解码任何 U 物理 ID，也不以标签作为门控输入字段。

全矩阵 3 个冻结模型，0 骨干/头/优化器更新。每 seed 33300 包 × teacher/strong 两次前向，共 66600 包次；三 seed 199800 包次、786 次批前向。CPU-only，一个串行进程，Torch/OMP/MKL/OpenBLAS 线程 2，不占 GPU。历史每模型 50 轮训练成本仅记为复用。

额外给出 uniform random batching 的精确超几何邻包可见概率，依据 pool 大小、每物理 cell 数量、window2 和 B256。L/V 的池大小不同于 U，源代理覆盖不能直接当 U 覆盖。U56700/cell630 仅作为已知协议的几何常数，不使用 U 标签、预测或 IQ，也不能推导 U 伪标签正确率。解析概率已经由独立组合数公式合成核验，但真实代理结果尚待运行。

## 验证、发布与失败规则

合成检查 PASS：原生物理 metadata/索引对应、拒绝 U 角色或隐藏标签、门控拒绝 TX 标签字段、门控正例、默认 eval identity 与完整 U logits 一致、strong identity 无参数/buffer 更新、配置禁止训练/目标/新头。唯一独立 P0/P1 审查 PASS，覆盖最终代码、源缓存生产边界、解析概率及 CPU 发布入口。不新增报告审查或封印。

执行代码唯一身份为发布归档实际 Git HEAD，写入 release_commit.txt；提交和 push 后独立比较远端 OID。精确命令：`python -u -m experiments.cvs_source_gate_audit.audit`；CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_source_gate_audit_20261014_r01`；普通 N607 用户及既有 CVS-RFFI Python，full FP32，CUDA_VISIBLE_DEVICES 为空。输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261014-phase1-source-legacy-gate-audit-manysig-m3-r01`，每 row 独占，既有输出不可覆盖。

一次归档 SHA 本地/远端比较、remote compile、真实已有默认 checkpoint 零 IQ no-query smoke。PASS 后直接运行上述源诊断；不执行 target prediction/scorer。任何所属来源、元数据、非有限值、默认路径或状态变化错误停止此诊断并保存 failure.json 和已有结果，不自动重试或影响其他 owner。实际 PID/CWD/argv/CPU/log-growth 独立回读。

此分析与正在运行的 Sinc scratch50 矩阵独立，不改变其配方、模型来源、源放行阈值或测试流程。源代理证据如支持新 SSL 因素，只能在之后另行固定公平实验；当前不热改健康任务。结果只支持源机制判断，不宣称目标泛化、实际 U 准确率或新 TX 迁移。

## 实际完成

[完整同轮结果](source_gate_analysis.md)及[完整聚合计数](results.json)已经返回。执行代码 0f514fa1ba4302bc7e5e25d0ad41337881acce7f；0训练/分类头/优化器更新，360记录，CPU104.59秒；三row自然结束、失败0。source V confidence+strong覆盖92.2815%/precision99.8221%，完整旧门控覆盖3.4099%/本次接收子集错误0；不能外推U或目标性能。

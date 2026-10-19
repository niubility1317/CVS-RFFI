# F＋跨 RX 同 TX：弱发射机特征困难样本对照

状态：RUNNING。两个首种子已并行进入实际训练，尚无本轮目标测试结果。

基底为原始 F CROSS_RX50：三路独立 SIG_TIME 局部投影、实际 joint160、统一居中物理 Sinc、RDrop α=1、旧门控＋ownEMA＋cos。85.1139% 是历史方法锚点；本机逐种子比较复用同配置、同数据、同后端的 BASE-s2026092701/02/03，不把历史均值当作本轮配对值。

## 固定矩阵与预算

| 条件 | 新增特征约束 | 首种子 | 条件扩展 |
| --- | --- | --- | --- |
| MEAN_WEAK_RX | 弱 TX × 源 RX 单元内全部有效锚点均值 | 2026092701 | 2026092702/03 |
| TAIL_WEAK_RX | 同单元内损失最高的 ceil(n/2) 个锚点均值 | 2026092701 | 2026092702/03 |

每模型从零 50 轮，33L＋17U，11100 次实际主优化器更新。前 5 轮预热，随后 10 轮线性增加权重至 0.1；第 1111 至 11100 步共 9990 次计划约束。两条件同时完成首种子源训练后，按下述固定源域规则独立决定是否原样扩种子，再读取本轮目标汇总。源域科学指标不通过仍自动完成首种子 clean 测试，保留负结果。

## 新机制

只使用当前实际增强后的 L128 和原有两次 RDrop 前向，不增加 IQ 读取、模型前向、部署参数或实际优化器更新。固定照顾源 V 已确定的弱 TX 索引 1、3（14-7、20-19），不依据新测试结果改变名单。原 CE、原浅层/中层 SupCon、U 管线和分类头均保留。

在实际 time_down（48 通道）和 t2.pool（96 通道）提取时间均值与总体标准差；使用原始当前 L 批次的 detached 标准化（下限 1e-4）与样本 L2 归一化。正对为同 TX、不同 RX，负对为不同 TX、相同 RX；温度 0.2。单锚点风险为 logsumexp(P∪N)−mean(logits_P)−log(|P|)。同 TX 同 RX 及异 TX 异 RX 均不进入此新增目标。

先对同一实际样本的两次 RDrop 风险取均值，再按弱 TX × RX 单元选取。每个弱 TX 内各已观察 RX 等权，两个弱 TX 等权，两个层位等权。TAIL 选择 detached 风险最高的上半部分，反传保留真实特征梯度。强 TX 参与负样本。缺正/负对的锚点排除；缺任一弱 TX 返回有限图零并记录不合格。缺 RX 单元按已观察单元平均，完整报告覆盖率。

分类头使用已有默认头，正常随主目标联合训练；新增特征目标无直接头梯度。不拟合、重训或替换头。EMA、推理及导出保持原坐标和部署特征。Sinc 仍按 p=p0+25000q、Fs=25 MHz，物理梯度 clip5，物理 p 的 origin-aware 衰减和单位映射更新，无双重衰减。

## 输入权限与测试

复用既有 READY/EXACT_MATCH 源数据缓存及 VALIDATED_ONCE clean capsule；不重建划分。源 RX=1/3/4/6/8，day=1/2/3，固定 split_seed=392005，L6300/U56700/V27000。U 的 TX 真值隐藏，原普通增强、LS0.01、门控及 EMA 配方不变。无星地增强。

源 V 只做既有 no_grad 默认头健康检查及最终 actual joint160 描述性几何分析；不进入训练、不拟合分类器或用于预测的原型。out_RX 描述质心只排除同 RX，编码器曾见全部源 RX，不能称为未见 RX 的模型验证。训练只固定 final50；不按 V 或目标挑 checkpoint。

测试仅 clean，每 row 冻结后的默认 student 头预测 168000 个目标样本，全部预测完成后由独立 CPU truth-last 评分。自动报告准确率、Macro-F1、6 TX 召回率、最差 RX/day、同 seed 配对差、实际 seed 数、均值、样本标准差与正差数。源域选模/扩展不读取目标真值或分数。该基准已经暴露，不声称新的独立盲测、Phase2 或星地性能。

## 目标访问前固定的扩种子规则

分别对每个条件，与同 seed 原始 BASE 源证据比较：每个弱 TX 的 out_RX 余弦 margin 第 10 百分位至少提高 0.002，RXspread/interTXgap 不超过 BASE 的 1.05 倍，零特征数为 0；默认头总准确率下降不超过 0.2 个百分点、最差 RX 不超过 0.5 个百分点、任一 TX 不超过 1 个百分点。实际计划调用 9990，合格比例至少 99%，每层弱 TX × RX 单元覆盖至少 95%。

只通过的条件原样扩展 02/03；首种子不通过仍自动 clean 评分，不追加目标驱动的参数扫描、改 seed 或选择性重跑。已有匹配 BASE 只读复用，不重复训练。最终是否改善弱发射机须以目标测试事实回答，不能用源域诊断或总准确率掩盖单个弱 TX 退化。

## 本地执行与停止规则

代码身份：实际 code/config Git commit `689546e29d3b83b53337dba7a6045beb82d1c7df`，已推送并独立核对远端；不可覆盖发布包复制核对和编译通过。两个 worker 的实际 GPU、命令、CWD 及训练日志增长已核实，见 launch_verification.json。CWD：IMMUTABLE_LOCAL_RELEASE/cvs_F_weakTX_tailRisk50_local_20261020_r01。环境：已核实 native Windows Python3.10 / PyTorch2.10+cu128，FULL_FP32_NO_TF32，threads2/workers0。输出：LOCAL_RUN/20261020-phase1-FcrossRX-weakTX-tailRisk-e50-clean-local-m2-r01，新建不可覆盖。

启动命令：

```text
LOCAL_BASE_PYTHON -X utf8 -u -m experiments.cvs_f_weaktx_tailrisk50.driver --profile LOCAL_PROFILE --metadata LOCAL_SOURCE_NATIVE_GATE_METADATA_R02 --base LOCAL_RUN/20261020-phase1-FcrossRX-weakTX-tailRisk-e50-clean-local-m2-r01 --reference-base LOCAL_RUN/20261019-phase1-FcrossRX-TXrisk-geometry-e50-clean-local-m4-r01 --commit RELEASE_COMMIT --max-workers 3
```

物理 GPU0 全 owner 合计最多 3 个 source/predict worker，派发前空闲 RAM 至少 4.5 GiB；独立 CPU scorer 最多 1 个。保留其他健康任务。

Role/provenance/config/output conflicts, nonfinite loss/gradient, incomplete native budget/EMA/Sinc/feature scope or failed worker stop new dispatch; preserve outputs and other healthy workers; no automatic or performance-based retries

预期产物：每行初始化与原生参数、完整 50 轮/11100 步日志、Sinc/RDrop/SupCon 观察、新增 feature_tail_progress/steps/training_observation、source_frozen、源域扩种子决定与 frozen_matrix、clean prediction、独立 score、同 row 及配对汇总。原始 IQ、权重、日志和逐包预测留本地，不提交 Git；提交实现、配置和精简实验结果。

## 已完成验证

CPU 与 CUDA 检查 PASS：10 项协议/配置负例；独立标量风险值和梯度核对；物理样本两视图先平均再选取；有限无对梯度；实际 F 联合反传到 Sinc、浅层及中层；辅助默认 W 梯度为 None，主目标 W 梯度正常；原初始化、预热、EMA、导出保持。实际卷积路径只有原有 2 次前向，新增风险计算不消耗主 CPU/CUDA RNG。一个匹配 real-checkpoint no-query smoke 通过并丢弃，正式训练从零。

一次独立 P0/P1 审查 PASS，无直接阻断问题。具体证据见 checks.json、review.json。检查读取源/目标 IQ 数、头拟合和实际优化器更新均为 0。不能将公式、梯度或 smoke 验证写成性能结果。

REJECTED_EXTRA_GATE：通用技能建议的额外设计审批暂停和全项目测试不属于本项目最小白名单；已有用户目标授权此有界研究，采用聚焦负例、真实 checkpoint smoke 及一次独立 P0/P1 审查。

此设计借鉴[监督对比学习](https://arxiv.org/abs/2004.11362)和[困难样本挖掘](https://openaccess.thecvf.com/content_cvpr_2016/html/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.html)的思路，属于受控 RF 特征实验，不预设有效或声称完整复现。

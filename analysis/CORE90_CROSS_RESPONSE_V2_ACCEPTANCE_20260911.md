> 2026-09-11后续修复：本报告保留历史审查状态；R1—R4及phase时序问题的当前结论见[修复与独立复核报告](CORE90_CROSS_RESPONSE_V2_REPAIR_20260911.md)。科学待完成项仍保留。

# CORE90交叉响应V2实现与确认验收

> 2026-09-11再次审查勘误：下文为上一轮验收记录。新发现1项P1、2项P2，尤其V2三条件门缺少真实训练调用，不能仅归为“真实源参数待冻结”。当前结论及逐条状态以[再次审查报告](CORE90_CROSS_RESPONSE_V2_REAUDIT_20260911.md)为准；原测试证据保留。

日期：2026-09-11。对应[完整实现计划](CORE90_CROSS_RESPONSE_V2_IMPLEMENTATION_PLAN_20260911.md)。独立分支`codex/core90-cross-response-v2-20260911`从计划提交`22777b27738b502e44da67e7323c4e9542fdf599`开始；旧V1代码、日志、checkpoint和评分均保留。

**确认结论：22项均已提供实现或明确的条件接口，本轮实现与有界测试通过；不能宣布完整科学验收全部通过。V218的历史首次分叉根因仍未完成，真实源三条件收益与待冻结参数也未被合成测试替代。**没有启动正式E200、多seed训练、N607新实验或target再次评分。

## 1.主要验收证据

入口：[机器可读索引](core90_v2_acceptance_evidence_20260911/index.json)、[测试原文](core90_v2_acceptance_evidence_20260911/pytest.log)。相关套件174项通过、12项环境条件跳过；跳过的是默认不自动启动的真实入口测试，已由独立CUDA脚本覆盖下述分支，不代表正式矩阵全部训练过。

|验收|结果与边界|
|---|---|
|P0角色/反馈|36角色、候选独立进度、恢复和供体隔离通过；反馈排列反例均0.025，每周期一次flush；失败反馈不写历史；K2矩形8记录曝光|
|FP32真实入口|真实CORE90、合成IQ、2epoch共8step、每分支40事件；U1/head_only输入、logits、基础loss、未缩放梯度、模型、主AdamW、EMA、原型、伪标签及RNG逐值一致，atol=rtol=0|
|AMP真实入口|相同8step主轨迹一致；辅助NaN/Inf分别注入后仍一致；第8步两条主分支均发生相同AMP溢出与跳步，逐位置比较非有限掩码及符号|
|V2真实断点恢复|head_only连续执行与E1恢复后逐值一致：模型、EMA、主optimizer/scaler、独立辅助optimizer/scaler、原型/伪标签、RNG、角色与未刷新反馈；见[恢复证据](core90_v2_acceptance_evidence_20260911/actual_v2_resume.json)|
|V2分支入口|6TX/5源RX合成数据；U3、Ux_normalized、head_only各2epoch、4次成功更新、16有效块；U3/Ux训练loss每步执行，纯梯度审计1次|
|CUDA决策等价|FP32最大绝对误差：loss1.19e-7、参照2.38e-7、原logits梯度1.86e-9；FP16三者逐值一致，valid mask相同|
|条件路由|G/T/R/X能量恒等式、各参数组只接收指定目标一次、共享参数排除、可靠性detach、缺证据关闭与单模块扩层均通过定点测试|
|独立审查|一次P0/P1审查发现1项条件分支P1：cls_head证据可保留旧扩层参数；已修复作用域匹配与参数回退，独立读回及回归通过|

真实入口终态码7来自验收脚本明确省略heldout，终态及冻结评价文件均为`OMITTED_SYNTHETIC_TEST`，不是正式晋级证明。大体积精确张量保留在本地日志，不作为正式权重提交。

## 2.逐条实现追踪

“条件通过”指接口/负测通过而真实源参数或科学证据未具备；“部分验收”保留未完成要求。

|ID|实现|确认状态|证据与限制|
|---|---|---|---|
|V201|matrix/config/forward_context|实现通过|新U1/U3同角色；U1_mask_off为同块无响应mask对照；U0仅普通loader桥接，不伪称完整2×2因子|
|V202|replay_audit/gradient_audit|实现通过|共同可达参数、None/零、已加权范数及两种比值口径；AdamW实际更新另列|
|V203|source_eval/source_baselines|实现通过|真实拟合full/TX-only/RX-only，同fit票据/步数，披露容量/初始化；双向shuffle，head_only重复性明示|
|V204|source_baselines|实现通过|6TX固定2查询+两组2供体；5RX用1查询+两组2供体，明确不是2×2独立查询交互；不足N/A|
|V205|replay/source_update脚本|有界诊断通过|3批×2路由及独立源查询/三LEO；真实CORE90受控CE+orth，不冒充完整基础项或真实源收益|
|V206|roles/sampler/schema|实现通过|36角色、独立RNG、选择后推进、预览、resume；记录索引不变|
|V207|coverage/integration|实现通过|observed/query/directed分开；实际候选池可达集合及实际覆盖分别输出|
|V208|scheduler/integration|实现通过|全部collect后一次finish；稳定求和、失败/空step/重复finish/pending恢复|
|V209|scheduler|实现通过|完整方向键、原始样本数、收缩和回退；未观察不是实测零|
|V210|coverage|实现通过|K2矩形8曝光，定向任务16记录；累计/唯一/任务计数分开|
|V211|scheduler/sampler|条件通过|新物理证据、陈旧度、可靠性、归一、加项占比和基础探索；无真实源三条件证据不启用|
|V212|tensor_ops/integration|实现通过|逐记录归一后K均值；尺度不变、近零mask、角间隔、类内尾部及错误类对；独立归一化loss候选|
|V213|decision_calibration/config|条件通过|L_s独立匹配组噪声、单一源V关键类对；条件→类对→全局回退；不足NaN；未冻结参数拒绝启用|
|V214|decision/gradient_audit/JSONL|实现通过|有效率/正缺口双分母、支持量/分位数；TX/类对真实参数梯度，分类器单列，非margin导数冒充|
|V215|tensor_ops/training|条件通过|predictor完整MSE、身份T+alpha*wX*X、域R+beta*wX*X分组autograd；不叠加总loss重复贡献|
|V216|gate_v2/integration|条件通过|三条件、困难源组/LEO保护；当前真实源门未成立；最多一个time_fuse或freq_fuse，拓扑排除共享，扩层需新证据|
|V217|replay/source_update|诊断通过|同探针验证domain-only间接改变下一orth身份梯度；直接响应梯度0不等于前端轨迹无影响|
|V218|train_ssdg/integration/replay|部分验收|可观测性、独立事务与新短回放通过；历史首次分叉原因仍UNRESOLVED|
|V219|decision/benchmark|实现通过|旧参考保留、奇偶RX中点分位数、去重、无参照及loss/gradient等价；旧CUDA半精度兼容修复单列|
|V220|integration/JSONL|实现通过|默认20步纯诊断；训练loss每步，非有限特征强制审计；audited_steps与更新次数分开|
|V221|cache/integration|实现通过|物理ID/原始裁剪/统计配置/scale版本键；只缓存detach原始统计，标准化即时应用；U1/U3无响应统计|
|V222|matrix/V2预登记|实现通过|多模型seed配对、data_seed392005独立、显式独立确认边界；全部launch=False，不虚构已冻结模型seed清单|

## 3.覆盖与成本

[角色证据](core90_v2_acceptance_evidence_20260911/role_reachability.json)使用6TX×5RX×3day、每cell70条独立合成记录，共6300条。225个全可行候选的直接查询全集450；实际128候选池也可达450，定向任务可达3798。49步、196块后，**观测矩形450，直接响应查询158，定向任务379**。这不是实际WiSig新run成绩；真实runtime从自身候选池计算可达集合，不预设450。

[CUDA计时](core90_v2_acceptance_evidence_20260911/decision_cuda.json)为RTX5070Ti、固定128×6logits、warmup2/repeats5：FP32参考/向量化前向1427.24/24.50ms，前向反向1482.98/23.35ms；FP16前向1504.34/25.40ms，前向反向1465.71/26.26ms。分段为同源码副本边界同步walltime，含host dispatch，不是纯kernel时间，也不能外推E200总加速。旧CUDA FP16参考曾因quantile dtype失败，修复为显式FP32计算再回写原dtype；FP32旧行为不变，[失败摘要](core90_v2_acceptance_evidence_20260911/decision_cuda_initial_failure.json)保留。

6TX/5源RX合成入口head_only缓存256条，值8192字节、键编码44288字节，hit384、miss256，未计Python容器额外开销。6300×8×float32=201600字节仅是值数组预算。U3/Ux_normalized缓存值均0。每次V2源复用/必要性审计额外8次主干前向、112次模型记录曝光、24次统计记录、6次辅助fit前向、12次辅助评价前向；成本单列，训练次数未减少。

## 4.科学与历史边界

[源更新诊断](core90_v2_acceptance_evidence_20260911/source_update.json)为scratch真实CORE90的受控目标：六类CE+0.05×原covariance orth，实际response_backward和已有AdamW矩。明确省略domain CE、GRL、pseudo、EMA、Fishr、SAT一致性，不能冒充完整基础训练一步。三个配对批次全数报告，查询物理ID互斥，源评价不反传，审计状态不合回。

domain-only直接身份响应梯度0、当前CE及梯度逐值不变，但下一orth身份梯度差范数7.20e-5—8.32e-5。joint-tail clean CE变化B−A均值约+5.13e-5、标准误约5.82e-5，三LEO均值同样为正；不能宣称响应改善身份风险，不能据此开放P2。

[历史范围审计](V2_HISTORICAL_DIVERGENCE_SCOPE_20260911.md)完整解析U1/head_only各200条epoch日志及stdout。E1已存在基础loss/源val差异，但两行E1均49/49成功，首跳步均E4；两行全程AMP开启且全局clip关闭，旧代码已有fork_rng。没有逐step票据、增强张量、梯度或scaler状态。因此共享scaler风险、后续10对8次跳步、clip或“忘记seed”都不是已证明的E1根因。

本次旧V1合成FP32八步也能主轨迹精确一致。首次原始比较因预期辅助optimizer组报键差异，排除辅助状态后[主轨迹比较](core90_v2_acceptance_evidence_20260911/legacy_fp32_main_reanalysis.json)一致。V218历史根因保留未完成；新可观测性、隔离与短回放分别通过，不降低原验收标准。

## 5.交付

新配置：[phase1_core90_cross_response_v2.json](../code/configs/phase1_core90_cross_response_v2.json)。delta分位数/组数、关键类对规则、源门阈值、P2可靠性及反馈参数仍需源侧冻结，生成器不自动补所谓最优值。

验收入口：[回归套件](../code/scripts/verify_cross_response_v2_suite.py)、[真实入口轨迹](../code/scripts/verify_cross_response_v2_entrypoint.py)、[6TX/5源RX分支](../code/scripts/verify_cross_response_v2_branches.py)、[源更新/间接耦合](../code/scripts/verify_cross_response_v2_source_update.py)、[CUDA计时](../code/scripts/benchmark_cross_response_v2_decision.py)。Git纳入代码、测试、配置、JSON/日志摘要及报告；数据集、权重和约123MB逐参数诊断留本地。

代码提交`310429025f8d22d9bd684e397fb77b2afd85ae6f`已push至`origin/codex/core90-cross-response-v2-20260911`，主Agent独立`git ls-remote`读回与本地HEAD一致：VERIFIED。本报告及小体积证据在后续独立文档提交交付；未上传数据集、权重或大型诊断张量。

剩余科学项是历史首次分叉、真实源参数冻结/收益确认及后续正式多seed独立确认，本轮没有自动发起这些实验。

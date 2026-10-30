# 固定 GMM 的已知类竞争与未知背景比值

本次检验一个独立拒识候选，保留选定的 `predicted_ratio` 主基线与 `pooled_ratio` 固定对照。旧 OVA/logits/cosine 已检验过类间竞争；新问题限于冻结 signal-GMM 的显式类别密度竞争，不宣称首次使用 margin。完整 Phase3 目标保持 ACTIVE。

Run `20261031-phase3-class-competition-s2026092701-r01`，单模型 seed2026092701；实际代码、逐行六类 seed、命令、环境/CWD、路径和执行资源在本 run `experiment.json` 维护。唯一 launch owner 为 codex-phase3-class-competition-20261031，不覆盖旧 run。

## 固定方法

原 F0 Tri-SIG-TIME SSL E50/11100/default6、source-only signal r03/dc01a402、DURA v1.0/R69 E20/4940、class-density/4af06683 的 known90/OE32、normalizer 与两组六阈值保持。身份始终取原 F0 默认头预测 c，不以密度 argmax 改身份，不另训身份头。

令 ℓ_j 为原共享白化空间中第 j 个已知类 15 个 RX/day Gaussian 的等质量 log-density，ℓ_OE 为原 OE mixture。令 b_c=logmeanexp(ℓ_j:j≠c)。四列固定为 pooled_ratio、predicted_ratio、class_competition、hybrid_competition，唯一主为最后一列：

- 诊断 class_competition：ℓ_c−b_c，不包含 OE。
- 主 hybrid_competition：ℓ_c−logaddexp(ℓ_OE,b_c)+log(2)，分母为 0.5 OE＋0.5 其他已知类密度。温度1、权重0.5固定，不搜索。
- 原 pooled/predicted 两列、原默认 logits/closed、同 DURA F0 baseline、signal160/metadata 逐值复用。

新模型/密度/normalizer/统计参数拟合均0；不是零校准：源 V27000 按既有 known12 宏 CK90/原 F0 闭集−AK≤2pp 规则，为两新列各拟合六阈值。30个 RX×scene 交叉组另报。全部三条件共享各自源阈值，不用 NV/U/Q 真值、真实 K、配额或 query 统计。纯类间比值可能对远处未知过度自信；混合背景是否改善必须由完整测试回答。

## 数据与顺序

复用 source contract392005/VALIDATED_ONCE，原 L6300/U_s56700/OE6134 为合规祖先，本轮新读0。V27000 是额外已知校准标签（旧六类各4500）；source NV3150 为19-15/19-16/7-1各1050、只在预测冻结后独立评分。旧六14-10/14-7/20-15/20-19/6-15/8-20；源 RX1/3/4/6/8 对应1-19/18-2/19-2/2-19/3-19，日期2021-03-08/15/23。原 L/(L+U_s)10% 不包含额外 V 标签。

跨 RX 的 clean/star Q 为 RX24-1/index26、2021-03-15 的同864物理记录，旧六/未知三类各96，三条件均接冻结 DURA。星地采用原 G10 MIX01/600km/25MHz/2.45GHz residual/post_sync/noEQ、referenceSNR30dB/reflect、六分量原权重/郊区城区各0.5，receiver2027/scene2026103001/channel2026104001；一个记录一份固定 received，不重抽信道。clean 仅为冻结测试例外，不进入任何学习；source/NV/Q 的既有 P1 特征缓存复用，不增加物理数据。

顺序：实际 F0/G/r03 各两条 synthetic 无 query smoke PASS后继续→核原 source-only/cache身份→固定原 density 和新分数公式→V 两新六τ＋旧两六τ exact→全τ冻结→source NV、clean Q、star Q 全四列预测→独立 CUDA-empty CPU 连接三份 truth。源失败不取消已登记 Q；全部方法全报，不改主/参数/候选或选择性重跑。

## 验收、成本与执行

主报逐真 TX 宏 AK、RU、H；附 CK、同输入原 F0 闭集损失、AUROC/OSCR、逐 TX/RX/day/scene 和同 ID 净增失。科学 gate：source known12、clean RU≥95%/star RU≥90%、两条件 CK≥90%/闭集损≤2pp、每预登记主景所需 roster 完整，并且主 star AK/RU 均不低于原 predicted_ratio。缺类保留 N/A；high 缺19-15，不能据总体高分认证完整主景。H 无单独硬阈值。

历史 source NV/devQ 已曝光，单 RX/seed/source 日期重合，不称盲测；U384/正式 RX24-18 不访问。本轮无目标 DA、发现或匿名注册，AN/H_OW 为 N/A；通过局部拒识也不等于 Phase3 完成。新神经更新/真实 G/F0/signal 前向0，三 CP synthetic2 各另计；原 F0/r03/G 训练、GMM 和缓存成本继承。NN834665参数、原density364340B；四组六τ192B（新增96B），无 source/U 行 bank 部署。完整过程时间不是逐包 B1 延迟。

N607 单 producer，启动前剩余显存≥4096MiB、每GPU新增本任务≤2；独立 CPU scorer。不得停止/修改/重启其他 owner。一次 preflight、单 archive SHA 比较、远端 compile、PID/CWD/argv/GPU/log 读回；真实 checkpoint smoke 是首步且通过即继续。非法角色/祖先、非有限结果、输出碰撞、无法产生合法预测时保留产物，不 partial truth/覆盖重启；UNKNOWN 先核原 PID/root。执行证据分别报告 VERIFIED/FAILED/UNKNOWN，缺遥测不补实验重跑。

背景资料：密度比用于减轻生成模型背景统计混杂的动机见[原论文](https://arxiv.org/abs/1906.02845)；本次固定双背景公式是项目独立假设，不把该论文的结果当作本项目有效性证据。

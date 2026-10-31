# 原生默认头的源域能量排序

Run `20261031-phase3-logit-energy-s2026092701-r01`，唯一 owner `codex-phase3-logit-energy-20261031`。保留已选 PCD＋DURA 研究基线，检验直接训练默认头的开放分数，而非继续修改目标矩或查询阈值。新学生为纯源候选；无新增目标适应、发现或匿名注册，完整目标 ACTIVE。

依据 [Liu 等能量 OOD 原论文](https://proceedings.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html)使用原生六类 logits 的能量读出。已知分数定义 s=logsumexp(logits)，T=1；高分接收，对应原论文负能量。这里的 RX/day 内软排序、源保持项与 RFFI 网络是本项目变体，不声称原论文复现。旧未发布 native-energy 草稿没有实际实验；本 run 使用独立新模块和当前 fresh 数据，不消费其旧 RX24-1 配置。

## 两臂与共同初始化

UNIFORM 与 RXPAIR 各独立完整复制合规原 F0 三视图 Tri-SIG-TIME SSL E50/11100 的原生 reader/default6。完整 reader232993 参数，默认头960；只冻结 Sinc48 与全部 buffers，可训练232945。骨干和原默认头联合训练，不拼 r03 骨干与 F0 头、不另训或替换头；无需新 projection。真实合成 smoke 核对初始 logits 等于原 F0。主部署为学生232993＋DURA266951＝499944 神经参数，原构造 wrapper 外667729不训练/部署；原 F0/r03/PCD/G 与历史产物保持。

两臂每步共同目标均 CE_L＋KD_L：L 真实六类 CE、LS0；KD 为原 F0 同 P1 logits 的 T=2 KL×4，权1，复用合法旧 L 教师缓存。UNIFORM 再加 KL(uniform6∥softmax(OE))；RXPAIR 再加 mean softplus(s_OE−s_L)，只对公共 RX/day 相同的 L/OE 配对，权1。主能量不设经验 margin、额外温度或 query 校准。OE 仅 outside-six 成员监督，无真实 TX 类别损失或采样。

source392005/VALIDATED_ONCE 同 DURA 星地 P1：L6300、OE6134。源五 RX1/3/4/6/8 为1-19/18-2/19-2/2-19/3-19，日期2021-03-08/15/23，eq1；六旧 TX14-10/14-7/20-15/20-19/6-15/8-20。Us56700只保留 Phase1 祖先，本轮不读；目标 U 值/ID、Q、V/NV、私有真值训练读取0。旧 PCD 有 U576 祖先，只作为冻结比较，不作新学生初始化或教师。

每步相同 L192/OE64、同一次256行前向。L按六类×15公共RX/day各抽2条=180，再每类抽2条=12，恰每类32；OE按15域各抽4条=60，再随机公共域抽4条=64，保证每条 L 有同域 OE。固定旧源采样流 L2026103017/OE2026103018，两臂完整索引流逐步一致；合法重复曝光不算新增物理样本，不使用 OE 真类或目标配额。每臂固定2220步、合计4440；AdamW lr1e-4/wd1e-4/eps1e-8，normclip5，FP32/TF32off、eval保留梯度。2220是更新步数，不称20个epoch。

新增学生真实训练前向1136640＝2×2220×256，L/OE每臂曝光426240/142080。旧 L 教师精确复用时真实新 F0 前向0；若来源不符则阻止使用，不静默换教师。记录全部4440步 CE/KD/OE/rank、sL/sOE、域内配对数、梯度/clip及索引证据；不以源 loss 小或源 V 涨分替代测试收益。

## 冻结读出与三类测试

每臂冻结后仅用源 L/OE 的 unit(raw native160) 拟合自己的 known90/OE32 聚合密度及 normalizer，辅助读出 predicted_ratio；主读出始终为原生六 logits 的 s，FP64 稳定 logaddexp 计算。Gaussian 不用于主能量决策，不将辅助读出更高的目标分数用来更换主。V27000只在模型/密度冻结后校准两读出各六个预测身份阈值；新24值加旧24共48个FP64/384B，主六τ48B。主部署不需要新的 Gaussian 或 source/U 行 bank。

固定八读出：E0 pooled/predicted、OLD_PCD pooled/predicted、UNIFORM energy/predicted、RXPAIR energy/predicted。唯一主 RXPAIR_energy。原 E0/PCD 四读出与同原 F0 baseline 精确复用 fresh `f4b5a0b80cafcf237c3ae9dac8ecd5b24c91bff7`，新学生按自己的默认头路由自身 τ 与辅助类条件密度；2个百分点保护始终对同 DURA 原 F0，另报学生闭集。主 energy列0 配对 E0/OLD_PCD predicted列1、UNIFORM energy列0，不能复用同列比较冒充主能量。

三类主输入：源已知接收机星地、跨 RX clean＋DURA、跨 RX 星地＋DURA。源 V27000 是旧校准角色，不是独立已知测试；NV3150为19-15/19-16/7-1各1050，仅独立评分。目标开发 RX3-2/index34，future03-15/23为5525/4713，共10238条，已知4508、未知5730。未知5-5/10-7/2-19/7-11名单仅builder/scorer；后来7-11不在早日U，此行任何U新读0。03-23缺20-19/8-20，完整名单AK/CK/H/闭集保持N/A；clean只冻结测试。正式RX24-18不读，历史NV/Q曝光、单seed/RX、global physical history UNKNOWN保持。

DURA v1.0 R69，E20/4940、reuse/graph开/native关、FP32/TF32off；G10保持600km/25MHz/2.45GHz、residual/post_sync/noEQ/refSNR30dB/reflect、input RMS一次/output RMS无，源scene3001/channel4001、future3133/3134、receiver2027，不重抽观测。新学生评估前向126120＝2×(源42584＋两Q20476)，Q共享原 F0/G 各20476。新真实信号分支合计1262760，源教师旧cache/恢复P1及原模型训练成本继承；实际合成前向单列，墙钟不是B1延迟。

神经/密度先于V冻结、所有τ先于NV/Q冻结；全部8×3预测完成后，独立CPU scorer才连接源NV/clean/star三truth各一次，Utruth0，不训练或调整状态。报告逐真实TX宏AK/CK/RU/H、原F0和学生闭集、逐RX/day/scene/day×scene/TX、CI/AUROC/OSCR与同记录配对。主须source known12及校准通过、clean RU≥95%、star RU≥90%、CK≥90%、原F0同G闭集−AK≤2pp和全部必需主景完整；star AK须高于UNIFORM energy及OLD_PCD predicted、RU不低于E0/UNIFORM/OLD_PCD控制，才source_logit_energy_success=True，否则NO_PROMOTION。不扩seed或改主，adaptation_success=None/new_target_DA=False；AN/H_OW N/A，整个目标未完成。

## 最小执行约定

实际 commit、精确命令、环境/CWD与路径见同 run 登记。默认 N607，卡0至7实际free≥4096MiB、每卡最多新增2本任务worker，不干预其他用户/owner。单不可覆盖root、一次独立P0/P1、首真实checkpoint无query smoke PASS立即训练、单发布包一次SHA比较及远端compile、实际PID/CWD/argv/GPU/log增长读回。停止只因协议/祖先/输出冲突、非有限、无法执行或无法生成合法预测；低分不触发技术重跑。预期两CP与全步日志、source densities/τ/三条件预测、独立score、pipeline、同row报告登记和Git交付。

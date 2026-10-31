# 源接收机留一的联合身份与开放表征训练

Run `20261031-phase3-source-meta-s2026092701-r01`，唯一owner `codex-phase3-source-meta-20261031`。本轮为纯源域泛化候选，不含新目标端适应、发现或注册；原PCD研究基线、DURA默认、历史失败结果与完整目标ACTIVE保持。源Us重放/PL与统计混合已有负结果，原生身份自训练未达到联合目标；新增RX34的恢复前后配对又显示原F0在raw星地闭集39.689884%、DURA后41.721059%，因此本轮检验源端跨RX优化的训练目标，而非重调当前目标阈值。

方法参考[Li等的MLDG原论文](https://arxiv.org/abs/1710.03463)：用源域内部的训练/留出划分，优化一次虚拟源训练更新后的留出损失。本项目此前的TX成员集合episode和一阶冲突投影不含该虚拟更新。本行是自选损失、预算与超参数的RFFI变体，不声称复现论文图像实验或已获得泛化收益。

## 两臂、原生参数与数据

ERM_NATIVE与META_NATIVE均独立初始化为完整原F0 reader/default6（232993参数）加合规纯源r03 projection（102688），总335681、冻结Sinc48后可训练335633。原F0的完整身份初值须由真实合成smoke确认logits一致，初始开放z不承诺等价r03；不加载目标适应PCD初始化。原F0/r03/PCD/DURA及学生buffers不更新，学生reader非Sinc参数、原默认头和projection由外层优化器联合训练。不部署reader外667729参数、虚拟模型或新分类头。

只用source392005/VALIDATED_ONCE的既有星地DURA P1：L6300六旧类真标签，OE6134仅outside-six成员监督及公共RX/day、无OE真TX/K。源RX为1/3/4/6/8（1-19/18-2/19-2/2-19/3-19），日期2021-03-08/15/23，eq1；六旧TX为14-10/14-7/20-15/20-19/6-15/8-20。Us56700只保留Phase1祖先，本轮不读；目标U值/U IDs/Q、V/NV及truth训练读取均0。旧PCD仅作冻结比较，有原RX34 U576祖先，不向新学生提供初始化或教师。

每步held RX按[1,3,4,6,8]循环。其余四RX提供L128（各32）与OE64（各16）；held RX提供L64/OE32。合法L按源y/RX/day分层、OE只按公共RX/day覆盖源三日，防止原跨RX/day候选为空；不按未知真身份采样。四独立固定采样流2026103180/81/82/83，两个arm逐步全部索引完全匹配。重复曝光不计为新增物理样本；held只是当前训练划分，原F0与父r03已见过源五RX，不能声称严格未见接收机盲测。

## 确切元目标与计算

源训练F在原参数θ上计算L128/OE64，保持旧共同目标：crossset_metric＋OE_topology_KL＋10×balanced_anchor＋CE_L＋KD_L。anchor为0.5×L均值＋0.5×OE均值，父z来自冻结纯源r03，KD来自原F0缓存同P1 logits（T2、KL×4），CE/KD权重1。

计算θ′＝θ−α∇F。ERM的α=0，META的α=0.001；外层目标均F＋0.5G(θ′)，beta=0.5固定。META使用`create_graph=True`与stateless functional_call，保留二阶链式梯度，不detach虚拟梯度或候选表征，不把临时权重写回原模型。Sinc不参与虚拟更新，buffers保持eval冻结；不存在目标query内更新或额外部署参数。

G只对held L64/OE32的anchor平均，但旧跨RX/day关系损失需要其他接收机候选。因此在θ′下重新前向[trainL128,heldL64,trainOE64,heldOE32]共288行，held anchors的关系候选由另外四RX的train bank提供，沿用原不同RX且不同day掩码；G的CE/KD及balanced_anchor只取held角色。新core实现anchor行选择，all-anchor形式应与旧损失数值/梯度一致；不把单RX关系损失置零或静默改成同RX。source train bank和held都使用θ′，不能混用θ的候选。

两个arm都实际执行原参数192行＋虚拟参数288行，共480行/步，固定各444步/合计888；外层AdamW lr1e-4/wd1e-4/eps1e-8、普通normclip5，模型eval但保留梯度。数学上unused参数的虚拟梯度若为None应显式记录并保持该参数，不造梯度；真实浅层time_fuse/中层t2/defaulthead/projection路径仍须有限且有梯度。保存每步源/held各分量、外层总loss、虚拟step范数、二阶状态、梯度/clip、源RX划分与完整索引，不根据Q修改alpha/beta或预算。

新增学生训练前向426240＝2×444×480，其中base170496、virtual255744；父r03源L/OE缓存12434，原F0 L教师6300精确复用既有cache，训练新F0/G真实0。合成smoke单列。评估新学生126120，包含两臂各42584源行＋两Q各10238；Q共享原F0/G各20476。新增信号前向含父cache共564794，继承原F0/r03/PCD/G训练与旧缓存成本保留；墙钟不是B1延迟。

## 冻结评估和判定

每臂冻结后只用源L/OE拟合自己的known90/OE32密度及normalizer，V27000仅校准两臂pooled/predicted四组六τ（24新值）。与旧E0/OLD_PCD的24值合计48个FP64/384B；旧前四读出精确复用。新学生各自默认头决定closed，并路由自身密度/τ；保护基准仍原F0同DURA闭集，另存学生自身闭集。

完整8方法×3DURA条件：E0/OLD_PCD/ERM_NATIVE/META_NATIVE各pooled/predicted，唯一主META_NATIVE_predicted_ratio。三类为源已知RX星地、跨RX clean＋DURA、跨RX星地＋DURA，clean只冻结测试。源NV3150三设备仅独立评分；目标开发RX3-2/index34的future03-15/23共10238条（known4508、unknown5730），仍为既有canonical clean/star配对，不读取早日U576。目标四未知5-5/10-7/2-19/7-11名单仅builder/scorer，后者早U未见；03-23缺旧20-19/8-20，完整宏AK/CK/H/闭集保持N/A。正式RX24-18不读，已有NV/Q曝光、单seed/RX与global physical history UNKNOWN保留。

神经和密度先于V冻结，所有τ先于NV/Q冻结；全部8×3预测固定后独立CPU scorer才连接NV/clean/star三truth各一次。主报逐真实TX宏A_K/C_K/R_U/H、原F0和学生闭集、RX/day/scene/TX/day×scene、AUROC/OSCR/CI及相同ID配对得失。源known12与42诊断分别报告。主clean RU≥95%、star RU≥90%、CK≥90%、原F0同G闭集−AK≤2pp且各必需主景完整；主star AK须高于ERM和OLD_PCD，RU不低于E0/ERM/OLD_PCD，全部满足才source_meta_success=True；不通过NO_PROMOTION、不扩seed或换主。即使源DG通过，adaptation_success=None/new_target_DA=False，AN/H_OW N/A，完整目标仍未完成。

固定DURA v1.0 R69、reuse/graph开/native关、FP32/TF32off；G10为600km/25MHz/2.45GHz、residual/post_sync/noEQ/refSNR30dB/reflect、input RMS一次/output RMS无，源scene3001/channel4001/future3133/3134/receiver2027继承reference。没有重抽信道或clean参考恢复。唯一实际checkpoint无query smoke为launcher首步，验证原生logits初值、虚拟状态不写回、真实二阶反向与Sinc/buffers保持，PASS立即继续源训练。

实际commit/命令/环境CWD/路径/GPU见本run登记。默认N607、卡0至7实际free≥4096MiB、每卡新增最多2本任务worker，其他owner保持。单不可覆盖run/output/release，单包一次SHA比较、remote compile、PID/CWD/argv/GPU/log-growth读回。停止限协议/祖先/输出冲突、非有限、无法执行或无法合法产生prediction，不按科学低分停止；技术失败保留root后一次针对修复与独立新run，不盲重启。预期train两CP与全888日志、models/τ/source与targets全部预测、独立score、pipeline/stdout与同row报告及登记Git交付。

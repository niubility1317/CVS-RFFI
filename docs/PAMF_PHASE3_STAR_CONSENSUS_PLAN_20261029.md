# G10 源端多分支证据拒识器

run `20261029-phase3-star-consensus-s2026092701-r01`。本轮是完整 Phase3 目标内的单 seed 源端可证伪实验，不因源端通过宣称跨接收机星地、无标签适应或未来注册成功。上一轮原生源表征增量主候选的 A_K/R_U/H 为 90.7259/25.3968/39.6847%，未知拒识仍失败；VIC 项还损伤源未知设备的跨 RX 检索。新假设是：保持原 Phase1 身份几何，以源端多分支证据训练一个小型拒识器，比单一闭集 energy 更能识别未知，而不再改变未知身份表征。

考虑过只调整 scalar 阈值、继续改骨干、增加独立拒识器。选择第三项，原因是它显式隔离拒识学习与身份表征变化。主候选事前固定 P1/mlp16_oe；P0、linear_oe、msp/maxlogit/margin/energy/cosine 均为配对诊断，不能以其更高分替换失败主候选。所有 2×7 预测必须完成，之后独立 scorer 才连接 source novel truth。

唯一 F 为原三视图 Phase1 F0 E50/11100、232993 参数和完整六类默认身份头。全部参数/BN/buffer 冻结，原分类头原样预测；不继承上一轮任一增量模型。P1 使用前轮已完成、同物理 ID/episode 的 DURA v1.0 缓存，G 不重训、不重跑。复用 `20261029-phase3-star-representation-s2026092701-r01/cache` 的 L6300/OE6134/V27000/novel3150，均为 G10 MIX01 星地观测，P0/P1 不表示 clean/星地。旧六类与 source RX1/3/4/6/8、三源日、原 L/U/V 角色保持。原 U56700仅祖先，本轮新读0；target/dev/formal U/Q全部0，无新数据构建、重复信道或新增独立物理记录。

每 path 只用 L 拟合八 branch geometry，构造100维 head-aligned consensus descriptor；各 branch 含 cosine 与 Mahalanobis 摘要，结合完整六类 logits 的对齐证据，类别列置换不改变描述。L 与 OE 描述符只用于二元 source membership 监督，不读 OE TX/真实类别数，不将新增源监督写成零源标签。OE21原6134不均衡域保持，与旧 OE4 的537物理重叠继续披露。novel3 为19-15/19-16/7-1、各1050，同源RX/日期且已有历史开发暴露，不是新TX/未见RX盲测。

完全复用既有 role/domain-balanced source reader：linear101与MLP100→16→1共1633参数，各固定200次CPU全批AdamW更新（lr.001、weight_decay.0001、betas.9/.999、eps1e-8）；15个RX×day域在每role等权，loss known/OE先验.5/.5，normalizer固定.6/.4（继承原既有定义，不按novel估计）。无早停、候选扫描或目标反馈。两 path 共800次小型读出更新；F/head更新0，初始模型seed2026092701，split392005，readoutseed2026092701；无新随机增强/support seed。

known V 只在本 path 模型/geometry/readout冻结后校准7个全局scalar阈值，接受score≥τ，完整ties。42组（overall+5RX+6scene+30RX×scene）同时满足逐TX宏C_K≥90%、原F0同输入同path closed−open≤2pp，以Fraction FRR1/10和loss1/50精确检查。缺类或闭集损伤已超限时，全接受fallback并记科学false；不能以新模型自己的闭集掩盖原F0损伤。两 path 的全部状态和阈值冻结后才打开 novel IQ；novel真实TX/scenario仅独立scorer读取。目标真值、真实K、配额或query分数不用于阈值。

主报逐TX宏A_K、R_U及H=2A_KR_U/(A_K+R_U)，并列C_K、原F0closed、原F0closed−open、宏AUROC/OSCR、每TX/RX/scene与覆盖。主P1/mlp16_oe须同时通过42known保护和unknown overall/5RX/6scene≥90%，RX×scene未知只作诊断；缺必需类别为N/A且不能认证。不另设H≥90。无DA/注册执行，A_N/H_OW为N/A。源门槛通过仅获得下一独立冻结小规模跨RX验证资格，不自动扩多seed或读取目标。

执行顺序：实际原F0无query合成G10 smoke → L/OE描述与固定训练 → known V阈值冻结 → 全novel预测 → 独立CPU scorer → 同row结果与Git交付。smoke为launcher首步，PASS立即继续，无token/receipt。新run/output root不可覆盖，技术错误保留partial；科学失败仍完成全部预测/评分，不调参重跑。一次针对下一实跑的独立P0/P1审查，无完整重审或报告审批。REJECTED_EXTRA_GATE：独立设计审批、文件hash/receipt和报告完整性不成为发布阻碍，沿用户既有自主研究授权及项目最小流程执行。

N607 run root 为 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase3-star-consensus-s2026092701-r01`，release 为对应 `releases/<run>-release`；固定已验证CVS-RFFI Python。单GPU预测worker，启动时实际空闲≥4096MB，每卡本任务新增≤2，不停止或修改其他owner。CPU读出训练在该预测进程内执行，独立评分CUDA不可见。精确commit/配置/命令/CWD/GPU/PID与产物路径保存在同run登记；正式发布后独立读回PID/CWD/cmdline/GPU/log增长，完成后取回紧凑产物和完整日志，不提交IQ/权重/原始预测。

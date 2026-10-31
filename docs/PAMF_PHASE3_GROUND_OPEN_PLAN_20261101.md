# 先强化地面开放训练

Run `20261101-phase3-ground-open-s2026092701-r01`，owner `codex-phase3-ground-open-20261101`。用户指定先强化地面训练；原 PCD＋DURA 基线及所有历史产物保留。完整 Phase3 目标 ACTIVE，ViM 草稿尚未发布，本轮不继续该方向。

本轮固定两个完整原 F0 学生，检验地面 clean 源域训练及普通源域半监督能否改善开放识别。主方法预先固定 SSL_energy，SUP_energy 为配对监督控制，F0_energy 为原识别器控制；不依据查询结果选主方法、阈值、checkpoint 或重跑。没有新增目标无标签适应、未知发现或匿名注册。

## 数据与权限

源数据继承 split392005 的 VALIDATED_ONCE 合同。eq1、256 点 IQ，原六已知 TX：14-10、14-7、20-15、20-19、6-15、8-20。五源 RX index1/3/4/6/8 对应1-19、18-2、19-2、2-19、3-19，三日2021-03-08/15/23。源 L6300有六类真标签，Us56700隐藏 TX 标签，OE6134只给 outside-six 成员监督和公共 RX/day，不提供 OE 真 TX。所有训练输入均为未叠加星地信道的 clean IQ，经固定 DURA 得到 G(clean)，不使用旧 G10/P1 代替 clean。

源 V27000按六类×五RX×三日90格，每格300条；按 opaque-ID 排序和固定seed2026110101，每格随机150校准、150已知测试，总计CAL13500和TEST13500，物理互斥且与L/Us/OE互斥。CAL标签只用于已知接收保护，TEST标签只进入独立评分。源未知NV3150为19-15、19-16、7-1各1050，真实身份只在CPU评分侧打开，训练和阈值冻结不读。TEST与NV固定打散，构成16650条role-blind source Q；clean与星地共用同序物理ID。整个V和开发基准已有历史曝光，这次细分不声称新的独立盲测，不改变原F0祖先合同。

跨RX测试沿用fresh开发RX3-2/index34，future03-15/23共10238条，分别5525/4713；已知4508、未知5730。未知TX5-5/10-7/2-19/7-11仅builder/scorer知道，后者不在旧早日U。本轮目标U值、U ID、目标真值、目标clean训练均0。03-23缺20-19/8-20，完整六类指标记N/A，不以present-only宏均值冒充完整名单。正式RX24-18不访问。源未知和目标未知使用各自真实身份命名空间。

复用旧received观测：G10 MIX01，600km/25MHz/2.45GHz，residual/post_sync/noEQ/refSNR30dB/reflect，输入RMS一次、输出不再RMS；源scene3001/channel4001，future3133/3134，receiver2027。源星地从既有DURA缓存提取相同TEST/NV记录，不重抽信道。跨RX clean及星地同10238条，每条接入固定DURA再由学生预测；原F0同DURA logits精确复用合法旧缓存。

## 完整原生学生与配对预算

合规初始化为完整原F0 Tri-SIG-TIME SSL E50/11100的reader与原默认六类头，232993参数；Sinc48和全部buffers冻结，可训练232945，默认头960与骨干联合更新。不拼接r03骨干和F0头，不引入额外头、投影、EMA或目标模型教师。DURA v1.0 R69固定266951参数、E20/4940、reuse=True/graph=True/native=False，FP32/TF32off。主部署学生＋G共499944神经参数，外加六个FP64阈值48B，不部署教师或源行bank。

SUP、SSL各独立复制同一原F0。共同损失为L真实标签CE（LS0）＋原F0同G(clean) L logits的T2 KL×4（权1）＋同公共RX/day的mean softplus(sOE−sL)能量排序（权1）。s=logsumexp六logits，T1、高分接收。OE只使用成员监督，不使用真实新类身份。SSL额外普通源Us伪标签：本学生detached置信度≥0.95、CE权1，mask后除以完整128行，未接收行不产生伪标签CE。没有EMA和复杂门控；SUP不打开Us的ID、metadata或IQ。

两臂每步相同L192/OE64，六类L各32，OE覆盖15公共域，每条L至少一个同域OE；固定L2026103017/OE2026103018采样流逐步配对。SSL另加Us128，seed2026110102。SUP每步256行、SSL384行，各2220步、总4440步；额外Us计算单列，不称等前向成本。AdamW lr1e-4/wd1e-4/eps1e-8，clip5，FP32/TF32off；2220是更新步数，不称20epoch。固定最后步checkpoint，不用CAL/TEST/NV/目标选模。

两臂真实训练学生前向1420800：SUP568320，SSL852480；L/OE每臂曝光426240/142080，SSL Us曝光284160（唯一Us56700）。数据派生G99284行、F0教师36450行；教师包括L6300及V27000/NV3150的原F0评估logits，后两者不进入学生训练。预测新学生134552行＝2×CAL13500＋4×sourceQ16650＋4×crossQ10238；预测新G20476行，F0新前向0。合成无query smoke成本另报，记录所有4440步损失、伪标签接收/类别分布、梯度/clip和实际耗时；阶段耗时不是B1时延。

## 校准、四条件与验收

只用CAL13500已知记录校准，每个方法独立得到六个“预测身份”条件阈值；默认头自身argmax路由，始终计算全部六类。候选为已知分数的公共低尾分位，不借助NV/target；在overall、5RX、3day、15RX×day共24组保护宏CK≥95%、原F0同G闭集−AK≤2个百分点、六类覆盖。用固定有理数候选与单调搜索取满足保护的最大低尾。若零低尾仍失败或缺预测类，保存明确科学失败，不中止合法预测、不把失败改成通过。

预登记12点：F0_energy/SUP_energy/SSL_energy×source_clean_DURA/source_star_DURA/cross_clean_DURA/cross_star_DURA。源clean和源星地同16650条；跨RX两条件同10238条。所有学生先于CAL冻结，18个阈值先于全部Q冻结；12点完成后独立CPU一次连接knownTEST、sourceNV、crossclean、crossstar四truth，CUDA空，不读训练Us真值。producer公开Q不含真y/known标记，CPU评分不训练、不改阈值。

主报逐真实TX宏AK、CK、RU与H=2AK×RU/(AK+RU)，并报告原F0和学生闭集、闭集降幅、逐RX/day/RX×day/scene/day×scene/TX、CI、AUROC/OSCR和同记录配对。AK为已知类正确且被接收；CK为已知类被接收（含误分类）；RU为未知拒识。宏平均分别对所需六已知TX及三源未知/四目标未知TX等权，完整名单缺失标N/A。

source_ground_success要求SSL在源clean独立Q的24组全部CK≥95%、RU≥95%、原F0闭集−AK≤2pp且所需名单完整。cross_clean_success保留跨RX clean CK≥90%、RU≥95%、同F0保护2pp及必需主场景完整。ground_training_success为上述两者同时成立。源及跨RX星地作为保留诊断，仍检查CK≥90%、RU≥90%、同F0保护2pp及覆盖，不降低完整Phase3星地目标。SSL相对SUP增益如实报告，不另设胜过SUP的隐藏门槛。不通过即NO_PROMOTION，单seed不自动扩展；adaptation_success=None/new_target_DA=False，AN/H_OW=N/A，完整Phase3 ACTIVE。

## 执行与停止

实际commit、完整命令、Python/CWD、输入和不可覆盖路径、GPU及预期产物记录在同run登记/runtime。N607卡0至7启动前free≥4096MiB，每卡本任务最多新增2worker，不停止或修改其他用户/owner任务。单driver依次data→train→predict→CPUscore；每GPU阶段首真实checkpoint无query smoke PASS即继续。一次独立P0/P1、一次发布包本地/远端SHA比较、远端compile、实际PID/CWD/argv/GPU/log增长读回。

仅协议/模型来源不符、不可覆盖路径冲突、非有限、无法执行或不能产出合法预测构成技术停止；保存产物，不按低分重跑。产物为data公开角色及私有TESTtruth、两个CP/完整步日志、18阈值/校准、四条件完整预测、CPU score、pipeline、同row报告登记/Git交付。数据、权重、实际日志和运行时不提交Git。

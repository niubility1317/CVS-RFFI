# 新接收机固定 PCD 验证

VERIFIED / ANALYZED_DEVELOPMENT：真实新池构建、PCD444、全部六读出三DURA预测及独立评分完成；科学 NO_PROMOTION，完整目标 ACTIVE。

# Phase3 新开发接收机的发现池与固定PCD验证（2026-10-31）

Run `20261031-phase3-fresh-development-rx3-2-s2026092701-r01`。本轮构造开发接收机 3-2 的早日发现/未来查询数据，按已选择的 PCD 固定配方验证零目标标签适应，不再在旧 Q 上试新神经机制。所选 PCD 仅是研究起点；原 4af 冻结参考和既有所有负结果不改。正式 RX24-18 不访问，本轮不是正式盲测或三 seed 确认。

## 物理角色、标签和观测

Full eq1复用本机`F:/FULL wisig`及既有158成员PASS的VALIDATED_ONCE元数据，不重新扫描/重验Full。RX3-2官方索引34；它不在四compact的35接收机并集、source5、Phase2正式7、旧Phase3开发24-1或正式24-18内。现有85个Phase3登记未发现其query声明，但不据此声称全项目历史packet从未使用。builder报告保留这些具体角色排除和全局历史UNKNOWN，canonical定义不变，U/未来Q逐record唯一互斥。

已注册六TX及顺序固定14-10、14-7、20-15、20-19、6-15、8-20。新候选在排除旧六、源OE21、源novel3、正式未知4共34TX后，按两个未来日最小容量降序、未来合计降序、TX字典序排序，固定前三5-5、10-7、2-19为早日可发现设备，第四7-11为仅未来未知；这是元数据容量选择，不依据识别分数。上述名单/真实类数/配额只在私有builder/scorer配置，不作为模型输入。

U来自2021_03_01：六旧+三可发现设备各64，固定无放回选择，总576。future-Q为2021_03_15和2021_03_23这十TX的全部可用记录，预期5525+4713=10238：旧六4508、可发现三4320、仅未来未知1410。7-11早日虽有873条，但全部不进入U。未来Q不抽成有利子集、不重采样场景；03-23实际缺旧类20-19、8-20，完整六旧宏指标必须N/A；合并两日全类覆盖与逐景实际覆盖分别报告。

本机CPU builder只解码三个已验证的RX3-2 PKL成员，按其node_list对齐TX/data和原packet index。原FP64[N,256,2]按原索引取样后送G10；`star_channel.observe`内部FP32/一次unit RMS，输出不再RMS或EQ。每canonical只有一份星地received；单独为同future-Q、同ID/顺序生成FP32/一次unit RMS的clean测试控制，不生成clean U。public仅manifest/ids/received三个文件，真实TX/y/day/scenario等在private truth，opaque ID不携带类别/角色含义。

G10仍MIX01六分量面积权重/600km/25MHz/2.45GHz/residual/post_sync/noEQ/refSNR30dB/reflect，receiver2027；新episode固定scene2026103133/channel2026103134，selection2026103131/shuffle2026103135，均独立模型/类别/角色、不按覆盖重抽。clean与star不是更多独立物理数据，真实独立条数10814。源训练/重放/校准仍用已核实G10 P1缓存，不改变源场景。

## 固定方法与一次未来评分

预登记三模型六列：E0/SOURCE/PCD各pooled与predicted ratio，主唯一PCD_predicted_ratio，两个predicted控制均必报。E0为原4af r03 source-only读出；SOURCE精确复用body-prototype r01/a336 source-only444控制及r02/4273源GMM/τ，不能继承其TARGET_PCD作为新源初始化。新PCD独立从原source-only r03训练444，源192L/64OE采样逐步匹配旧SOURCE，目标U576全部无标签；不读取未来Q或旧Q预测作为训练输入。

沿用已固定PCD：Sinc48、所有buffers、原F0/defaulthead/G/r03冻结，只更新body231985+projection102688；AdamW1e-4/wd1e-4/eps1e-8/普通clip5。source crossset+OE_KL+10×父源z锚点，目标`.1*(global+local)`；固定两phase/gain强view，不重抽物理信道，弱全U576分配停止梯度，Ubatch64/seed2026103112、prototype容量16/seed2026103113/25Lloyd、aug2026103114、Sinkhorn20/eps.05/T.1、父最近其他record不变。16不是新真实类数，不加CCD、PL、EMA或身份头。新模型/由其拟合源聚合必须标U576祖先。

新PCD神经状态冻结后只以原源L6300/OE6134拟合known90/OE32/normalizer，以原V27000额外源标签拟合PCD两组六τ12新值；E0/SOURCE四组六τ24值精确复用，总36τ288B。source seenRX星地DURA、fresh future cleanDURA、fresh future starDURA三条件全部六列预测固定后，独立CPU连接原source NV3150及两future truth，各一次；Utruth0。目标四个未知的真实名单/数量只由scorer使用，模型从不据其路由。clean只冻结测试，两目标条件共享同一组源阈值，默认头闭集保护使用同一DURA输出。

当前旧RX24-1和新RX3-2结果均保留，不能用新unknown集合的较好成绩替换旧困难集合或声称普遍达标。最新人类v3仍为clean宏R_U≥95%、star≥90%、C_K≥90%、同DURA原F0闭集−A_K≤2pp，主报宏A_K/R_U/H；sourceknown12与所需覆盖保持。适应另要求主star A_K同时严格高于E0和SOURCE，R_U不低于两者。合并/逐RX/day/scene及每TX完整报告；缺格N/A不缩roster。未满足联合门槛不晋级或扩seed；发现及匿名注册本轮只准备合法数据，A_N/H_OW=N/A，完整goalACTIVE。

## 执行与成本

唯一owner root。同一固定代码commit：本机CPU不可覆盖pool构建→完成句柄→一次归档发布N607→新PCD训练→源fit/校准冻结→全预测→CPU truth-last。公共channel合成smoke先于真实Full读取，首次模型步骤真实三checkpoint noquery smoke先于source/U；PASS自动继续，无额外许可token/设计审批或数据重验。聚焦协议负测/一次独立P0P1、最小登记/一次N607preflight/单实际archive SHA比较compile及PID/CWD/argv/log证据保持。每GPU本任务新增≤2，本轮单train/predict worker顺序使用实际空余≥4096MiB，不动其他用户。

builder G10实际调用10814，future clean10238是同物理控制；无F/G训练。新训练父signal12434+576=13010，学生source113664+weak255744+strong56832=426240，G576/F0real0；PCD新增444，SOURCE旧444仅继承。评估newPCD source42584，三个open模型未来各20476，共104012 signal；F0/G各20476。新链signal543262/G21052/F020476（以实际计数校核），合成probe独列，源继承F011100/r03原2220/G4940及旧source控制开销不抹去。单主部署仍834665，临时16原型和source/U行bank不部署。

发布前固定完整矩阵、数据角色和科学判定；技术失败仅祖先/角色/冻结状态错误、非法数据、nonfinite、碰撞或不能闭合法预测时保存停止，科学低分不停止或省略future-Q。缺关键合法输入时如实保留状态，禁止回收历史Q、热改健康任务或同root覆盖重跑。全部完成后报告实际指标与未达项，Git提交/推送/OID读回。


实际代码 `f4b5a0b80cafcf237c3ae9dac8ecd5b24c91bff7`；本机 builder 环境 `C:/Users/lh594/.conda/envs/ssr-gpu/python.exe`，CWD `E:\type10-7\github_publish\CVS-RFFI-repo`。远端 CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-fresh-development-rx3-2-s2026092701-r01-release`；命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.fresh_world_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-fresh-development-rx3-2-s2026092701-r01-release/runtime.json`。矩阵与其他精确路径见 experiment.json。数据 builder 先完成再冻结公共句柄，单次发布含9个public文件与2个future truth，不传U truth。

## 完整结果与科学判断

source_DURA: 89.733333/96.748148/64.285714/74.907247%; 同G闭集损1.248148pp；clean_DURA: 31.661804/43.761391/61.979647/41.912794%; 同G闭集损16.905211pp；star_DURA: 27.951722/48.582412/49.307698/35.678111%; 同G闭集损13.769336pp。星地主 AK 高于 E0/SOURCE，但 R_U 分别低3.350826/2.962281个百分点，未知拒对分别净少190/168条；已知接收率、拒识率及闭集损失目标均失败。未晋级或扩seed，不按本次Q换主pooled。

实际 U576/双未来日 Q10238，共10814个唯一canonical；query旧4508/未注册5730，future-only7-11的1410条未进入U，03-23缺两个旧类，完整六旧AK/CK/H保持N/A。来源于新RX与新未知/日期/容量组合，不能和旧RX24-1结果作单因素因果比较，也不能替代旧困难集合；全局packet历史未用仍UNKNOWN。

完整444步、源SOURCE逐索引匹配、18运行点、逐TX/day/六景、macro与计数差别、后到未知诊断和成本见`docs/PAMF_PHASE3_FRESH_DEVELOPMENT_RESULTS_20261031.md`与私有analysis.json。源V校准非独立known测试；sourceNV/两Q各独立truth1次，Utruth0；原F0/defaulthead/G/Sinc/buffers保持。无匿名REG，AN/HOW N/A，正式RX未访问。

实际阶段秒数{"train": 83.716393, "predict": 35.859623, "score": 18.341892, "pipeline": 138.107009}（非B1时延）。独立live确认driver1601671/train1601716和predict1604856的CWD/fullargv/GPU3（UUID7b4bacb3），train1382MiB/predict580MiB是离散采样而非峰值；CPU1605578仅pipeline gpuNone/代码CUDAempty与完成产物，未捕获独立live，不继承其他run的CPU证据。最终owned空、foreign0。实际发布740个代码/运行配置文件及11个数据文件，单archive SHA/remote compile/runtime VERIFIED，无U truth传输。

本机私有辅助问题：首个监控因缺psutil返回失败，原builder健康继续并由Win32_Process+进程头复核后完整结束；初runtime构造漏view_family，模型前验证拒绝，补私有生成字段后通过。均未重启真实builder或修改生产代码/算法/预算；生产三阶段全部COMPLETE/0。

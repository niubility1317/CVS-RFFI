# 身份与拒识共享骨干联合适应

VERIFIED / ANALYZED_DEVELOPMENT：两新学生各444步、八读出三DURA及独立三truth评分完成；科学 NO_PROMOTION，完整目标 ACTIVE。

# Phase3 共享身份与拒识表征的固定验证（2026-10-31）

Run `20261031-phase3-joint-identity-s2026092701-r01`。上一轮 RX3-2 固定 PCD 的开放分支保持身份输出不变，无法纠正身份错误；新组合的原 F0 同 DURA 闭集也弱。此为研究动机，不证明某种新方案有效，也不据查询选择参数。本轮固定一个共享骨干学生方案，新增源身份 CE 与原默认头联合训练，保持既有 PCD 其余配方。Phase1/2 原方法、所选历史研究起点及全部负结果不改。

## 方法与初始化

原 F0/defaulthead、source-only r03、DURA R69 与旧 U 适配权重冻结。两新学生均独立严格复制 source-only r03 的 headless body232033＋projection102688，再恢复同源原 F0 默认头六类权重960和原生 cosine 分类计算；不得从旧 TARGET_PCD 初始化。学生总335681，固定Sinc48及全部buffers，联合训练body231985/projection102688/defaulthead960，共335633；TARGET另有16×160临时原型，不部署。保留三视图固定时间重排的父结构和所有特征机制，默认头不替换。

源每步192L/64OE，损失固定为旧`crossset_metric+OE_KL+10×父源z锚点`加`1×CE(student_native_logits_L,y_L)`。源CE只用合法L真实标签，OE仍仅原有成员蒸馏，V/NV/U_s/Q不参与神经训练。SOURCE_JOINT仅此源目标；TARGET_JOINT再加既有`.1×(U_global+U_local)`，不新加伪标签或目标身份CE。target全弱U576停止梯度分配、两phase/gain强view、K16容量/25Lloyd/Sinkhorn20 eps.05/T.1与父近邻保持；不是完整发现或匿名注册。

两臂各444步，普通AdamW lr1e−4/wd1e−4/eps1e−8、普通norm clip5，source seeds2026103017/18、Ubatch2026103112/proto2026103113/aug2026103114固定。SOURCE先完成再打开U，TARGET从相同纯源初值独立训练；全部444源索引匹配，TARGET U/augmentation与既有fresh PCD同构。所有原模型/Sinc/buffers无更新，默认头和body/projection各444更新。初始头来自F0但body来自r03，不宣称初始学生完整logits等于F0。

## 数据与评测

复用上一轮VALIDATED_ONCE数据：源L6300/OE6134、V27000额外源校准标签、源NV3150仅评分；早期RX3-2/03-01星地U576，双未来日03-15/23 Q10238，同ID clean/star各一份，旧4508/未知5730。目标全部学习只用星地，无clean U；未来专有7-11的1410条从U整类排除。真TX/K/配额仅builder/scorer可见。新run不生成新信道或扩大标签，不访问正式RX24-18。该Q已在前一开发run评分，非新盲测；未知集合/日期/容量差异不能与RX24-1作单因素因果比较。

固定四模型八读出：E0、OLD_PCD、SOURCE_JOINT、TARGET_JOINT各pooled/predicted ratio，主唯一TARGET_JOINT_predicted_ratio。E0与OLD_PCD的同源/同Q预测、密度和24个旧τ从fresh f4b5/r01精确复用，仅作控制；新学生两份known90/OE32密度和normalizer仅由原L/OE拟合，V校准四组六τ24新值，共48τ384B。新默认头身份须与自己的密度路由/τ索引一致，学生闭集与原F0闭集分别存储，不能标classification_unchanged。

测试包括source seenRX star_DURA、crossRX clean_DURA、crossRX star_DURA，使用同一固定DURA。新学生共享一次实际DURA输出；原F0在同输出上取得原闭集基准，旧控制保持其固定物理观测/原DURA版本和精度身份。全部神经、源密度与V阈值先冻，所有八列三条件完整预测后独立CPU连接原sourceNV及两future真值各一次，Utruth0。SOURCE和TARGET每列使用各自预测身份，分组与宏平均仍按必需真TX等权；03-23缺两个旧类的AK/CK/H保持N/A，四未知完整时RU独立可报。

门槛仍为clean宏RU≥95%、star≥90%、CK≥90%、原F0同DURA闭集−AK≤2pp、sourceknown12及必需场景覆盖。适应还要求主star AK严格高于SOURCE_JOINT和OLD_PCD，RU不低于E0/SOURCE_JOINT/OLD_PCD；报告每一比较，不凭弱控制或H微增晋级，不因源侧失败省略预登记Q。若联合门槛未全过，不扩三seed，科学NO_PROMOTION，完整目标ACTIVE。AN/HOW仍N/A。

## 执行与成本

唯一owner root；同一实际代码commit与不可覆盖run root。聚焦协议负测及一次独立P0/P1后登记；N607一次preflight、一次archive SHA比对/compile，第一模型步骤真实F0/r03/G无query smoke及学生默认分类梯度smoke，PASS后继续SOURCE→TARGET→全部预测→独立truth-last。复用未变public capsule/split与source handles，不重验Full或加额外签名/许可链。每GPU本任务新增≤2，本轮顺序worker、实际空余≥4096MiB，其他owner不变。

预算预期：新source学生113664、TARGET426240条次，加父source12434/U576，共训练signal552914；G576/F0真实0。评估两新学生各source42584+future20476，共signal126120，F0/G各20476；旧E0/OLD_PCD前向0。新链signal679034/G21052/F020476，两臂共888优化，旧对照训练与父F011100/r03原2220/G4940另列继承。合成probe和source密度2/新24τ统计成本按实际记录。单主部署共享学生335681＋G266951＝602632参数，不部署原型/source或U行bank；原F0另用于评估保护基准，不把评估时全控制内存当作单主部署成本。

技术停止仅非法来源/角色、冻结状态被改、nonfinite、输出碰撞或不能产生合法prediction；保留原产物，不同root不盲目重启，不动外来进程。科学低分完成固定评测并如实报告；本轮不依据Q重选主、阈值或预算。报告所有训练记录、逐TX/RX/day/scene、相对收益与闭集分解，Git提交推送并读回远端OID。


实际代码 `f540589c0478bed77a9fb4e664481c80ffe7eecc`，环境 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-joint-identity-s2026092701-r01-release`，owner `codex-phase3-joint-identity-20261031`，命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.joint_identity_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-joint-identity-s2026092701-r01-release/runtime.json`。固定矩阵与具体输入输出见 experiment.json。

## 完整结果与科学判断

source_DURA: 89.718519/95.888889/67.777778/77.219871%; 原F0闭集90.981481, 学生闭集91.581481, 原F0损1.262963pp；clean_DURA: 21.147817/34.325475/63.883173/31.776407%; 原F0闭集48.567015, 学生闭集42.228981, 原F0损27.419198pp；star_DURA: 21.357750/40.113129/52.561046/30.373484%; 原F0闭集41.721059, 学生闭集38.639796, 原F0损20.363309pp。主star相对SOURCE_JOINT的AK+0.664492pp，但RU−5.012763pp；相对OLD_PCD的AK−6.593973pp。绝对与相对门槛均未过，不替换研究起点或扩seed，不按Q改主/阈值。

原F0同G闭集、学生闭集与AK独立保存，各学生native预测进入自己的density/τ，原默认头没有替换、没有另训头。原模型/Sinc/buffers保持；新head/body/projection每臂各444。SOURCE无U、TARGET_U576，源索引全444匹配。源V27000参与校准非known盲测，全Q是已曝光开发数据；03-23缺两旧类宏AK/CK/H仍N/A，正式RX0、Utruth0，无匿名REG/AN-HOW NA。

两学生初值为r03 body/projection与原F0 W，其完整初始logits并不等原F0，且本轮无合并初值Q读出。因此SOURCE_JOINT对F0的退化不能唯一归因源CE，匹配TARGET-SOURCE才隔离本配方PCD差异。下一步可先作合法源端初始化与身份保真诊断，不能把本Q当调参集。完整888步/24点/三配对/分组与成本见`docs/PAMF_PHASE3_JOINT_IDENTITY_RESULTS_20261031.md`与私有analysis.json。

实际过程秒数{"train": 117.055821, "predict": 49.889152, "score": 22.929024, "pipeline": 190.067902}（非B1时延）。driver1639732/train1639753独立CWD/fullargv/CUDA5、UUIDef5f59f2/406→738→1382MiB离散观测；predict1641045仅pipeline GPU5/产物，未独立live；CPU1641570独立CWD/fullargv/CUDAempty核实。终态owned空、foreign0。751文件/data0归档SHA比较、remote compile/runtime VERIFIED，未重新读取Full或生成信道。

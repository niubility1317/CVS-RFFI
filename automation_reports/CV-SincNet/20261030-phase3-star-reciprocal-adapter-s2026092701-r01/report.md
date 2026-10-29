# PAMF Phase3 U池互近邻适应与两阶段匿名注册

## 预登记（2026-10-30）

Run `20261030-phase3-star-reciprocal-adapter-s2026092701-r01`，实际code `d3d8f5680fd72dc7ded272ffd1ded3143e490244`；唯一主reciprocal::DA1_REG1，self_pl为匹配控制，两臂都完整DA0/1×REG0/1。详细机制/参数/角色和停止规则见`docs/PAMF_PHASE3_STAR_RECIPROCAL_ADAPTER_PLAN_20261030.md`、对应config；矩阵唯一维护在experiment.json。

原F0 Tri-SIG-TIME SSL E50/11100/default6head/232993与DURA R69/266951、oldneighborhood源聚合prior(codecb58217de28eb2cb5536d2392aef268b79006877)和pairedMLP6τ全部冻结，不继承上轮relationOE失败F，不回读source IQ/逐样本features，不调readout/τ。sourceL6300/Us56700/V27000及OE成员监督为祖先，目标标签0不等于零源监督。真实head输入梯度及原logits、F/G参数冻结的合成G10checkpoint smoke首步、在U首次读取前执行。

合法G10 RX24-1 U384/2021-03-08用于适应/发现/注册，futureQ864/2021-03-15只逐样本预测；source RX1/3/4/6/8互斥。使用已有capsule/split/VALIDATED_ONCE，不重抽raw记录或信道，P0就是星地received、P1为同received的冻结G输出。receiver2027/scene2026103001/channel2026104001及原sixscene概率保持。算法无targetTX/真K/配额/Qtruth；正式RX24-18不读，已暴露开发RX/设备/Q保持开发身份。

两臂小g=160→32ReLU→160/10432参数、CPU seed2026092701同init、final0；unit空间预归一化残差L2≤.1，归一化后恢复原raw范数，原default6头不替换/重训。原U margin≥0且MSP.95为trusted；selfPL仅这些自身argmax，主对accepted nontrusted中≥3个trusted MNN10邻居且modalvote≥.8扩标签，其他U不强归旧六。FP64余弦/opaque-ID稳定ties；两臂全部pseudo在任何optimizer前固定。每臂fullU200步AdamW .001/wd.0001/clip5，CE伪标签+allU offdiagGram1，无增强/采样/EMA/早停；空mask显式0step且U/Q完全旁路g保持精确DA0，不以舍入差伪造适应收益。

DA0/DA1各U-only注册：两old margin均负的候选在当前mutual10NN图取连通分量，K可不同或0；min16、与trusted邻接点比例≤.05、交替两半中心距≤LOO95%半径；只导出center160/半径，不在Q回读U featurebank。REG先保留oldmargin≥0的原旧label，仅old拒识后允许anonymous ball，防旧类被匿名竞争覆盖。该性质不证明未来新类有效或later拒识，需独立Q。

全部g/registry/τ冻结先Q；共享一次U和逐批Q的原F0 P0/P1提取，两adapter顺序独立。全部2×4prediction完成后CPU scorer先核publicgate/finitebudget/jointoldgate/ID和任意K，再读取Utruth冻结四个独立匿名映射，全部map完成才Qtruth；trueK/映射不回算法。AN分母含全部真实到达设备，漏发现0；REG0 AN/HOW N/A，未匹配anonymous是acceptedwrong非reject，缺必需TX/scene覆盖N/A。

主报宏AK/RU/H，附CK/原同输入F0closed损、AN/laterRU/同DA REG旧保留/HOW、逐sceneTXRXday/CI及配对得失和U伪标签/聚类污染诊断。DA1REG0须绝对CK90/RU90/F0loss2pp及positiveoldAKgain，主DA1REG1须AN80/intendedCK90/laterRU90/oldprotect2pp/REGretain2pp且七主组覆盖；DA0未适应对照的绝对门槛只诊断，不额外要求其先达90才验证改进。source90不前置，不以control替主、不自动扩seed/formal。开发PASS仅下一独立确认资格，完整goal仍ACTIVE/finalfalse。

### 执行、存储、资源及成本

- N607 Python `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-reciprocal-adapter-s2026092701-r01-release`，唯一owner `codex-phase3-star-reciprocal-20261030`。
- 命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_reciprocal_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-reciprocal-adapter-s2026092701-r01-release/runtime.json`。
- 不可覆盖 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-reciprocal-adapter-s2026092701-r01`，predict config13公有字段，无truth/K/targets。单GPU predict/adapterworker内部两臂全部四态后独立CPUscore；truth和targets仅后者argv。actualfree≥4096MB卡0..7取最高，≤2新增own/GPU，本次1；foreign不停止/修改/重启。
- 源/目标F0和G只读，预期realF0full2×1248=2496+合成4，G1248+合成2，headgradient合成2另计。若两mask非空：g训练153600行+终态诊断768行，U/Q额外g/head各2496行；空mask按实际0更新/旁路，计数在summary/costs。原source/Gcache/Phase1祖先成本保留，不当独立新样本。
- 实际sourcecontainer存/每次evidence算四个3233读出，不报成只一份；F/G/四readouts512876参数，单adapter523308、同时双adapter533740，非NNgeometry/anchors/registries另计。每realrow baseline+两个adapted evidence共3次（空mask减相应次数），共14976 realMLP行forward与smoke8；无B1实测，不以批量耗时或显存采样预报加速。
- 预期launch/pipeline/configs/logs；predictsmoke、两adapter model/200steps/summary、U_evidence/pseudo/每armDAregistry/diag、Ufreezecomplete、all8predictions/complete；score scores/complete及observations。权重/IQ/逐行pred/rawlogs private，不提交Git。
- 技术fail保留partial，科学负分/emptyseed/K0不重启/重选τ/改K；UNKNOWN先核原PID/root，不重复提交或触碰foreign。

验证：集成48 passed, 1584 warnings, 39 subtests passed in 11.10s、10AST/实际runtimePASS；之后仅producer4定向PASS5.15s含1新增empty-mask精确旁路（与主suite3项重叠，不累加为独立全suite）。唯一独立直接P0/P1 PASS。一次preflight source原F0/聚合/公有capsule/split/G在位、未读UQ或truth值，8卡free11195–18793MB。单code/configcommit、单archiveSHA比较/remotecompile、真实checkpoint首步smoke、实际PID/CWD/argv/GPU/log增长及prediction全完后truthlast；不增签名/报告审批/重复数据验证。

## 实际完成：零目标标签适应、发现与未来匿名注册（2026-10-30）

VERIFIED / ANALYZED_DEVELOPMENT。两臂×四态完整预测和独立评分，唯一主reciprocal::DA1_REG1仍NO_PROMOTION。目标端trueTX/K/配额输入0、source逐行feature/IQ新读取0、F0/G/defaulthead/readout/6τ更新0；各10432参数g只由U384学习200步，所有U状态先Q864冻结。四个scorer匿名map全部用Utruth固定后才读Qtruth，模型未接触映射或truth。完整v3目标ACTIVE/finalfalse，正式RX24-18未访问。

| 臂/状态 | 旧类宏 A_O（A_K） | 旧类 C_O | 当前 R_U | H-score | 未来 A_N | H_OW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| self_pl/DA0_REG0 | 65.972222% | 88.368056% | 76.388889% | 70.799458% | N/A | N/A |
| self_pl/DA1_REG0 | 69.618056% | 93.402778% | 60.763889% | 64.890331% | N/A | N/A |
| self_pl/DA0_REG1 | 65.972222% | 91.493056% | 65.625000% | 65.798153% | 4.687500% | 12.308736% |
| self_pl/DA1_REG1 | 69.618056% | 96.006944% | 25.000000% | 36.788991% | 13.541667% | 23.399066% |
| reciprocal/DA0_REG0 | 65.972222% | 88.368056% | 76.388889% | 70.799458% | N/A | N/A |
| reciprocal/DA1_REG0 | 68.923611% | 93.229167% | 60.763889% | 64.587052% | N/A | N/A |
| reciprocal/DA0_REG1 | 65.972222% | 91.493056% | 65.625000% | 65.798153% | 4.687500% | 12.308736% |
| reciprocal/DA1_REG1 | 68.923611% | 93.229167% | 27.083333% | 38.886377% | 11.458333% | 21.628696% |

REG0 R_U分母为全部三未知6/7/8；REG1主R_U为U未出现的plannedlater8，不能直接以两态H/RU差表示注册影响。A_N分母始终两个真实U到达类，漏发现记0；REG0 AN/HOW N/A。REG1另报实际未注册集合和intended旧六+两到达类接收率，不能把old C_O93.23%当全应注册类C_K已达标：主intended C=78.385417%仍低于90%。

主DA1REG0较同Q DA0REG0：known正确380→397净+17，A_O+2.951389pp；接受509→537净+28，C_O+4.861111pp；unknown拒对220→175净−45，R_U−15.625pp。原F0同P1闭集420/576=72.916667%，主开放397/576=68.923611%，仍损3.993056pp>2。适应有旧类开放收益，但不满足未知保护或完整联合条件；控制DA1旧正确401/576，比主多4，不能替换固定主。

主DA1只有一个35条匿名cluster，Utruth显示全为到达类7、old污染0；另一到达类6未发现。独立futureQ类7只22/96正确、类6为0/96，所以AN22/192=11.458333%，不能报U纯度100%为注册成功。原DA0匿名cluster16条含14old+2new6；控制DA1 K3包含34纯new7、16纯old3及原16混合cluster，weightedpurity93.939394%同样不代表未来AN13.541667%达标。

两阶段oldgate已在实际Q实现同DA REG旧A_O损失精确0；主later拒对在DA1REG前后均26/96=27.083333%，本轮REG没有再损伤该量。later的大幅损失发生在适应：DA0 REG0为64/96=66.666667%，DA1为26/96。注册仍漏设备、未来识别率低，不能仅凭oldretain通过宣布开放系统有效。

trusted136条中128old+8unknown，old伪标签正确103/128=80.468750%；主扩15条均old，13正确2错误，且0条改变原argmax。主selected151中143old+8unknown、old标签正确116/143=81.118881%。这是全部预测冻结后的Utruth诊断，未回算法。扩展虽增加较干净的known样本，却没有纠正trusted错误，当前训练不证明真实身份学习改善。

完整400条训练步和全部3stdout已读取，每臂200fullU384=76800曝光，共153600；伪标签CE曝光control27200/main30200。全部loss/gradientfinite、损失分解一致，真实parameters_changed8832/10432，各max preNorm residual .1000000164为FP32舍入量，.1常量未改。control首末total .0153068006→.0008124282，main .0246448591→.0011712685；F0/head/G参数与buffer和sourcegeometry/readout/τ不变，训练loss下降不能替代joint目标。

source祖先仍原三视图SSL E50/11100/392005 L6300/Us56700/V27000及sourceOE成员监督。当前实际新IQ只有G10旧U384/Mar8和futureQ864/Mar15，同RX24-1、source五RX互斥、P0received/P1冻结DURA、同MIX01/receiver2027/scene2026103001/channel2026104001，无clean新路线/重抽物理/信道，不把1248既有记录当独立新数据或新盲测。targetQ高郊区缺某必需unknown类，按原覆盖规则N/A且不认证，不重抽样凑指标。

### 完成后的身份与路由分解（post-hoc，非预登记候选）

在全部预测/主评分固定后，另一个独立CPU同row分析只重算固定数组：实际m1+原τ[currentclass]还原readout raw，再用原F0预测类的原τ索引；也检查保持m1但只恢复原F0身份。无新模型/阈值拟合、NNforward、Utruth读取或匿名重匹配，主predictions/verdict完全不动。该诊断不作新盲测、晋级或重判本主结果。

主candidate闭集413/576=71.701389%，较原F0净−7（3得10失）；开放误拒再损16/576=2.777778pp，总3.993056=闭集1.215278+正确误拒2.777778pp。只恢复原F0身份、保留当前gate，AK403/576=69.965278%、C93.229167%、RU60.763889%、H65.040763%，原F0损2.951389pp仍失败。再恢复原class路由为AK69.791667/C93.055556/RU60.763889/H64.965647%，仍失败。控制相应samegateAK70.138889%、origrouteAK69.791667/RU60.416667，也未过。

按同一顺序，原身份下gate改变带来known正确380→403净+23，再更换适应后身份403→397净−6，最终+17；不是闭集身份性能改善。条件τ路由变化也不能单独修复unknownR下降，不能据诊断直接换成更高数字版本。该证据将下一科学问题从单纯head/routing错误收束到target域knownness漂移及对未知设备的保护。

### 成本、发布与可观测证据

F0 full forward2500=2496real+4synthetic，G1250=1248real+2synthetic；g训练153600+终态诊断768+UQ推理2496=156864行，额外原head特征前向153600+2496+2=156098。source四个3233快照都存/每evidence都算四个，real14976MLP行、smoke8，sourceL/OE/旧Gcache/Phase1祖先成本不抹去。神经实际同时双g533740、单g523308；geometry/90聚合/原型/radii/部署封装另计，无source逐行bank、无B1时延证据。

code d3d8f5680fd72dc7ded272ffd1ded3143e490244/prereg544a313006685fa0f45f82a8a06e44923af30b4c；48tests39subPASS11.10s，后producer4定向PASS5.15s含1新增（非另一完整suite），10AST/runtime与唯一独立直接P0P1 PASS。一次preflight/archiveSHA1638a485996cc10480a733f3a4bdf7f008e6770fd166b955dba1b7d23a7e4d08（387files/data0）比较/compile VERIFIED。实际F0/G syntheticG10 smoke先U PASS，head inputgrad4.5401769、head logits roundtrip差0，realrows0读/updates0。

仅driver3267295独立live抓到正确CWD/argv/CUDA空；predict3267384由pipeline分配GPU6、score3267483 CPU，仅jobs/stdout，未抓子进程瞬时独立GPU显存或live log增长，不为遥测重跑。终态live/ownedGPU空、foreign干预0、predict/score/driver日志1052/1574/21B（driver0→21已观察）；三个complete齐。pipeline23.247853s、predictjob18.405200s/score4.763123s，predict内部13.545977s，含加载/数学检查、非B1或因果并行提速。

完整逐类/场景/CI、400步训练窗口、K与污染、配对计数和posthoc表见docs/PAMF_PHASE3_STAR_RECIPROCAL_ADAPTER_RESULTS_20261030.md。当前不扩正式/seed、不更换控制、不调整旧阈值；后续独立机制需同时解决knownness的未知保护和新设备跨日泛化，不能把本轮oldgate结构改进或U纯cluster缩减成整个Phase3完成。

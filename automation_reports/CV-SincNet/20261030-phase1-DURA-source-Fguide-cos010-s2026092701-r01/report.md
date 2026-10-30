# DURA单阶段源F特征指导：cos0.1独立对照

- run_id：`20261030-phase1-DURA-source-Fguide-cos010-s2026092701-r01`
- group_id：`phase1-dura-single-stage-efficiency`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Explicitlyauthorized source F-guided,notF-free;GscratchE20/4940noGancestors;noVarbase+.1frozenFcleanfeaturecos;TXdirect0/target0;automatic7formal

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 授权、继承和匹配范围

用户明确回复“允许新增独立F指导对照”。科学权限已更新PROJECT_PROTOCOL的仅此分支例外；G从零/单阶段/不直接读TX标签，F训练源为合法7%Ls＋63%Us、原source-final50/11100。该F权重含源有标签知识，故方法明确为F-guided，不称F-free或完全无TX监督。默认DURA和原F-free成果保持。

原noVarAux的源目标与普通随机初始化/原生插值/E20/4940保留，只增加0.1×F特征余弦约束。Gsource的Ls6300/Us56700物理ID集合已与Fsource角色独立只读匹配；不读任何TX标签值，不重建或重验未变数据。加载F仍执行既有严格metadata/checkpoint/sourcecontract检查，不因新权限绕过来源禁令。

Fclean特征只缓存一次63000条，数值40.32MB，实际读取/前向/保存耗时和总准备时间单列；之后1260000次F(restored)前反向只到G，F参数/状态不变。F既有50轮训练预算明确披露，不能称无教师成本。G初始化祖先0，不继承任何已测试G；目标结果不选择父或回调系数。实际GPU启动仍需真实两Ls/Fcheckpoints特征到IQ非零梯度、G严格重载、Funchanged首步smoke PASS，随后直接继续。

全七景和独立truth-last自动闭合，原F50/defaulthead，不能挪用当前Tri-SIG-TIME SSL或Phase2/3成绩。损失特征一致不证明TX逐细节波形真实，仍检查source IQ与正式TX/RX/day/净纠错/clean伤害及完整成本；六景3pp目标不降低。

## 首步真实检查与实际启动

VERIFIED/RUNNING。用户明确授权后的执行commit 2cee2bec08165769f787b2aa85d1a8d97177844d，一次N607 preflight、单归档两端SHA和远端compile通过，唯一P0/P1审查PASS，8项聚焦检查通过。CWD=`releases/20261030-DURA-source-Fguide-cos010-r01`。实际原F50 checkpoint＋两Ls数据smoke验证F特征到restored IQ的有限非零梯度、G全部源梯度、G严格重载、F参数/状态不变及原默认头有限输出；不读TX/target。

CPUcontroller PID3956900，GPU4 worker PID3958132；独立/proc核实存活、CWD、完整argv和CUDA绑定，并从nvidia-smi匹配worker显存。训练日志由0增长到1627bytes，至少完成E3。源clean F缓存63000条/160D、40.32MB数值、耗时3.782270秒，物理ID按原G顺序写入；每轮避免重复计算同一source clean F目标，额外restored F前反向仍实际执行并计数，不能宣称F指导没有开销。

当前training F access=True、F_updates=0、直接TX labels=0，G祖先0；F既有source-final50/11100知识明确继承。F完整状态将在训练末再次核验；当前smoke通过不冒充训练末证明或正式性能。固定最后E20后自动source1800×7、target168000×7、匹配时延与独立truth-last；六景3pp目标继续ACTIVE，默认DURA保持。

## 完整源F指导训练与七景正式结果

VERIFIED/ANALYZED。20个epoch/4940更新、L126000/U1134000曝光完整；G从零一次联合训练、0祖先G/直接TX标签/V/target训练。源F clean缓存63000次、restored训练前向1260000次，F参数/状态不变、更新0、原分类头拟合0。执行commit2cee2bec08165769f787b2aa85d1a8d97177844d；真实G/F首步梯度/严格重载smoke与独立P0/P1审查均PASS，实际PID/CWD/命令/GPU/日志增长已核实。

完整训练1068.366秒，源准备4.881秒（含clean特征缓存3.782秒），新增G训练流程共1073.247秒/17.887分钟。266951参数、峰值allocated 1372.14MiB；同卡交替G/G+F B1/B128、存储/MAC/求解预算见results.json。共享卡时段不同，不能与F-free耗时直接推断因果倍率。F既有50轮/11100更新及源有标签知识明确复用，不能称F-free或完全TX无监督。

source1800×7和target168000×7固定完成后独立CPU truth-last，再从不可变logits/truth独立复算126个总体/TX/RX/day分组一致；controller/worker自然结束。原F50/defaulthead配对、单seed，不扩展到当前Tri-SIG-TIME SSL/Phase2/3。

|场景|准确率%|相对匹配noVar pp|相对默认DURA pp|距raw clean pp|
|---|---:|---:|---:|---:|
|clean|84.904167|+0.161905|+0.226190|-0.227976|
|practical_high|80.698214|-0.205952|-2.126786|3.977976|
|practical_mid|79.301786|-0.104762|-2.258929|5.374405|
|practical_low_suburban|77.837500|-0.031548|-2.220833|6.838690|
|practical_high_urban|79.730952|-0.267857|-2.014881|4.945238|
|practical_mid_urban|73.745833|-0.647024|-1.740476|10.930357|
|practical_low_urban|67.827381|-0.643452|-1.062500|16.848810|

六个星地均低于匹配noVarAux对照，lowurban−0.643452pp到67.827381%，相对默认低1.062500pp。clean84.904167%，较raw多383条正确（596纠错/213损伤）；不能用clean提高代替星地验收。六星地距raw clean3.977976至16.848810pp，逐景3pp=False，原R17逐景最多1pp容忍=True。不更换默认。

完整源E20 IQNMSE0.02539897、余弦误差0.01134746、末轮裁剪0；源F特征改善未转为正式星地改善。已在本目标分数读取前完成source-task-geometry诊断并否决简单升至beta0.3，其源误差降低约26%但clean-F选择一致率改善较小，后续若检验原头可见的特征方向只以该源证据固定方案。保留全部负结果，不选中间轮次、不按target类别/场景调参或选择性重跑。目标继续ACTIVE。

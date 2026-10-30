# DURA单阶段源F特征指导：cos0.1独立对照

- run_id：`20261030-phase1-DURA-source-Fguide-cos010-s2026092701-r01`
- group_id：`phase1-dura-single-stage-efficiency`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

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

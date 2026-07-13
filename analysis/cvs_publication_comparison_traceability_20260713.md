# CVS论文级对比实验追踪表

## 目标

在统一CVS数据协议、训练预算和评价口径下比较CVS与指定论文方法，并形成可用于论文发表的可复核证据。论文原始复现和CVS扩展必须分离；监督式MRIOR-SDA、DADDA-SDA及CIL方法的Stage2-C版本统一标记`cvs_extension=true`。

|ID|来源|要求|目标文件/产物|状态|验证|备注|
|---|---|---|---|---|---|---|
|PUB-01|用户|Phase1比较CVS、CVCNN-CE、RIEI-FD、DRIFT|Phase1矩阵、launcher、报告|implemented|现有三个baseline入口可达；CVS候选仍在运行|最终统计需选定CVS主方法后同seed重跑|
|PUB-02|用户|CVCNN使用最常用结构和参数|`baselines/cvcnn_ce`及锁定配置|verified|三层ComplexBlock`32-64-128`、128维embedding、CE、AdamW`2e-4`、batch64、200epoch|Sinc、伪标签和sat训练增强默认关闭|
|PUB-03|用户|Phase2域适应统一使用带标签target-old support|CVS监督适配协议|pending|待实现统一support manifest和监督入口|query不得参与训练/选模|
|PUB-04|用户|ProtoNet CDA监督域适应|`paper_reproduction/protonet_cda`+CVS adapter|implemented|现有K-shot prototype/eval路径|需统一base checkpoint、support索引和LEO query|
|PUB-05|用户|MRIOR改为监督域适应|`paper_reproduction/cvs_aligned/mrior_sda.py`|pending|待测试target CE和真实标签优先级|不得修改paper-faithful UDA声明|
|PUB-06|用户|DADDA改为监督域适应|`paper_reproduction/cvs_aligned/dadda_sda.py`|pending|待测试target CE和真实标签LMMD|不得修改paper-faithful UDA声明|
|PUB-07|用户|Phase2新类比较CSIL|CSIL CVS adapter|pending|待实现Stage2-C support/query路径|原实现是ADS-B闭集CIL|
|PUB-08|用户|Phase2新类比较MoPC-HR|MoPC-HR复现与CVS adapter|blocked|当前只有PDF/官方代码登记|需本地实现和验证后才能启动|
|PUB-09|用户|Phase2新类比较Orthogonal Incremental|Orthogonal CVS adapter|pending|原始WiSig闭集FSCIL已可运行|需同一`R_t`旧/新类Stage2-C适配|
|PUB-10|`项目.md`|Phase1严格source-only|split manifest和训练命令|pending|最终矩阵验证|禁止目标receiver/query参与训练或选模|
|PUB-11|`项目.md`|Phase2`R_s/R_t`不相交且`Y_old/Y_new`互斥|Stage2 manifest|pending|协议validator|使用ManySig target-old+ManyTx target-new并按receiver label对齐|
|PUB-12|`项目.md`|简化LEO残余信道是主结果|`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`|implemented|baseline evaluator已支持这些scenario|clean单独作为control|
|PUB-13|发表要求|公平训练/适应预算|预登记协议、配置|pending|矩阵审计|同checkpoint、support、query、K、seed、更新参数/步数预算|
|PUB-14|发表要求|至少多seed并报告不确定性|结果聚合器|pending|待实现|主表计划5seed；先运行seed713101启动锚点|
|PUB-15|发表要求|统计显著性与效应量|统计脚本/报告|pending|待实现|paired seed/bootstrap CI、校正后的检验、效应量|
|PUB-16|发表要求|完整损失与final-vs-best证据|logs、metrics、loss trace|pending|完成后全日志解析|不得以启动PASS替代完成|
|PUB-17|AGENTS|本地验证、Git、报告、sync和N607清理|报告、commit、manifest|pending|逐阶段更新|根目录非Git，发布面为`github_publish/CVS-RFFI-repo`|

## 论文级主结果预登记

- Phase1主指标：strict UDU、receiver floor、三种简化LEO视图准确率及satellite floor。
- Stage2-B主指标：target-old full accuracy、accepted accuracy/coverage、`old_acc_delta_pp`、harm/rescue/net gain、更新成本。
- Stage2-C主指标：`old_acc`、`seen_new_acc`、`H_old_new`、old/new双向混淆、遗忘、存储、延迟和可训练参数量。
- Phase3 unknown指标不进入Phase2主排序。

## 当前最高风险

MoPC-HR尚无本地复现代码；在该方法完成论文逐项实现和CVS适配前，类增量主表不完整。

# CVS论文级对比实验追踪表

## 目标

在统一CVS数据协议、训练预算和评价口径下比较CVS与指定论文方法，并形成可用于论文发表的可复核证据。论文原始复现和CVS扩展必须分离；监督式MRIOR-SDA、DADDA-SDA及CIL方法的Stage2-C版本统一标记`cvs_extension=true`。

|ID|来源|要求|目标文件/产物|状态|验证|备注|
|---|---|---|---|---|---|---|
|PUB-01|用户|Phase1比较CVS、CVCNN-CE、RIEI-FD、DRIFT|Phase1矩阵、launcher、报告|implemented|现有三个baseline入口可达；CVS候选仍在运行|最终统计需选定CVS主方法后同seed重跑|
|PUB-02|用户|CVCNN使用最常用结构和参数|`baselines/cvcnn_ce`及锁定配置|verified|三层ComplexBlock`32-64-128`、128维embedding、CE、AdamW`2e-4`、batch64、200epoch|Sinc、伪标签和sat训练增强默认关闭|
|PUB-03|用户|Phase2域适应统一使用带标签target-old support|CVS监督适配协议|pending|待实现统一support manifest和监督入口|query不得参与训练/选模|
|PUB-04|用户|ProtoNet CDA监督域适应|`paper_reproduction/protonet_cda`+CVS adapter|implemented|现有K-shot prototype/eval路径|需统一base checkpoint、support索引和LEO query|
|PUB-05|用户|MRIOR改为监督域适应|`paper_reproduction/cvs_aligned/supervised_da.py`|verified|target CE、DV-KL梯度和support-only manifest测试通过|未修改paper-faithful UDA声明|
|PUB-06|用户|DADDA改为监督域适应|`paper_reproduction/cvs_aligned/supervised_da.py`|verified|target CE、真实标签LMMD、动态alpha和梯度测试通过|未修改paper-faithful UDA声明|
|PUB-07|用户|Phase2新类比较CSIL|`paper_reproduction/cvs_aligned/class_incremental.py`|implemented|真实tensor机制测试通过，N607数据smoke待跑|CVS扩展，不冒充ADS-B原论文结果|
|PUB-08|用户|Phase2新类比较MoPC-HR|MoPC-HR核心+CVS adapter|implemented|论文公式逐项核对、6项核心测试及CVS tensor机制测试通过|N607数据smoke待跑|
|PUB-09|用户|Phase2新类比较Orthogonal Incremental|`paper_reproduction/cvs_aligned/class_incremental.py`|implemented|真实tensor机制测试通过，N607数据smoke待跑|共享同一`R_t`旧/新类manifest|
|PUB-10|`项目.md`|Phase1严格source-only|split manifest和训练命令|pending|最终矩阵验证|禁止目标receiver/query参与训练或选模|
|PUB-11|`项目.md`|Phase2`R_s/R_t`不相交且`Y_old/Y_new`互斥|Stage2 manifest|pending|协议validator|使用ManySig target-old+ManyTx target-new并按receiver label对齐|
|PUB-12|`项目.md`|简化LEO残余信道是主结果|`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`|implemented|baseline evaluator已支持这些scenario|clean单独作为control|
|PUB-13|发表要求|公平训练/适应预算|预登记协议、配置|pending|矩阵审计|同checkpoint、support、query、K、seed、更新参数/步数预算|
|PUB-14|发表要求|至少多seed并报告不确定性|`tools/analyze_publication_comparison.py`|implemented|确定性bootstrap mean/paired-delta 95%CI测试通过|正式证据仍需5seed完成结果|
|PUB-15|发表要求|统计显著性与效应量|`tools/analyze_publication_comparison.py`|verified|exact paired sign-flip、paired dz和Holm校正测试通过|整体检验将在完整矩阵后按指标分组补充|
|PUB-16|发表要求|完整损失与final-vs-best证据|logs、metrics、loss trace|pending|完成后全日志解析|不得以启动PASS替代完成|
|PUB-17|AGENTS|本地验证、Git、报告、sync和N607清理|报告、commit、manifest|implemented|协议提交`271e50a/c93774c`；Phase1启动与SSH清理已记录|Phase2代码尚待提交/同步|
|PUB-18|用户/`项目.md`|正式测试样本全部叠加简化LEO星地信道|三场景配置、scenario/seed/启用证据|implemented|Phase1 evaluator及Stage2-C配置强制三种`leo_*_weak`|clean只能分表control，不进入主统计|
|PUB-19|用户/`项目.md`|每个实验输出逐receiver/逐transmitter详细结果|sample score table、四层group CSV/JSON、稀疏confusion|implemented|新增Phase1 sat detailed evaluator测试和Stage2-C detailed breakdown测试|当前已启动Phase1需用终局checkpoint补跑新详细评估|

## 论文级主结果预登记

- Phase1主指标：strict UDU、receiver floor、三种简化LEO视图准确率及satellite floor。
- Stage2-B主指标：target-old full accuracy、accepted accuracy/coverage、`old_acc_delta_pp`、harm/rescue/net gain、更新成本。
- Stage2-C主指标：`old_acc`、`seen_new_acc`、`H_old_new`、old/new双向混淆、遗忘、存储、延迟和可训练参数量。
- Phase3 unknown指标不进入Phase2主排序。

## 当前最高风险

监督DA仍缺少正式数据runner；已启动的Phase1进程在详细统计代码同步前启动，必须对其终局checkpoint补跑同scenario、同seed的satellite detailed evaluation，不能直接把现有overall-only artifact用于论文主表。

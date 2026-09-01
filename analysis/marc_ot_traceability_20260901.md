# MARC-OT设计落地追踪

设计来源：用户提供的MARC-OT/Meta-SF-RDC报告、`项目.md`、既有WISER-P3/D92/Meta-adapter实现与历史实验结果。

|ID|来源章节|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|PROTO-01|协议边界|只读合法support，query/truth/role/quota不可达|MARC-OT runner/pilot/tests|pending|待TDD|复用WISER loader与truth-last生命周期|
|BANK-01|设计4|稳定block schema与base checkpoint绑定|`meta_weight_bank.py`|pending|待TDD|禁止Sinc/domain/head/buffer进入首版bank|
|BANK-02|设计4|提取、压缩、重构分块weight delta|`meta_weight_bank.py`|pending|待TDD|未允许状态bitwise不变|
|BANK-03|设计4|bundle round-trip和禁止成员检查|`meta_weight_bank_checkpoint.py`|pending|待TDD|Phase2加载后bank冻结|
|ENC-01|设计5|permutation-invariant support set encoder|`meta_support_set_encoder.py`|pending|待TDD|同一物理样本增强不增加`n_eff`|
|CAL-01|设计5/6|预测`q/u/gamma/eta`并有界组合bank初始化|`meta_weight_calibrator.py`|pending|待TDD|失败整体回退`theta_0`|
|META-01|设计7|复用FOMAML并让outer梯度进入新元参数|`meta_bank_inner_loop.py`、`meta_bank_trainer.py`|pending|待TDD|保持一阶路径|
|OT-01|设计6|support与冻结bank/摘要之间的合法OT|`stage2_marc_ot.py`|pending|待TDD|禁止跨query OT|
|LOSS-01|设计6|冻结head、cross-fit、LOO、SupCon与统计校准|`stage2_marc_ot.py`|pending|待TDD|temporary prototype不持久化|
|GRAD-01|设计6|按block独立主任务优先梯度投影|`stage2_marc_ot.py`|pending|待TDD|替换WISER全局dot，不把cosine诊断冒充投影|
|SAFE-01|设计6|渐进开放、support-only早停和`alpha=0`精确回退|`stage2_marc_ot_runner.py`|pending|待TDD|恢复参数、dual和buffer|
|MATRIX-01|设计8|R0/R1/R2/R4/R6/R8最小矩阵|`stage2_marc_ot_pilot.py`、config|pending|待TDD|单seed三场景先证伪|
|MRIOR-01|设计3/8|新增H/B/HB控制接口且不倒填历史结果|pilot/control config|pending|待TDD|宽权限控制需显式标注|
|SCORE-01|设计8|REG0绝对指标、同row增量、per-class/help-harm/资源|`stage2_marc_ot_scoring.py`|pending|待TDD|独立truth-last|
|CLI-01|设计9|不可覆盖run root与正式CLI入口|script/config/tests|pending|待TDD|不读取query选择状态|
|VERIFY-01|设计10|聚焦测试、compile、CLI help、真实checkpoint无query smoke|tests/report|pending|待执行|本地优先|
|REVIEW-01|最小流程|一次独立P0/P1审查，定点复审最多一次|审查报告|pending|待执行|禁止额外gate|
|RELEASE-01|最小流程|Git提交、push、远端OID核验、release SHA和远端编译|run report|pending|待执行|不覆盖既有run|
|PILOT-01|设计8|三场景prediction闭合和独立评分|run report|pending|待晋级流程|低性能只是不晋级|
|T25-01|设计8|pilot通过后才运行Target25|后续run|blocked|等待PILOT-01|不属于首轮软件阻断项|
|STAGEB-01|协议边界|Stage2-C默认冻结`phi_D`并另训`phi_R`|后续设计|blocked|等待阶段A确认|本轮不实现|

初始统计：`pending=18`、`blocked=2`、`verified=0`、`deferred=0`、`rejected=0`。

当前最高风险是`META-01`与`CAL-01`的接口闭合：现有Meta-adapter虽有真实FOMAML episode和模块级Meta-SGD，但没有显式weight delta bank、support set encoder或任务条件化block gate/LR。首轮不得把旧activation adapter或IQ operator bank重命名为MARC-OT能力。

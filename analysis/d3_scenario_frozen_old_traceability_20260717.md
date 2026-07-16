# D3 scenario-frozen-old incremental head追踪

日期：2026-07-17

|ID|Source section|Requirement|Target files|Status|Verification|Notes|
|---|---|---|---|---|---|---|
|D3-01|`项目.md`9.2、9.3、10.3.1|Stage2-B旧类适应与Stage2-C新类注册同row执行，注册后保留旧类决策面|`code/cvsrffi/stage2_diag_cosine_exploration.py`；row pipeline|local_pass_remote_pending|focused pytest；development K10 new5/10/20|pipeline已把同row before COMMIT/root传给after；Stage2-C只训练新增类参数|
|D3-02|`项目.md`7.1、7.2、7.3|只读取密封`LEO_weak` target support/query；无clean/source/query truth/role/quota/global assignment|同上；execution receipt tests|local_pass_remote_pending|协议字段和无query fit接口测试|parent receipt的完整Phase2协议字段与query-fit资源字段已fail closed核验；场景metadata不是query类别角色|
|D3-03|`项目.md`10.3.1|三个LEO场景使用对应场景support head，query逐样本对全部注册类决策|同上|local_pass_remote_pending|scenario dispatch与全注册类预测测试|场景来自密封LEO scenario槽；若实际部署无法观测场景metadata，不得称无场景Oracle|
|D3-04|`项目.md`10.3.1|开发只使用receiver`20-1`、seed`713101`、K10选择统一candidate|development launcher/report|pending|manifest及命令审计|new5/10/20不得分别调参|
|D3-05|`项目.md`10.3.1|参数不超过50k、每场景20epoch、状态不超过256KB、无dense query图|resource receipt；tests|local_pass_remote_pending|实际N607资源审计|准确报告`epochs_per_scenario=20`、`total_epoch_passes=60`、真实optimizer updates和最终NPZ实测字节；MAC字段明确为不含backbone/FFT/RF的head范围|
|D3-06|`项目.md`10.3.1|保存完整loss trace、不可变prediction、COMMIT和独立scorer|diag runner；row pipeline|local_pass_remote_pending|artifact SHA/COMMIT测试|truth只在before/after不可变prediction完成后由独立scorer读取|
|D3-07|预登记推进门|三个new规模C-old/floor不低于D1，且每个规模`C-old - B-old >= -0.5pp`；new下降不超过2pp，new5/new10至少提升1pp|development report/scorer|locked_before_result|同row结果表|原“forgetting至少改善3pp”对D1 new5仅1.67pp的基线不可满足，已在D3结果产生前修正；失败即淘汰，不进入确认矩阵|
|D3-08|`AGENTS.md`版本与N607规则|本地先改、`ssr-gpu`验证、Git提交、再SCP和远端验证|Git/report/sync record|pending|git diff/tests/hash/preflight|保留无关工作树改动|
|D3-09|用户追加要求；`项目.md`10.3.1|同时优化每个旧类floor，不能用overall平均掩盖弱类|incremental support audit；scorer/report|local_support_pass_query_pending|逐旧类margin/intrusion测试；逐TX开发结果|逐场景逐旧类support intrusion已为0；query floor仍必须由独立scorer验证，重点跟踪`20-19`、`6-15`、`14-7`|

最高风险项：Stage2-C继承Stage2-B state必须由同row不可变COMMIT绑定，不能接受外部未绑定prior state路径。

实现定位：严格设计 parity，不以confirmation query调参；若实际代码只能实现近似方案，必须在进入N607前将对应行改为`deferred`或`rejected`。
